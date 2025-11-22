import os
from layers import *
from utils import *
import time
from tqdm import tqdm
from collections import OrderedDict
import numpy as np
from torch.optim.lr_scheduler import _LRScheduler
import torchvision.utils as vutils
import torchvision.models.vgg as models
import torch
import random
from Gan_Dataloader import train_loader,test_loader

class WarmupCosineAnnealingLR(_LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_iters = max_iters
        self.cur_iters = 0
        self.lrs=[]
        super(WarmupCosineAnnealingLR, self).__init__(optimizer)


    def get_lr(self):
        if self.cur_iters < self.warmup:
            return [base_lr * self.cur_iters / self.warmup
                    for base_lr in self.base_lrs]  # linear warmup strategy
        else:
            return [base_lr * (1 + math.cos(math.pi * (self.cur_iters - self.warmup) / (self.max_iters - self.warmup))) / 2
                    for base_lr in self.base_lrs]  # cosine decay

    def step(self, epoch=None):
        self.cur_iters += 1
        super(WarmupCosineAnnealingLR, self).step(epoch)
        self.lrs.append(self.get_lr())

class ImagePool:


    def __init__(self, pool_size):
        self.pool_size = pool_size
        self.images = []
        self.num_imgs = 0

    @torch.no_grad()
    def query(self, images):

        if self.pool_size == 0:
            return images

        return_images = []
        for image in images:
            image = image.unsqueeze(0)  # keep batch dimension of 1

            if self.num_imgs < self.pool_size:
                self.images.append(image)
                self.num_imgs += 1
                return_images.append(image)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    old_image = self.images[idx].clone()
                    self.images[idx] = image
                    return_images.append(old_image)
                else:
                    return_images.append(image)

        return torch.cat(return_images, 0)
class CXLoss(nn.Module):
    def __init__(self, sigma=0.1, b=1.0):
        super(CXLoss, self).__init__()
        self.sigma = sigma
        self.b = b
        self.eps = 1e-8  # numerical stability

    def center_by_T(self, featureI, featureT):
        """Subtract mean of target features from both input and target"""
        meanT = featureT.mean(dim=(0, 2, 3), keepdim=True)
        return featureI - meanT, featureT - meanT

    def l2_normalize_channelwise(self, features):
        """Normalize feature maps channel-wise"""
        norms = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features / (norms + self.eps)
        return features

    def patch_decomposition(self, features):
        """Decompose feature map into patches"""
        N, C, H, W = features.shape
        assert N == 1, "Batch size must be 1 for CX computation."
        P = H * W
        patches = features.view(1, 1, C, P).permute(3, 2, 0, 1)
        return patches

    def calc_relative_distances(self, raw_dist, axis=1):
        """Compute relative distances with stability"""
        div = torch.min(raw_dist, dim=axis, keepdim=True)[0]
        relative_dist = raw_dist / (div + self.eps)
        return relative_dist

    def calc_CX(self, dist, axis=1):
        """Compute contextual similarity"""
        W = torch.exp((self.b - dist) / (self.sigma + self.eps))
        W_sum = W.sum(dim=axis, keepdim=True)
        CX = W / (W_sum + self.eps)
        return CX

    def forward(self, featureT, featureI):
        """
        Contextual loss between target (featureT) and inference (featureI)
        """
        # Center and normalize
        featureI, featureT = self.center_by_T(featureI, featureT)
        featureI = self.l2_normalize_channelwise(featureI)
        featureT = self.l2_normalize_channelwise(featureT)

        dist_list = []
        N = featureT.size(0)

        for i in range(N):
            featureT_i = featureT[i:i+1]
            featureI_i = featureI[i:i+1]
            featureT_patch = self.patch_decomposition(featureT_i)

            # Cosine similarity
            sim_i = F.conv2d(featureI_i, featureT_patch)
            dist_i = (1. - sim_i) / 2.0  # convert similarity to distance
            dist_list.append(dist_i)

        raw_dist = torch.cat(dist_list, dim=0)
        relative_dist = self.calc_relative_distances(raw_dist)
        CX = self.calc_CX(relative_dist)

        # Aggregate contextual similarities
        CX = CX.amax(dim=(2, 3))  # max over H and W
        CX = CX.mean(dim=1)       # average over channels

        CX_loss = -torch.log(CX + self.eps)  # avoid log(0)
        CX_loss = torch.mean(CX_loss)

        return CX_loss
class VGG(nn.Module):
    def __init__(self, pool='max'):
        super(VGG, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]


"********************************************"
class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        features = models.vgg19(pretrained=True).features
        self.relu1_1 = nn.Sequential()
        self.relu1_2 = nn.Sequential()

        self.relu2_1 = nn.Sequential()
        self.relu2_2 = nn.Sequential()

        self.relu3_1 = nn.Sequential()
        self.relu3_2 = nn.Sequential()
        self.relu3_3 = nn.Sequential()
        self.relu3_4 = nn.Sequential()
        self.max3 = nn.Sequential()

        self.relu4_1 = nn.Sequential()
        self.relu4_2 = nn.Sequential()
        self.relu4_3 = nn.Sequential()
        self.relu4_4 = nn.Sequential()

        self.relu5_1 = nn.Sequential()
        self.relu5_2 = nn.Sequential()
        self.relu5_3 = nn.Sequential()
        self.relu5_4 = nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 19):
            self.max3.add_module(str(x), features[x])

        for x in range(19, 22):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(22, 24):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(24, 26):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(26, 28):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(28, 31):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(31, 33):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(33, 35):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(35, 37):
            self.relu5_4.add_module(str(x), features[x])

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        relu1_1 = self.relu1_1(x)
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)
        max_3 = self.max3(relu3_4)

        relu4_1 = self.relu4_1(max_3)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,
            'max_3': max_3,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out
class PerPixelLoss(nn.Module):
    def __init__(self,):
        super(PerPixelLoss, self).__init__()

        self.criterion = nn.L1Loss(reduction='mean')

    def forward(self, image_gen, image_gt):

        per_pixel_loss = self.criterion(image_gen, image_gt)

        return per_pixel_loss
class StyleLoss(nn.Module):
    def __init__(self, vgg_model):

        super(StyleLoss, self).__init__()
        self.vgg = vgg_model.eval()  # ensure VGG in eval mode
        for p in self.vgg.parameters():
            p.requires_grad = False

        # Register ImageNet mean/std for normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def compute_gram(self, x):
        b, ch, h, w = x.size()
        f = x.view(b, ch, -1)  # flatten H*W
        gram = torch.bmm(f, f.transpose(1, 2)) / (ch * h * w)
        return gram

    def forward(self, x, y):
        self.vgg.to(x.device)
        # Scale from [-1,1] to [0,1]
        x = (x + 1) / 2
        y = (y + 1) / 2

        # Normalize
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        # Normalize
        x = (x - mean) / std
        y = (y - mean) / std

        # Extract VGG features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # Compute style loss (Gram matrices)
        style_loss = 0.0
        for layer in ['relu2_2', 'relu3_3', 'relu4_3', 'relu5_2']:
            gram_x = self.compute_gram(x_vgg[layer])
            gram_y = self.compute_gram(y_vgg[layer])
            style_loss += torch.mean((gram_x - gram_y) ** 2)

        return style_loss
class PerceptualLoss(nn.Module):
    def __init__(self, vgg_model, weights=None):

        super(PerceptualLoss, self).__init__()
        if weights is None:
            weights = [1.0, 1.0, 1.0, 1.0, 1.0]

        self.vgg = vgg_model.eval()  # Ensure VGG runs in eval mode (no dropout/bn updates)
        for p in self.vgg.parameters():
            p.requires_grad = False

        self.criterion = nn.L1Loss()
        self.weights = weights

        # Register mean/std as buffers so they move with .to(device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        self.vgg.to(x.device)

        # ---- Input expected in range [-1, 1]; normalize to [0,1]
        x = (x + 1) / 2
        y = (y + 1) / 2

        # ---- Normalize with ImageNet mean/std
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)

        # Normalize
        x = (x - mean) / std
        y = (y - mean) / std


        # ---- Extract features
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)

        # ---- Weighted perceptual loss
        content_loss = 0.0
        content_loss += self.weights[0] * self.criterion(x_vgg['relu1_1'], y_vgg['relu1_1'])
        content_loss += self.weights[1] * self.criterion(x_vgg['relu2_1'], y_vgg['relu2_1'])
        content_loss += self.weights[2] * self.criterion(x_vgg['relu3_1'], y_vgg['relu3_1'])
        content_loss += self.weights[3] * self.criterion(x_vgg['relu4_1'], y_vgg['relu4_1'])
        content_loss += self.weights[4] * self.criterion(x_vgg['relu5_1'], y_vgg['relu5_1'])

        return content_loss
class TotalVariationLoss(nn.Module):
    def __init__(self):
        super(TotalVariationLoss, self).__init__()

    def forward(self, image, mask=None):
        loss_col = torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :])
        loss_row = torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])
        total_loss = loss_col.mean() + loss_row.mean()
        return total_loss


def compute_gradient_penalty(discriminator, real_images, fake_images, mask=None):

    device = real_images.device
    batch_size = real_images.size(0)

    # Random interpolation between real and fake images
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = alpha * real_images + (1 - alpha) * fake_images
    interpolates.requires_grad_(True)

    # Discriminator output
    disc_interpolates = discriminator(interpolates)

    # Flatten output if necessary (e.g., [B,1,H,W] -> [B])
    if disc_interpolates.dim() > 2:
        disc_interpolates = disc_interpolates.view(batch_size, -1).mean(1)

    # Compute gradients w.r.t. interpolated images
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates, device=device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)

    # Apply mask if provided
    if mask is not None:
        if mask.dim() == 3:
            mask = mask.unsqueeze(1)  # [B,1,H,W]
        mask = mask.bool().to(device)
        rgb_mask = mask.repeat(1, real_images.size(1), 1, 1)  # repeat for C channels
        rgb_mask = rgb_mask.view(batch_size, -1)
        gradients = gradients * rgb_mask
        valid_pixels = rgb_mask.sum(dim=1)
        # Avoid division by zero
        gradient_norm = torch.sqrt((gradients ** 2).sum(dim=1) + 1e-12) / (valid_pixels + 1e-12)
    else:
        gradient_norm = gradients.norm(2, dim=1)

    # Compute gradient penalty
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty

"********************************************"




class TransferModel:
    def __init__(self,):
        self.vgg_path='E://projects/ADGAN/deepfashion/vgg19-dcbb9e9d.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.netG =  Generator(dim=64, style_dim=256, n_downsample=2, n_res=8, mlp_dim=256, activ='lrelu').to(self.device)
        init_weights(self.netG, init_type='normal')

        self.load_VGG(self.netG.enc_style.vgg)


        norm_layer = get_norm_layer(norm_type="instance")
        self.netD_PB = Discriminator(16,64, norm_layer=norm_layer, use_dropout=True, n_blocks=3, padding_type='reflect', use_sigmoid=False,n_downsampling=2).to(self.device)  # """generated image + target pose heatmap / target image + target pose heatmap"""
        self.netD_PP = Discriminator(6,64, norm_layer=norm_layer, use_dropout=True, n_blocks=3, padding_type='reflect', use_sigmoid=False,n_downsampling=2).to(self.device)  # """target image+source image / fake image + source image"""

        self.vgg = VGG().to(self.device)
        self.vgg.load_state_dict(torch.load('E://projects/ADGAN/deepfashion/'+ '/vgg_conv.pth',weights_only=True))
        for param in self.vgg.parameters():
            param.requires_grad = False





        self.per_pixel_loss = PerPixelLoss()
        self.style_loss = StyleLoss(VGG19())
        self.Perceptual_Loss = PerceptualLoss(VGG19())
        self.tv_loss = TotalVariationLoss()

        self.CX_loss = CXLoss(sigma=0.5).to(self.device)



        # === Optimizers ===
        self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()),lr=1e-4, betas=(0.5, 0.9))
        self.optimizer_D_PB = torch.optim.Adam(self.netD_PB.parameters(), lr=1e-4, betas=(0.5, 0.9))
        self.optimizer_D_PP = torch.optim.Adam(self.netD_PP.parameters(), lr=1e-4, betas=(0.5, 0.9))

        # === Scheduler ===
        total_steps = 100 * 162
        warmup_iters = int(0.2 * total_steps)
        self.G_scheduler = WarmupCosineAnnealingLR(self.optimizer_G,  total_steps // 2,total_steps*2)
        self.D_PB_scheduler = WarmupCosineAnnealingLR(self.optimizer_D_PB,total_steps // 2,total_steps*2)
        self.D_PP_scheduler = WarmupCosineAnnealingLR(self.optimizer_D_PP, total_steps // 2,total_steps*2)

        self.fake_PB_pool = ImagePool(70)
        self.fake_PP_pool = ImagePool(70)

        print('=' * 80)
        print('Networks initialized')
        print(f'Generator parameters: {sum(p.numel() for p in self.netG.parameters()):,}')
        print(f'Discriminator PB parameters: {sum(p.numel() for p in self.netD_PB.parameters()):,}')
        print(f'Discriminator PP parameters: {sum(p.numel() for p in self.netD_PP.parameters()):,}')
        print('=' * 80)

    def load_VGG(self, network):

        vgg19 = models.vgg19(pretrained=None)
        vgg19.load_state_dict(torch.load(self.vgg_path,weights_only=True))
        pretrained_model = vgg19.features

        pretrained_dict = pretrained_model.state_dict()

        model_dict = network.state_dict()

        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # load the new state dict
        network.load_state_dict(model_dict)

    def preprocess_vgg(self,x):
        # Convert from [-1, 1] → [0, 1]
        x = (x + 1) / 2
        # Normalize using ImageNet mean/std
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std

    def set_input(self, batch):
        self.input_P1 = batch['P1'].to(self.device)
        self.input_BP1 = batch['BP1'].to(self.device)
        self.input_P2 = batch['P2'].to(self.device)
        self.input_BP2 = batch['BP2'].to(self.device)

    def forward(self):
        self.fake_p2 = self.netG(self.input_BP2, self.input_P1)

    @torch.no_grad()
    def save_fake_img(self, epoch, save_dir="results"):

        self.netG.eval()
        os.makedirs(save_dir, exist_ok=True)

        # Generate fake image
        fake_p2 = self.netG(self.input_BP2, self.input_P1)

        # ---- Helper: Denormalize from [-1,1] to [0,1] ----
        def denorm(x):
            return (x + 1) / 2.0  # if your images are normalized using mean=0.5, std=0.5

        P1_vis = denorm(self.input_P1.detach().cpu())
        fake_vis = denorm(fake_p2.detach().cpu())
        P2_vis = denorm(self.input_P2.detach().cpu())
        BP2_vis = self.input_BP2.detach().cpu().numpy().sum(axis=1)

        grid = torch.cat([P1_vis, fake_vis,P2_vis], dim=0)

        save_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}_comparison.jpg")
        vutils.save_image(grid, save_path, nrow=P1_vis.shape[0], normalize=False)

    def compute_G_loss(self):
        losses = {}
        total_loss = 0

        pred_fake_PB = self.netD_PB(torch.cat((self.fake_p2, self.input_BP2), dim=1))
        pred_fake_PP = self.netD_PP(torch.cat((self.fake_p2, self.input_P1), dim=1))

        loss_G_GAN_PB = -torch.mean(pred_fake_PB)
        loss_G_GAN_PP = -torch.mean(pred_fake_PP)

        pair_GANloss = (loss_G_GAN_PB + loss_G_GAN_PP)
        losses['G_GAN_PB'] = loss_G_GAN_PB
        losses['G_GAN_PP'] = loss_G_GAN_PP

        # --- Pixel Loss ---
        l1_loss = self.per_pixel_loss(self.fake_p2, self.input_P2)
        losses['L1'] = l1_loss

        # --- Perceptual Loss ---
        perceptual_loss = self.Perceptual_Loss(self.fake_p2, self.input_P2)
        losses['Perceptual_loss'] = perceptual_loss

        # --- Style Loss ---
        style_loss = self.style_loss(self.fake_p2, self.input_P2)
        losses['Style_loss'] = style_loss

        tv_loss = self.tv_loss(self.fake_p2)
        losses['tv_loss'] = tv_loss

        # --- Contextual (CX) Loss ---
        """style_layers = ['r32', 'r42']
        vgg_style = self.vgg(self.preprocess_vgg(self.input_P2), style_layers)
        vgg_fake = self.vgg(self.preprocess_vgg(self.fake_p2), style_layers)
        cx_loss = sum(self.CX_loss(vgg_style[i], vgg_fake[i]) for i in range(len(vgg_fake)))
        losses['CX'] = cx_loss"""

        total_loss =  pair_GANloss  + 50.0 * l1_loss \
                     + 10.0 * perceptual_loss  + 0.5 * style_loss+ 0.05 * tv_loss

        """total_loss = 1. * pair_GANloss  + 10.0 * l1_loss \
                     + 1.0 * perceptual_loss + 1.0 * style_loss + 0.5 * cx_loss+ 0.1 * tv_loss"""

        losses['total'] = total_loss
        return losses

    def compute_D_loss(self, netD, real, fake):

        pred_real = netD(real)

        pred_fake = netD(fake.detach())

        loss_D = torch.mean(F.relu(1 - pred_real)) + torch.mean(F.relu(1 + pred_fake))

        gp = compute_gradient_penalty(discriminator=netD,real_images=real,fake_images=fake.detach(),mask=None )
        loss_D+= gp * 10.

        return loss_D

    def backward_D_PP(self):
        real_PP = torch.cat((self.input_P2, self.input_P1), 1)
        fake_PP =torch.cat((self.fake_p2, self.input_P1), 1)
        fake_PP = self.fake_PP_pool.query(fake_PP)
        return self.compute_D_loss(self.netD_PP, real_PP, fake_PP)

    def backward_D_PB(self):
        real_PB = torch.cat((self.input_P2, self.input_BP2), 1)
        fake_PB = torch.cat((self.fake_p2, self.input_BP2), 1)
        fake_PB = self.fake_PB_pool.query(fake_PB)
        return self.compute_D_loss(self.netD_PB, real_PB, fake_PB)

    def save_networks(self, epoch, save_dir='checkpoints'):



        checkpoint = {
            'epoch': epoch,
            'netG_state_dict': self.netG.state_dict(),
            'netD_PB_state_dict': self.netD_PB.state_dict(),
            'netD_PP_state_dict': self.netD_PP.state_dict(),
            'optimizer_G_state_dict': self.optimizer_G.state_dict(),
            'optimizer_D_PB_state_dict': self.optimizer_D_PB.state_dict(),
            'optimizer_D_PP_state_dict': self.optimizer_D_PP.state_dict(),
        }

        save_path = f'checkpoints/checkpoint_epoch_{epoch+1}.pth'
        torch.save(checkpoint, save_path)
        print(f'Saved checkpoint: {save_path}')

    def load_networks(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.netG.load_state_dict(checkpoint['netG_state_dict'])
        self.netD_PB.load_state_dict(checkpoint['netD_PB_state_dict'])
        self.netD_PP.load_state_dict(checkpoint['netD_PP_state_dict'])
        self.optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
        self.optimizer_D_PB.load_state_dict(checkpoint['optimizer_D_PB_state_dict'])
        self.optimizer_D_PP.load_state_dict(checkpoint['optimizer_D_PP_state_dict'])

        print(f'Loaded checkpoint: {checkpoint_path}')
        return checkpoint['epoch']

class Trainer:
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.current_epoch = 0
        self.total_steps = 0

    def train_epoch(self):
        self.model.netG.train()
        self.model.netD_PB.train()
        self.model.netD_PP.train()

        epoch_losses = OrderedDict()
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch + 1}/100')

        for batch_idx, batch in enumerate(pbar):
            self.total_steps += 1
            self.model.set_input(batch)
            self.model.forward()

            self.model.optimizer_G.zero_grad()
            g_losses = self.model.compute_G_loss()
            g_losses['total'].backward()
            # torch.nn.utils.clip_grad_norm_( self.model.netG.parameters(), max_norm=1.0)
            self.model.optimizer_G.step()
            self.model.G_scheduler.step()


            d_losses = {}
            loss_D_PP=loss_D_PB=0.0

            for _ in range(3):
                self.model.optimizer_D_PP.zero_grad()
                loss_D_PP = self.model.backward_D_PP()
                loss_D_PP.backward()
                self.model.optimizer_D_PP.step()
                self.model.D_PP_scheduler.step()

            d_losses['D_PP'] = loss_D_PP.item()
            for _ in range(3):
                self.model.optimizer_D_PB.zero_grad()
                loss_D_PB = self.model.backward_D_PB()
                loss_D_PB.backward()
                self.model.optimizer_D_PB.step()
                self.model.D_PB_scheduler.step()

            d_losses['D_PB'] = loss_D_PB.item()




            losses_dict = {}
            for k, v in {**g_losses, **d_losses}.items():
                if torch.is_tensor(v):
                    losses_dict[k] = v.item()
                elif v is not None:
                    losses_dict[k] = v

            for k, v in losses_dict.items():
                if k not in epoch_losses:
                    epoch_losses[k] = []
                if v is not None:
                    epoch_losses[k].append(v)

            if 'total' in losses_dict:
                pbar.set_postfix({
                    'G_total': f"{losses_dict['total']:.4f}",
                    'D_PB': f"{losses_dict['D_PB']:.4f}",
                    'D_PP': f"{losses_dict['D_PP']:.4f}",
                })


        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items() if len(v) > 0}
        return avg_losses

    def validate(self):
        if self.val_loader is None:
            return {}

        self.model.netG.eval()

        val_losses = OrderedDict()


        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc='Validation', leave=False)
            for batch in pbar:
                self.model.set_input(batch)
                self.model.forward()

                # Compute generator losses only
                g_losses = self.model.compute_G_loss()

                for k, v in g_losses.items():
                    if k not in val_losses:
                        val_losses[k] = []
                    if torch.is_tensor(v):
                        val_losses[k].append(v.item())
                    elif v is not None:
                        val_losses[k].append(v)




        avg_val_losses = {k: np.mean(v) for k, v in val_losses.items() if len(v) > 0}
        return avg_val_losses

    def train(self):
        """Main training loop"""
        print(f'\nStarting training for {100} epochs...\n')
        start_time = time.time()

        for epoch in range(self.current_epoch, 100):
            self.current_epoch = epoch
            train_losses = self.train_epoch()
            val_losses = self.validate()
            self.model.save_networks(epoch)
            self.model.save_fake_img(epoch, save_dir="../results/preview/")
            print(f'\nEpoch {epoch + 1}/{100} - Training Losses:{train_losses} / Validate Losses:{val_losses}')


model = TransferModel()

trainer = Trainer(model, train_loader,test_loader)

trainer.train()



