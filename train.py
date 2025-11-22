from losses import *
from dataloader import *
from model_seq2seq import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.optim.lr_scheduler import _LRScheduler
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


def train(Genrator, train_loader, val_loader, epochs, lr, device, save_path):
    G = Genrator.to(device)

    total_steps = epochs * len(train_loader)
    warmup_iters = int(0.2 * total_steps)

    optimizer_G = optim.Adam(G.parameters(), lr=lr,betas=(0.9, 0.999))

    scheduler_G = WarmupCosineAnnealingLR(optimizer_G, warmup_iters, total_steps)

    criterion_recon = nn.MSELoss()

    """criterion = nn.L1Loss()"""

    best_val = float('inf')
    train_losses = []
    train_D_losses = []
    val_losses = []

    for epoch in range(1, epochs + 1):
        G.train()
        total_G_loss, total_D_loss = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        for src, decoder_input, delta_target in pbar:

            src = src.to(device)                  # (B, past_len, feat)
            decoder_input = decoder_input.to(device)  # zeros (B, future_len, feat)
            delta_target = delta_target.to(device)    # (B, future_len, feat)

            # =====================================================
            # 2️⃣ Train Generator
            # =====================================================
            optimizer_G.zero_grad()
            delta_pred = G(src, decoder_input)   # predict future deltas

            recon_loss = criterion_recon(delta_pred, delta_target)

            G_loss = recon_loss

            G_loss.backward()
            torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1.0)
            optimizer_G.step()
            scheduler_G.step()

            total_G_loss += G_loss.item() * src.size(0)

            pbar.set_postfix({'G_loss': total_G_loss / ((pbar.n + 1) * src.size(0))})

        avg_train_loss = total_G_loss / len(train_loader.dataset)

        train_losses.append(avg_train_loss)


        # -----------------------------
        # Validation
        # -----------------------------
        G.eval()
        val_loss = 0.0
        with torch.no_grad():
            for src, decoder_input, delta_target in val_loader:
                src = src.to(device)
                decoder_input = decoder_input.to(device)
                delta_target = delta_target.to(device)

                delta_pred = G(src, decoder_input)
                loss = criterion_recon(delta_pred, delta_target)
                val_loss += loss.item() * src.size(0)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        print(f"Epoch {epoch} train_loss={avg_train_loss:.6f}  val_loss={avg_val_loss:.6f}")
        # -----------------------------
        # Save best checkpoint
        # -----------------------------
        if avg_val_loss < best_val:
            best_val = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state': G.state_dict(),
                'optimizer_state': optimizer_G.state_dict(),
                'val_loss': avg_val_loss
            }, save_path)
            print("Saved best checkpoint:", save_path)

    return train_losses,train_D_losses, val_losses




epochs = 300
lr = 1e-4
train_losses, train_D_losses,val_losses = train(generator, train_loader, test_loader, epochs=epochs, lr=lr, device=device, save_path="../VAE_pose/checkpoints/pose_transformer.pth")
