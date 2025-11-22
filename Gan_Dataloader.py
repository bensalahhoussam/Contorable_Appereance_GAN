import torch
from torch.utils.data import Dataset, DataLoader,random_split
import os
import numpy as np
import scipy.io as sio
import random
from scipy.io import loadmat
import cv2
import matplotlib.pyplot as plt
import shutil
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import os
import random
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms




def joints_to_heatmap(joints, H=256, W=256, sigma=2):

    num_joints = joints.shape[0]

    y = np.arange(H).reshape(H, 1)
    x = np.arange(W).reshape(1, W)

    heatmaps = np.zeros((num_joints, H, W), dtype=np.float32)

    for j in range(num_joints):
        cx, cy = joints[j]
        if cx < 0 or cy < 0 or cx >= W or cy >= H:
            continue

        heatmaps[j] = np.exp(-((x - cx)**2 + (y - cy)**2) / (2 * sigma * sigma))

    return heatmaps


connections = [(0, 2),(0, 1),(1, 7),(7, 9),(9, 11),
    (2, 8),(8, 10),(10, 12),(2, 4),(4, 6),(1, 3),
    (3, 5),(1,2),(7,8)]
def center_object(img,  keypoints,bbox,keypoints_relative=False):
    x_min, y_min, x_max, y_max = bbox
    h, w, _ = img.shape
    # --- Compute bbox center ---

    bbox_cx = (x_min + x_max) / 2
    bbox_cy = (y_min + y_max) / 2

    # --- Compute image center ---

    img_cx = w / 2
    img_cy = h / 2

    # --- Compute translation offsets ---

    dx = int(img_cx - bbox_cx)
    dy = int(img_cy - bbox_cy)

    # --- Create translation matrix ---

    M = np.float32([[1, 0, dx], [0, 1, dy]])

    img = cv2.warpAffine(img, M, (w, h), borderValue=(0, 0, 0))

    # --- Shift keypoints ---
    keypoints[:, 0] += dx
    keypoints[:, 1] += dy

    bbox = (x_min + dx,y_min + dy,x_max + dx,y_max + dy)

    return img,keypoints,bbox

def draw_bbox(image,bbox,joints):
    x1,y1,x2,y2 = bbox
    margin = 10
    inside = ((joints[:, 0] >= x1-margin) &
            (joints[:, 0] <= x2+margin) &
            (joints[:, 1] >= y1-margin) &
            (joints[:, 1] <= y2+margin))



    all_inside = inside.all()
    box = [int(y1)+margin,int(y2)+margin,int(x1)-margin,int(x2)+margin]

    #cv2.rectangle(image,(int(x1)-margin,int(y1)+margin),(int(x2)+margin,int(y2)+margin),(0,255,0),1)

    image_zeros = np.zeros_like(image)

    h,w,_ = image_zeros.shape


    heatmap = joints_to_heatmap(joints,h,w)

    for j, (kx, ky) in enumerate(joints):

        kx_int, ky_int = int(kx), int(ky)

        cv2.circle( image_zeros, (kx_int, ky_int), 1, (255, 255, 255), -1)





    return image,all_inside,box,image_zeros,heatmap



# --- Base paths ---
base_folder = "../VAE_pose/dataset/frames/"
mat_folder_path = "../VAE_pose/dataset/labels/"
output_root = "../new_data/train"
output_root_mask = "../new_data/mask/"

# --- Loop over every folder in base_folder ---
folders = [f for f in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, f))]

for selected_folder in folders:
    print(f"Processing folder: {selected_folder}")

    # --- Load corresponding .mat file ---
    mat_file = os.path.join(mat_folder_path, f"{selected_folder}.mat")
    if not os.path.exists(mat_file):
        print(f"⚠️ No .mat file for {selected_folder}, skipping.")
        continue

    data = sio.loadmat(mat_file, spmatrix=False)
    bbox = data['bbox'].squeeze()
    xcoords = np.array(data['x'], dtype=np.float32)
    ycoords = np.array(data['y'], dtype=np.float32)
    joints = np.stack([xcoords, ycoords], axis=-1)


    folder_path = os.path.join(base_folder, selected_folder)
    frames = sorted([f for f in os.listdir(folder_path) if f.endswith('.jpg')])

    # --- Create output folder ---
    output_folder = os.path.join(output_root, selected_folder)
    output_mask = os.path.join(output_root_mask, selected_folder)

    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_mask, exist_ok=True)
    # --- Process every frame ---
    for frame_name in frames:
        frame_index = int(frame_name.split(".")[0]) - 1  # adjust index
        if frame_index < 0 or frame_index >= len(bbox):
            print(f"⚠️ Skipping {frame_name}: index {frame_index} out of range (bbox len={len(bbox)})")
            continue
        img_path = os.path.join(folder_path, frame_name)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        box = bbox[frame_index]
        joint = joints[frame_index]

        # Apply your custom functions
        img , joints_out, box_out = center_object(img, joint, box, False)
        img_drawn,draw_pox,boxes,mask_pose,heatmap = draw_bbox(img, box_out, joints_out)
        [y1,y2,x1,x2] = boxes
        if x2 <= x1 or y2 <= y1:
            print(f"⚠️ Invalid crop region: ({x1},{y1})→({x2},{y2}), skipping.")
            continue
        target_size = (176, 256)
        crop_img = img[y1:y2,x1:x2]
        crop_img = cv2.resize(crop_img, target_size, interpolation=cv2.INTER_LINEAR)

        crop_mask = heatmap[...,y1:y2,x1:x2]
        resized_channels = [cv2.resize(ch, target_size, interpolation=cv2.INTER_LINEAR) for ch in crop_mask]
        crop_mask = np.stack(resized_channels, axis=0)

        if draw_pox :



            save_path = os.path.join(output_folder, frame_name)
            save_mask = os.path.join(output_mask,frame_name)

            cv2.imwrite(save_path, cv2.cvtColor(crop_img, cv2.COLOR_RGB2BGR))
            np.save(save_mask,crop_mask )

    print(f"✅ Done with {selected_folder}\n")



"""file1= "../data/mask/0001/000030.jpg.npy"
file2 = "../data/train/0001/000030.jpg"

pose = np.load(file1)

img1 = cv2.imread(file2)/255
print(img1.shape)
print(pose.shape)
print("******")
bp2 = pose.transpose(1,2,0).max(axis=-1).reshape(img1.shape[0],img1.shape[1],1)
heat2 = np.concatenate([bp2, bp2, bp2], axis=-1)
img = np.concatenate([img1,heat2],axis=1)
plt.imshow(img)
plt.show()
"""


def delete_empty_pairs(image_root, mask_root, valid_img_exts=(".jpg", ".png"), valid_mask_exts=(".npy",)):
    """
    Delete folder pairs (from image_root and mask_root) if both are empty.
    A folder is considered empty if it contains no valid image or mask files.
    """

    count_deleted = 0

    image_folders = sorted([f for f in os.listdir(image_root) if os.path.isdir(os.path.join(image_root, f))])

    for folder_id in image_folders:
        img_folder = os.path.join(image_root, folder_id)
        mask_folder = os.path.join(mask_root, folder_id)

        # Skip if mask folder doesn't exist
        if not os.path.exists(mask_folder):
            print(f"⚠️ Skipping {folder_id}: mask folder missing")
            continue

        # Check if both folders contain valid files
        img_files = [f for f in os.listdir(img_folder) if f.lower().endswith(valid_img_exts)]
        mask_files = [f for f in os.listdir(mask_folder) if f.lower().endswith(valid_mask_exts)]

        if len(img_files) == 0 and len(mask_files) == 0:
            print(f"🗑️ Deleting empty pair: {folder_id}")
            shutil.rmtree(img_folder)
            shutil.rmtree(mask_folder)
            count_deleted += 1

    print(f"\n✅ Done. Deleted {count_deleted} empty folder pairs.")




'''delete_empty_pairs(output_root, output_root_mask)'''









class PoseDataset(Dataset):
    def __init__(self, image_root, mask_root, transform=None,phase="Train"):
        self.image_root = image_root
        self.mask_root = mask_root
        self.transform = transform
        self.phase = phase
        # collect all folder ids
        self.folder_ids = sorted(os.listdir(image_root))

        # store all valid folders (those that have at least 2 frames)
        self.valid_folders = []
        for folder_id in self.folder_ids:
            img_folder = os.path.join(image_root, folder_id)
            frame_names = sorted([f for f in os.listdir(img_folder) if f.endswith(".jpg")])
            if len(frame_names) >= 2:  # need at least 2 for pairs
                self.valid_folders.append((folder_id, frame_names))

    def __len__(self):
        # we can return number of folders or arbitrary large number to sample many random pairs
        return len(self.valid_folders) * 10  # can be any large multiple

    def __getitem__(self, idx):
        # randomly choose one folder
        folder_id, frame_names = random.choice(self.valid_folders)

        # randomly select two *different* frames from same folder
        init_name, target_name = random.sample(frame_names, 2)
        P1_path = os.path.join(self.image_root, folder_id, init_name)
        BP1_path = os.path.join(self.mask_root, folder_id, init_name + ".npy")
        P2_path = os.path.join(self.image_root, folder_id, target_name)
        BP2_path = os.path.join(self.mask_root, folder_id, target_name + ".npy")

        P1_img = Image.open(P1_path).convert('RGB')
        P2_img = Image.open(P2_path).convert('RGB')
        BP1_img = np.load(BP1_path).astype(np.float32)
        BP2_img = np.load(BP2_path).astype(np.float32)

        if self.phase == 'train' and self.use_flip and random.random() > 0.5:
            P1_img = P1_img.transpose(Image.FLIP_LEFT_RIGHT)
            P2_img = P2_img.transpose(Image.FLIP_LEFT_RIGHT)
            BP1_img = BP1_img[:, ::-1, :].copy()
            BP2_img = BP2_img[:, ::-1, :].copy()

        if self.transform:
            P1 = self.transform(P1_img)  # normalized [-1, 1] if using Normalize(0.5, 0.5, 0.5)
            P2 = self.transform(P2_img)
        else:
            P1 = torch.tensor(np.array(P1_img) / 255.0).permute(2, 0, 1)
            P2 = torch.tensor(np.array(P2_img) / 255.0).permute(2, 0, 1)

        BP1 = torch.from_numpy(BP1_img).float()
        BP2 = torch.from_numpy(BP2_img).float()




        return {'P1': P1,'BP1': BP1,'P2': P2,'BP2': BP2,}


transform = transforms.Compose([transforms.Resize((256, 176)),transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


dataset = PoseDataset(image_root="../data/train",mask_root="../data/mask", transform=transform,phase="Train")

train_ratio = 0.9
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size

# random split
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# build loaders
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=0)



print(len(train_loader))


