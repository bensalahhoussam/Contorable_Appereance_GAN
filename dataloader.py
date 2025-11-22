import torch
from torch.utils.data import Dataset, DataLoader,random_split
import os
import numpy as np
import scipy.io as sio


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mat_path="../VAE_pose/dataset/labels/"
test_path="../VAE_pose/dataset/test/"


class PennActionSeqDataset(Dataset):
    def __init__(self, mat_dir, past_len=5, future_len=10, max_frames=30):
        self.mat_path = mat_dir
        self.past_len = past_len
        self.future_len = future_len
        self.inputs = []
        self.targets = []


        all_frames_list = []
        mat_files = sorted([f"{self.mat_path}/{f}" for f in os.listdir(self.mat_path)])
        videos = []
        for mat_path in mat_files:
            data = sio.loadmat(mat_path, spmatrix=False)
            xcoords = np.array(data['x'], dtype=np.float32)
            ycoords = np.array(data['y'], dtype=np.float32)
            print(xcoords.shape)
            n_frames, n_joints = xcoords.shape
            joints = np.stack([xcoords, ycoords], axis=-1)
            print(joints.shape)
            if joints.shape[0] < 70:

                frames = joints.reshape(n_frames, -1)
                videos.append(frames)
                all_frames_list.append(frames)

        # Compute dataset-wide mean/std
        all_frames_concat = np.concatenate(all_frames_list, axis=0)
        self.mean = np.mean(all_frames_concat, axis=0)
        self.std = np.std(all_frames_concat, axis=0) + 1e-8

        # Normalize each video and create samples
        for frames in videos:
            frames_norm = (frames - self.mean) / self.std
            num_samples = len(frames_norm) - self.past_len - self.future_len
            for idx in range(num_samples):
                src = frames_norm[idx : idx + self.past_len]
                tgt_full = frames_norm[idx + self.past_len : idx + self.past_len + self.future_len]
                delta = np.diff(np.vstack([src[-1:], tgt_full]), axis=0)

                self.inputs.append(src)
                self.targets.append(delta)

        self.inputs = np.array(self.inputs, dtype=np.float32)
        self.targets = np.array(self.targets, dtype=np.float32)
        self.decoder_input = np.zeros_like(self.targets)

        print(f"Dataset prepared: {self.inputs.shape} inputs, {self.targets.shape} targets")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.inputs[idx]).float()
        y = torch.from_numpy(self.decoder_input[idx]).float()
        z = torch.from_numpy(self.targets[idx]).float()
        return x, y, z



dataset = PennActionSeqDataset(mat_path)



train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(dataset, [train_size, val_size])


train_loader = DataLoader(train_dataset, batch_size=16*2, shuffle=True)
test_loader = DataLoader(val_dataset, batch_size=16*2, shuffle=False)




