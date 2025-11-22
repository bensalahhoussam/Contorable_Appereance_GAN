import os
import numpy as np
import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt
from dataloader import test_loader,train_loader,dataset
from model_seq2seq import *

from matplotlib.animation import FuncAnimation, PillowWriter



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@torch.no_grad()
def predict_future(model, src_seq, future_len=10, device="cuda"):

    model.eval()
    src_seq = src_seq.unsqueeze(0).to(device)  # (1, past_len, input_size)

    # Encoder pass
    memory = model.encoder(model.pos_enc(model.encoder_embed(src_seq)))

    preds = []
    decoder_input = torch.zeros(1, 1, src_seq.size(2)).to(device)  # start token (Δ=0)
    last_pose = src_seq[:, -1:, :]  # last observed pose

    for _ in range(future_len):
        tgt_embed = model.pos_enc(model.decoder_embed(decoder_input))
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embed.size(1)).to(device)
        out = model.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
        delta = model.fc_out(out[:, -1:, :])           # predicted Δ-pose
        next_pose = last_pose + delta                  # reconstruct absolute pose
        preds.append(next_pose.squeeze(0).cpu().numpy())
        decoder_input = torch.cat([decoder_input, delta], dim=1)
        last_pose = next_pose

    return np.concatenate(preds, axis=0)  # (future_len, input_size)

skeleton = [
    (0, 1),   # Head to neck
    (1, 2), (2, 3), (3, 4),       # Right arm
    (1, 5), (5, 6), (6, 7),       # Left arm
    (1, 8), (8, 9), (9, 10),      # Right leg
    (1, 11), (11, 12),             # Left leg
    (8, 11)                        # Hip connection
]


def denormalize(normed, mean, std):
    return normed * std + mean
def plot_pose(ax, pose, color='blue', title=None):
    n_joints = pose.size // 2
    coords = pose.reshape(n_joints, 2)
    ax.scatter(coords[:, 0], coords[:, 1], color=color)
    ax.invert_yaxis()
    if title:
        ax.set_title(title)


def update(i):
    ax.clear()
    plot_pose(ax, true_future_denorm[i], color='green')  # true
    plot_pose(ax, future_denorm[i], color='red')         # predicted




    ax.set_title(f"Frame {i+1}/{future_len}")
    ax.legend(['True', 'Pred'])
    ax.set_xlim(np.min(true_future_denorm) - 10, np.max(true_future_denorm) + 10)
    ax.set_ylim(np.max(true_future_denorm) + 10, np.min(true_future_denorm) - 10)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")

mean = dataset.mean
std = dataset.std


input_size = 26
embed_dim = 128
num_heads = 4
num_layers = 4
past_len = 5
future_len = 10
dropout = 0.1

model = Seq2Seq(input_size, embed_dim, num_heads, num_layers, future_len, dropout).to(device)



ckpt_path = "../VAE_pose/checkpoints/pose_transformer.pth"
ckpt = torch.load(ckpt_path, weights_only=True,map_location=device)
model.load_state_dict(ckpt["model_state"], strict=True)
model.eval()

index=4







src_seq, _, tgt_delta = next(iter(test_loader))

tgt_delta = tgt_delta[index].to(device)
future_len = tgt_delta.shape[0]


src_seq = src_seq[index].to(device)





future_norm = predict_future(model, src_seq, future_len=10,device="cuda")

true_future_norm = src_seq[-1].cpu().numpy() + np.cumsum(tgt_delta.cpu().numpy(), axis=0)


src_denorm = denormalize(src_seq.cpu().numpy(), mean, std)
future_denorm = denormalize(future_norm, mean, std)
true_future_denorm = denormalize(true_future_norm, mean, std)

print("Source (past):", src_denorm.shape)
print("Predicted (future):", future_denorm.shape)
print("True (future):", true_future_denorm.shape)




fig, ax = plt.subplots(figsize=(4, 4))


# --- Animation object ---
ani = FuncAnimation(fig, update, frames=10, interval=300, repeat=False)

# --- Save animation as GIF ---
ani.save("pose_prediction.gif", writer=PillowWriter(fps=3))
print("✅ Saved animation as 'pose_prediction.gif'")



