import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class Seq2Seq(nn.Module):
    def __init__(self, input_size=26, embed_dim=128, num_heads=4, num_layers=4, future_len=20, dropout=0.1):
        super(Seq2Seq,self).__init__()
        self.future_len = future_len

        self.encoder_embed = nn.Linear(input_size, embed_dim)
        self.decoder_embed = nn.Linear(input_size, embed_dim)
        self.pos_enc = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True, dropout=dropout)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=512, batch_first=True, dropout=dropout)

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.fc_out = nn.Linear(embed_dim, input_size)

    def forward(self, src, tgt):

        src_embed = self.pos_enc(self.encoder_embed(src))
        tgt_embed = self.pos_enc(self.decoder_embed(tgt))

        memory = self.encoder(src_embed)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_embed.size(1),device=src.device)
        out = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)

        delta = self.fc_out(out)

        return delta



class Discriminator(nn.Module):
    def __init__(self, input_size=26, channels=(64, 128, 256), kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        in_ch = input_size

        for out_ch in channels:
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size, stride=1, padding=kernel_size//2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(dropout)
            ]
            in_ch = out_ch

        # Global pooling + final classification
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(channels[-1], 1)

    def forward(self, motion_seq):
        """
        motion_seq: (B, T, F)
        """
        x = motion_seq.transpose(1, 2)  # (B, F, T) for Conv1d
        feat = self.conv(x)              # (B, C, T)
        pooled = feat.mean(dim=2)        # Global average pooling across time
        out = self.fc(pooled)            # (B, 1)
        return out


input_size = 26
embed_dim = 128
num_heads = 4
num_layers = 4
past_len = 5
future_len = 10
dropout = 0.1

generator = Seq2Seq(input_size, embed_dim, num_heads, num_layers, future_len, dropout)

discriminator = Discriminator()