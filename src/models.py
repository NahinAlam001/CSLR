import torch
import torch.nn as nn
import math
import random
from src.utils import PositionalEncoding

class FusionModule(nn.Module):
    """Fuse MediaPipe and I3D features via cross-attention."""
    def __init__(self, pose_dim: int = 99, i3d_dim: int = 1024, out_dim: int = 512):
        super().__init__()
        self.pose_proj = nn.Linear(pose_dim, out_dim)
        self.i3d_proj = nn.Linear(i3d_dim, out_dim)
        self.cross_attn = nn.MultiheadAttention(out_dim, 8, batch_first=True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        B, T, D = features.shape
        pose = features[:, :, :99]
        i3d = features[:, :, 99:]
        pose_emb = self.pose_proj(pose)
        i3d_emb = self.i3d_proj(i3d)
        fused, _ = self.cross_attn(pose_emb, i3d_emb, i3d_emb)
        return fused

class TransformerEncoder(nn.Module):
    """Encoder with fusion and positional encoding."""
    def __init__(self, input_dim: int, hid_dim: int, n_layers: int, n_heads: int, pf_dim: int, dropout: float):
        super().__init__()
        self.fusion = FusionModule()
        self.pos_encoder = PositionalEncoding(hid_dim, dropout)
        encoder_layers = nn.TransformerEncoderLayer(hid_dim, n_heads, pf_dim, dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layers, n_layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        src = self.fusion(src)
        src = self.pos_encoder(src)
        output = self.encoder(src)
        return output

class TransformerDecoder(nn.Module):
    """Decoder with embedding and masking support."""
    def __init__(self, output_dim: int, hid_dim: int, n_layers: int, n_heads: int, pf_dim: int, dropout: float):
        super().__init__()
        self.embedding = nn.Embedding(output_dim, hid_dim)
        self.pos_encoder = PositionalEncoding(hid_dim, dropout)
        decoder_layers = nn.TransformerDecoderLayer(hid_dim, n_heads, pf_dim, dropout, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layers, n_layers)
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg: torch.Tensor, enc_out: torch.Tensor, trg_mask: torch.Tensor = None) -> torch.Tensor:
        trg_emb = self.embedding(trg)
        trg_emb = self.pos_encoder(trg_emb)
        output = self.decoder(trg_emb, enc_out, tgt_mask=trg_mask)
        prediction = self.fc_out(output)
        return prediction

class Seq2Seq(nn.Module):
    """Sequence-to-sequence transformer model."""
    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device, sos_id: int, eos_id: int, pad_id: int):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.sos_id = sos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def forward(self, src: torch.Tensor, trg: torch.Tensor) -> torch.Tensor:
        # Parallel training with mask
        enc_out = self.encoder(src)
        trg_input = trg[:, :-1]  # Shift right
        trg_mask = self.generate_square_subsequent_mask(trg_input.size(1)).to(self.device)
        output = self.decoder(trg_input, enc_out, trg_mask)
        return output

    def generate(self, src: torch.Tensor, max_len: int = 100) -> torch.Tensor:
        """Autoregressive generation for inference."""
        enc_out = self.encoder(src)
        trg = torch.full((src.size(0), 1), self.sos_id, dtype=torch.long, device=self.device)
        for _ in range(max_len):
            trg_mask = self.generate_square_subsequent_mask(trg.size(1)).to(self.device)
            out = self.decoder(trg, enc_out, trg_mask)
            next_token = out[:, -1, :].argmax(-1).unsqueeze(1)
            trg = torch.cat([trg, next_token], dim=1)
            if (next_token == self.eos_id).all():
                break
        return trg

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=self.device)) == 1
        mask = mask.transpose(0, 1).float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
