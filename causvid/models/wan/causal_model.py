from causvid.models.wan.wan_base.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    sinusoidal_embedding_1d
)
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math


# -------------------------
# RoPE FREQUENCY BUILDER (REAL ANGLES)
# -------------------------
def build_rope_freqs(seq_len, dim):
    theta = 10000 ** (-torch.arange(0, dim, 2).float() / dim)
    pos = torch.arange(seq_len).float()
    freqs = torch.einsum("i,j->ij", pos, theta)  # [seq_len, dim/2]
    return freqs


# -------------------------
# Standard RoPE (ONNX + TRT SAFE)
# -------------------------
def rope_apply(x, grid_sizes, freqs):
    B, L, H, D = x.shape
    D2 = D // 2

    freqs = freqs[:L].to(x.device)
    if freqs.shape[0] < L:
        idx = torch.arange(L, device=x.device) % freqs.shape[0]
        freqs = freqs[idx]

    cos = torch.cos(freqs).unsqueeze(0).unsqueeze(2)  # [1, L, 1, D2]
    sin = torch.sin(freqs).unsqueeze(0).unsqueeze(2)

    x1 = x[..., :D2]
    x2 = x[..., D2:]

    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos

    return torch.cat([out1, out2], dim=-1)


# -------------------------
# Self Attention (SDPA)
# -------------------------
class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, qk_norm=True, eps=1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, **kwargs):
        B, L, C = x.shape
        H, D = self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(B, L, H, D)
        k = self.norm_k(self.k(x)).view(B, L, H, D)
        v = self.v(x).view(B, L, H, D)

        q = rope_apply(q, grid_sizes, freqs).type_as(v)
        k = rope_apply(k, grid_sizes, freqs).type_as(v)

        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=None
        )

        out = out.transpose(2, 1).reshape(B, L, C)
        return self.o(out)


# -------------------------
# Transformer Block
# -------------------------
class CausalWanAttentionBlock(nn.Module):
    def __init__(self, cross_attn_type, dim, ffn_dim, num_heads,
                 qk_norm=True, cross_attn_norm=False, eps=1e-6):
        super().__init__()

        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, qk_norm, eps)

        self.norm3 = WanLayerNorm(dim, eps) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](
            dim, num_heads, (-1, -1), qk_norm, eps
        )

        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(self, x, e, seq_lens, grid_sizes, freqs, context, context_lens, **kwargs):
        B, L, C = x.shape
        num_frames = e.shape[1]
        frame_seqlen = L // num_frames

        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)

        y = self.self_attn(
            (self.norm1(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes, freqs
        )

        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # Cross-attention (can be disabled if needed)
        x = x + self.cross_attn(self.norm3(x), context, context_lens)

        y = self.ffn(
            (self.norm2(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
        )

        x = x + (y.unflatten(1, (num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
        return x


# -------------------------
# Output Head
# -------------------------
class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        B, L, C = x.shape
        num_frames = e.shape[1]
        frame_seqlen = L // num_frames

        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        return self.head(
            self.norm(x).unflatten(1, (num_frames, frame_seqlen)) * (1 + e[1]) + e[0]
        )


# -------------------------
# MAIN MODEL
# -------------------------
class CausalWanModel(ModelMixin, ConfigMixin):
    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
    ):
        super().__init__()

        self.model_type = model_type
        self.patch_size = patch_size
        self.text_len = text_len
        self.freq_dim = freq_dim

        self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(dim, dim),
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        cross_attn_type = "t2v_cross_attn" if model_type == "t2v" else "i2v_cross_attn"

        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads, qk_norm, cross_attn_norm, eps
                )
                for _ in range(num_layers)
            ]
        )

        self.head = CausalHead(dim, out_dim, patch_size, eps)

        d = dim // num_heads
        self.freqs = build_rope_freqs(1024, d)

    # -------------------------
    # Minimal Forward (ONNX-safe)
    # -------------------------
    def forward(self, x, t, context, seq_len, clip_fea=None, y=None):
        device = self.patch_embedding.weight.device
        self.freqs = self.freqs.to(device)

        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long, device=device)

        x = torch.cat(x)

        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x)
        )

        e0 = self.time_projection(e).unflatten(1, (6, x.shape[-1])).unflatten(0, t.shape)

        context = self.text_embedding(
            torch.stack([
                torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ])
        )

        for block in self.blocks:
            x = block(
                x,
                e0,
                seq_lens,
                grid_sizes,
                self.freqs,
                context,
                None,
            )

        x = self.head(x, e.unflatten(0, t.shape).unsqueeze(2))
        return x
