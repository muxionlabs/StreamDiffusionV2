import os
USE_TENSORRT_EXPORT = os.environ.get("USE_TENSORRT_EXPORT", "0") == "1"  # Set to True for ONNX/TensorRT export logic
from causvid.models.wan.wan_base.modules.attention import attention
from causvid.models.wan.wan_base.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WAN_CROSSATTENTION_CLASSES,
    Head,
    rope_params,
    MLPProj,
    sinusoidal_embedding_1d
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import torch.nn as nn
import torch
import math
from flash_attn import flash_attn_interface
import torch.distributed as dist

# wan 1.3B model has a weird channel / head configurations and require max-autotune to work with flexattention
# see https://github.com/pytorch/pytorch/issues/133254
# change to default for other models
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune")


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    if not USE_TENSORRT_EXPORT:
        n, c = x.size(2), x.size(3) // 2
        # split freqs
        freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
        # loop over samples
        output = []
        for i, (f, h, w) in enumerate(grid_sizes.tolist()):
            seq_len = f * h * w
            sf = start_frame[i].item() if isinstance(start_frame, torch.Tensor) else start_frame
            # precompute multipliers
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
            freqs_i = torch.cat([
                freqs[0][sf:sf + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
            ],
                dim=-1).reshape(seq_len, 1, -1)
            # apply rotary embedding
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])
            # append to collection
            output.append(x_i)
        return torch.stack(output).type_as(x)
    else:
        # ONNX/TensorRT-friendly vectorized version
        n, c = x.size(2), x.size(3) // 2
        freqs_split = [c - 2 * (c // 3), c // 3, c // 3]
        freqs_0, freqs_1, freqs_2 = torch.split(freqs, freqs_split, dim=1)
        f = grid_sizes[:, 0]
        h = grid_sizes[:, 1]
        w = grid_sizes[:, 2]
        seq_len = f * h * w
        if isinstance(start_frame, torch.Tensor):
            sf = start_frame
        else:
            sf = torch.full_like(f, start_frame)
        max_f = torch.amax(f, dim=0)
        max_h = torch.amax(h, dim=0)
        max_w = torch.amax(w, dim=0)
        max_seq_len = torch.amax(seq_len, dim=0)
        B = x.size(0)
        f_idx = torch.arange(max_f, device=x.device)
        h_idx = torch.arange(max_h, device=x.device)
        w_idx = torch.arange(max_w, device=x.device)
        f_grid, h_grid, w_grid = torch.meshgrid(f_idx, h_idx, w_idx, indexing='ij')
        f_grid = f_grid.unsqueeze(0).expand(B, -1, -1, -1)
        h_grid = h_grid.unsqueeze(0).expand(B, -1, -1, -1)
        w_grid = w_grid.unsqueeze(0).expand(B, -1, -1, -1)
        valid_mask = (f_grid < f.view(-1, 1, 1, 1)) & (h_grid < h.view(-1, 1, 1, 1)) & (w_grid < w.view(-1, 1, 1, 1))
        freq0_idx = sf.view(-1, 1, 1, 1) + f_grid
        freq1_idx = h_grid
        freq2_idx = w_grid
        freq0 = freqs_0[freq0_idx.clamp(max=freqs_0.size(0)-1)]
        freq1 = freqs_1[freq1_idx.clamp(max=freqs_1.size(0)-1)]
        freq2 = freqs_2[freq2_idx.clamp(max=freqs_2.size(0)-1)]
        freqs_i = torch.cat([freq0, freq1, freq2], dim=-1)
        freqs_i = freqs_i.view(B, -1, 1, freqs_i.size(-1))
        x_reshaped = x[:, :max_seq_len].to(torch.float64).reshape(B, max_seq_len, n, 2)
        x_complex = torch.view_as_complex(x_reshaped)
        x_complex = x_complex * freqs_i[..., :x_complex.size(-1)]
        x_real = torch.view_as_real(x_complex).flatten(2)
        x_cat = torch.cat([x_real, x[:, max_seq_len:]], dim=1)
        x_cat = x_cat * valid_mask.view(B, -1, 1).to(x_cat.dtype)
        return x_cat.type_as(x)


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        
        self.sink_size = 0

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, current_end=0):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
            roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)
            if not USE_TENSORRT_EXPORT:
                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            else:
                # ONNX/TensorRT-friendly: use tensor ops only
                padded_length = ((q.shape[1] + 127) // 128) * 128 - q.shape[1]
            padded_roped_query = torch.cat(
                [roped_query,
                 torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                             device=q.device, dtype=v.dtype)],
                dim=1
            )
            padded_roped_key = torch.cat(
                [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                        device=k.device, dtype=v.dtype)],
                dim=1
            )
            padded_v = torch.cat(
                [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                device=v.device, dtype=v.dtype)],
                dim=1
            )
            x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask
            )[:, :, :-padded_length].transpose(2, 1)
        else:
            if not USE_TENSORRT_EXPORT:
                frame_seqlen = math.prod(grid_sizes[0][1:]).item()
                current_start_frame = current_start // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
                seq_lens = []
                for i, c_start in enumerate(current_start):
                    current_end = c_start + roped_query.shape[1]
                    sink_tokens = self.sink_size * frame_seqlen
                    kv_cache_size = kv_cache["k"].shape[1]
                    num_new_tokens = roped_query.shape[1]
                    if c_start + num_new_tokens >= kv_cache_size:
                        kv_cache["global_end_index"][i].fill_(c_start)
                        kv_cache["local_end_index"][i].fill_(kv_cache_size)
                    if (current_end > kv_cache["global_end_index"][i].item()) and (
                            num_new_tokens + kv_cache["local_end_index"][i].item() > kv_cache_size):
                        num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"][i].item() - kv_cache_size
                        num_rolled_tokens = kv_cache["local_end_index"][i].item() - num_evicted_tokens - sink_tokens
                        kv_cache["k"][i:i+1, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["k"][i:i+1, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        kv_cache["v"][i:i+1, sink_tokens:sink_tokens + num_rolled_tokens] = \
                            kv_cache["v"][i:i+1, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                        local_end_index = kv_cache["local_end_index"][i].item() + current_end - \
                            kv_cache["global_end_index"][i].item() - num_evicted_tokens
                    else:
                        local_end_index = kv_cache["local_end_index"][i].item() + current_end - kv_cache["global_end_index"][i].item()
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][i:i+1, local_start_index:local_end_index] = roped_key[i:i+1]
                    kv_cache["v"][i:i+1, local_start_index:local_end_index] = v[i:i+1]
                    seq_lens.append(local_end_index)
                    kv_cache["global_end_index"][i].fill_(current_end)
                    kv_cache["local_end_index"][i].fill_(local_end_index)
                seq_lens = torch.tensor(seq_lens, dtype=torch.int32, device=roped_query.device)
                x = flash_attn_interface.flash_attn_with_kvcache(
                    q=roped_query,
                    k_cache=kv_cache["k"][:, :seq_lens.max()],
                    v_cache=kv_cache["v"][:, :seq_lens.max()],
                    cache_seqlens=seq_lens,
                )
            else:
                # ONNX/TensorRT export: skip cache logic, just return zeros or a stub
                # (ONNX export is for stateless inference, not incremental decoding)
                B, L, N, D = q.shape
                x = torch.zeros(B, L, N * D, device=q.device, dtype=q.dtype)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 cross_attn_type,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, window_size, qk_norm,
                                                eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim,
                                                                      num_heads,
                                                                      (-1, -1),
                                                                      qk_norm,
                                                                      eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        current_end=0
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen))
             * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, current_end)

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen))
                 * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                 frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(
            self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) *
            (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
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
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        cross_attn_type = 't2v_cross_attn' if model_type == 't2v' else 'i2v_cross_attn'
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(cross_attn_type, dim, ffn_dim, num_heads,
                                    window_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        if model_type == 'i2v':
            self.img_emb = MLPProj(1280, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None

        self.num_frame_per_block = 1

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        current_end: int = 0,
        block_mode: str = 'input',
        block_num: int = [-1],
        patched_x_shape: torch.Tensor = None,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if block_mode == 'input':
            if y is not None:
                x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

            # embeddings
            x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
            bsz, cch, tlen, hh, ww = x[0].shape
            patched_x_shape = torch.tensor([bsz, cch, tlen, hh, ww], dtype=torch.int64, device=device)
        else:
            bsz, cch, tlen, hh, ww = [int(i) for i in patched_x_shape.tolist()]
            x = [u.permute(1,0).reshape(bsz, cch, tlen, hh, ww) for u in x]
            
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat(x)
        """
        torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        """

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                assert False
            else:
                if (block_mode == 'output' or block_mode == 'middle') and block_index < block_num[0]:
                    continue
                if (block_mode == 'input' or block_mode == 'middle') and block_index == block_num[-1]:
                    return x, patched_x_shape
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "current_end": current_end
                    }
                )
                x = block(x, **kwargs)
        if block_mode == 'input' and block_num[-1] == len(self.blocks):
            return x, patched_x_shape

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        if USE_TENSORRT_EXPORT:
            # ONNX/TensorRT branch: unpatchify returns (tensor, F_ps0, H_ps1, W_ps2)
            x, F_ps0, H_ps1, W_ps2 = self.unpatchify(x, grid_sizes)
            return x, F_ps0, H_ps1, W_ps2
        else:
            x = self.unpatchify(x, grid_sizes)
            return torch.stack(x)

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        if self.block_mask is None:
            self.block_mask = self._prepare_blockwise_causal_attn_mask(
                device, num_frames=x.shape[2],
                frame_seqlen=x.shape[-2] *
                x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                num_frame_per_block=self.num_frame_per_block
            )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])

        # time embeddings
        # with amp.autocast(dtype=torch.float32):
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
        e0 = self.time_projection(e).unflatten(
            1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
        # assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask)

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        # unpatchify
        if USE_TENSORRT_EXPORT:
            # ONNX/TensorRT branch: unpatchify returns (tensor, F_ps0, H_ps1, W_ps2)
            x, F_ps0, H_ps1, W_ps2 = self.unpatchify(x, grid_sizes)
            return x, F_ps0, H_ps1, W_ps2
        else:
            x = self.unpatchify(x, grid_sizes)
            return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        if not USE_TENSORRT_EXPORT:
            out = []
            for u, v in zip(x, grid_sizes.tolist()):
                u = u[:math.prod(v)].view(*v, *self.patch_size, c)
                u = torch.einsum('fhwpqrc->cfphqwr', u)
                u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
                out.append(u)
            return out
        else:
            # ONNX/TensorRT-friendly vectorized version (fully tensorized, no Python for-loops)
            # x: [B, L, ...], grid_sizes: [B, 3]
            B = x.shape[0]
            v = grid_sizes  # [B, 3]
            L = x.shape[1]
            patch_size = torch.tensor(self.patch_size, device=x.device)
            prod_v = v.prod(dim=1)  # [B]
            idx = torch.arange(L, device=x.device).unsqueeze(0) < prod_v.unsqueeze(1)  # [B, L]
            x_masked = x * idx.unsqueeze(-1).to(x.dtype)
            F, H, W = v[:, 0], v[:, 1], v[:, 2]
            max_F, max_H, max_W = F.max(), H.max(), W.max()
            pad_L = max_F * max_H * max_W - L
            if pad_L > 0:
                x_masked = torch.cat([x_masked, torch.zeros(B, pad_L, x.shape[2], device=x.device, dtype=x.dtype)], dim=1)
            x_reshaped = x_masked.view(B, max_F, max_H, max_W, *self.patch_size, c)
            x_ein = torch.einsum('bfhwpqrc->bcfphqwr', x_reshaped)
            F_ps0 = F * self.patch_size[0]
            H_ps1 = H * self.patch_size[1]
            W_ps2 = W * self.patch_size[2]
            max_Fps0 = F_ps0.max()
            max_Hps1 = H_ps1.max()
            max_Wps2 = W_ps2.max()
            mask_F = torch.arange(max_Fps0, device=x.device).unsqueeze(0) < F_ps0.unsqueeze(1)  # [B, max_Fps0]
            mask_H = torch.arange(max_Hps1, device=x.device).unsqueeze(0) < H_ps1.unsqueeze(1)  # [B, max_Hps1]
            mask_W = torch.arange(max_Wps2, device=x.device).unsqueeze(0) < W_ps2.unsqueeze(1)  # [B, max_Wps2]
            mask_F = mask_F.unsqueeze(2).unsqueeze(3)
            mask_H = mask_H.unsqueeze(1).unsqueeze(3)
            mask_W = mask_W.unsqueeze(1).unsqueeze(2)
            mask = mask_F & mask_H & mask_W  # [B, max_Fps0, max_Hps1, max_Wps2]
            x_ein_padded = torch.zeros(B, c, max_Fps0, max_Hps1, max_Wps2, device=x_ein.device, dtype=x_ein.dtype)
            x_ein_padded = x_ein_padded + x_ein * mask.to(x_ein.dtype)
            return x_ein_padded, F_ps0, H_ps1, W_ps2

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
