# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from einops import rearrange, repeat
from timm.models.vision_transformer import Mlp, PatchEmbed
from torch.utils.checkpoint import checkpoint
# the xformers lib allows less memory, faster training and inference
try:
    import xformers
    import xformers.ops
except:
    XFORMERS_IS_AVAILBLE = False

# from timm.models.layers.helpers import to_2tuple
# from timm.models.layers.trace_utils import _assert

def modulate(x, shift, scale):
    return x * (1 + scale) + shift

def matrix_mul(x, u, w):
    return torch.einsum('nm,...nd,dk->...mk', u, x, w)

def matrix_mul_softmax(x, u, w):
    return torch.einsum('nm,...nd,dk->...mk', torch.nn.functional.softmax(u, dim=0), x, w)

def matrix_mul_normalized_l2(x, u, w):
    return torch.einsum('nm,...nd,dk->...mk', u/torch.linalg.norm(u, dim=0, keepdim=True, ord=2) + 1e-6, x, w)

def matrix_mul_normalized_l1(x, u, w):
    return torch.einsum('nm,...nd,dk->...mk', u/torch.linalg.norm(u, dim=0, keepdim=True, ord=1) + 1e-6, x, w)

def matrix_mul_one_side(x, w):
    return torch.einsum('...nd,dk->...nk', x, w)

#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False, attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous() # 3, B, num_heads, N, head_dim
        q, k, v = qkv.unbind(0)   # make torchscript happy (cannot use tensor as tuple)
        if self.attention_mode == 'xformers': # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1,2).contiguous()
            k_xf = k.transpose(1,2).contiguous()
            v_xf = v.transpose(1,2).contiguous()
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(B, N, C)

        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C) # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def __repr__(self):
        return f"{self.__class__.__name__}(dim, num_heads={self.num_heads}, attention_mode={self.attention_mode})"


class MatrixLinear(nn.Module):
    def __init__(
        self,
        in_features: Tuple[int, int],
        out_features: Tuple[int, int],
        bias=True,
        bias_type='matrix', # matrix, row, col
        u_type='param',
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias_type = bias_type
        self.u_type = u_type

        match u_type:
            case 'param' | 'softmax' | 'normalized_l1' | 'normalized_l2':
                self.u = nn.Parameter(torch.empty((in_features[0], out_features[0]), **factory_kwargs))
            case 'identity':
                assert in_features[0] == out_features[0], f"in_features[0] size {in_features[0]} must be equal to out_features[0] size {out_features[0]} for identity u"
                u = torch.eye(in_features[0], **factory_kwargs)
                self.register_buffer('u', u)
            case _:
                raise NotImplementedError(f"Unknown u_type: {u_type}")

        if u_type == 'softmax':
            self.u_temperature = nn.Parameter(torch.ones((1, out_features[0]), **factory_kwargs))

        self.w = nn.Parameter(torch.empty((in_features[1], out_features[1]), **factory_kwargs))

        if bias:
            match bias_type:
                case 'matrix':
                    self.bias = nn.Parameter(torch.empty(out_features[0], out_features[1], **factory_kwargs))
                case 'row':
                    self.bias = nn.Parameter(torch.empty(1, out_features[1], **factory_kwargs))
                case 'col':
                    self.bias = nn.Parameter(torch.empty(out_features[0], 1, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.u_type == 'softmax':
            x = matrix_mul_softmax(input, self.u / self.u_temperature, self.w)
        elif self.u_type == 'normalized_l1':
            x = matrix_mul_normalized_l1(input, self.u, self.w)
        elif self.u_type == 'normalized_l2':
            x = matrix_mul_normalized_l2(input, self.u, self.w)
        else:
            x = matrix_mul(input, self.u, self.w)
        # x = matrix_mul_one_side(input, self.w)
        
        if self.bias is not None:
            x += self.bias
        return x
    
    def __repr__(self):
        return f"{self.__class__.__name__}({self.in_features} -> {self.out_features}, u_type={self.u_type}, bias_type={self.bias_type})"


class MatrixAttention(nn.Module):
    """
    Matrix attention block.
    This is a simplified version of the attention block that does not use RoPE.
    It is used in the DiT model for the final layer.
    """
    def __init__(
        self, 
        col_dim: int,
        row_dim: int,
        qk_col_dim: Optional[int] = None,
        v_col_dim: Optional[int] = None,
        num_col_heads: int = 4,
        num_row_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias=True,
        bias_type='matrix',
        u_type='param',
        attention_mode='math',
    ):
        super().__init__()
        assert qk_col_dim % num_col_heads == 0, "qk_col_dim must be divisible by num_col_heads"
        assert v_col_dim % num_col_heads == 0, "v_col_dim must be divisible by num_col_heads"
        assert row_dim % num_row_heads == 0, "row_dim must be divisible by num_row_heads"


        self.col_dim = col_dim
        self.row_dim = row_dim

        self.qk_col_dim = qk_col_dim
        self.v_col_dim = v_col_dim

        self.num_col_heads = num_col_heads
        self.num_row_heads = num_row_heads
        self.num_heads = num_col_heads * num_row_heads
        self.use_bias = use_bias

        self.head_row_dim = row_dim // num_row_heads
        self.qk_head_col_dim = self.qk_col_dim // num_col_heads
        self.v_head_col_dim = self.v_col_dim // num_col_heads

        self.linear_q = MatrixLinear((self.col_dim, self.row_dim), (self.qk_col_dim, self.row_dim), bias=self.use_bias, bias_type=bias_type, u_type=u_type)
        self.linear_k = MatrixLinear((self.col_dim, self.row_dim), (self.qk_col_dim, self.row_dim), bias=self.use_bias, bias_type=bias_type, u_type=u_type)
        self.linear_v = MatrixLinear((self.col_dim, self.row_dim), (self.v_col_dim, self.row_dim), bias=self.use_bias, bias_type=bias_type, u_type=u_type)
        self.proj_v = MatrixLinear((self.v_col_dim, self.row_dim), (self.col_dim, self.row_dim), bias=self.use_bias, bias_type=bias_type, u_type=u_type)

        self.scale = (self.qk_head_col_dim*self.head_row_dim)**-0.5

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        self.attention_mode = attention_mode

    def forward(
        self, 
        x: torch.Tensor, 
        timestep=None,
        height: int = None,
        width: int = None
    ) -> torch.Tensor:
        if hasattr(self, "store_attn_map"):
            raise NotImplementedError("MatrixAttention does not support storing attention maps.")

        # B: batch_size
        # T: n_frames
        # N: n_tokens per frame
        # D: dim of a token
        B, T, N, D = x.shape

        q = self.linear_q(x)  # B, T, qk, row_dim
        k = self.linear_k(x)  # B, T, N, row_dim
        v = self.linear_v(x)  # B, T, N, row_dim

        # Rearrange
        q = rearrange(q, 'B T (C N) (R D) -> B (C R) T (N D)', B=B, T=T, C=self.num_col_heads, R=self.num_row_heads, N=self.qk_head_col_dim, D=self.head_row_dim)
        k = rearrange(k, 'B T (C N) (R D) -> B (C R) T (N D)', B=B, T=T, C=self.num_col_heads, R=self.num_row_heads, N=self.qk_head_col_dim, D=self.head_row_dim)
        v = rearrange(v, 'B T (C N) (R D) -> B (C R) T (N D)', B=B, T=T, C=self.num_col_heads, R=self.num_row_heads, N=self.v_head_col_dim, D=self.head_row_dim)

        if self.attention_mode == 'xformers':
           raise NotImplementedError("MatrixAttention does not support xformers attention mode.")
        elif self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, T, N*D) # (B, col_num_head * row_num_head, num_tokens, N*D)
        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, col_num_head * row_num_head, T, T)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v) # B, col_num_head * row_num_head, T, N*D
            x = rearrange(x, 'B (C R) T (N D) -> B T (C N) (R D)', C=self.num_col_heads, R=self.num_row_heads, N=self.v_head_col_dim, D=self.head_row_dim)
        else:
            raise NotImplementedError(f"Unknown attention mode: {self.attention_mode}")

        x = self.proj_v(x)
        x = self.proj_drop(x)

        return x


class FusedMatrixAttention(nn.Module):
    def __init__(
        self,
        col_dim: int,
        row_dim: int,
        qk_col_dim: Optional[int] = None,
        v_col_dim: Optional[int] = None,
        num_col_heads: int = 4,
        num_row_heads: int = 4,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        use_bias=True,
        bias_type='matrix',
        u_type='param',
        fuse_mode='gated',
        attention_mode='math',
    ):
        """
        Attention: local attention on rows
        MatrixAttention: global attention on columns
        """
        super().__init__()
        self.vanilla_attention = Attention(
            dim=row_dim,
            num_heads=num_row_heads,
            qkv_bias=use_bias,
            attention_mode=attention_mode,
        )
        
        self.fuse_mode = fuse_mode
        assert fuse_mode in ['gated', 'sum', 'concat', 'local'], f"Unknown fuse mode: {fuse_mode}"

        if self.fuse_mode != 'local':
            self.matrix_attention = MatrixAttention(
                col_dim=col_dim,
                row_dim=row_dim,
                qk_col_dim=qk_col_dim,
                v_col_dim=v_col_dim,
                num_col_heads=num_col_heads,
                num_row_heads=num_row_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                use_bias=use_bias,
                bias_type=bias_type,
                u_type=u_type,
                attention_mode=attention_mode,
            )
            self.norm_local  = nn.LayerNorm(row_dim)
            self.norm_global = nn.LayerNorm(row_dim)

        if self.fuse_mode == 'gated':
            self.alpha = nn.Parameter(torch.zeros(row_dim)) if fuse_mode else None
            self.gamma_local  = nn.Parameter(1e-4 * torch.ones(row_dim))
            self.gamma_global = nn.Parameter(1e-4 * torch.ones(row_dim))
        elif self.fuse_mode == 'concat':
            self.output_linear = nn.Linear(2 * row_dim, row_dim)
    

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        local_x = self.vanilla_attention(rearrange(x, 'B T N D -> (B N) T D'))
        local_x = rearrange(local_x, '(B N) T D -> B T N D', B=B, N=N)
        if self.fuse_mode == 'local':
            return local_x

        local_x = self.norm_local(local_x)
        global_x = self.matrix_attention(x)
        global_x = self.norm_global(global_x)

        if self.fuse_mode == 'gated':
            alpha = torch.sigmoid(self.alpha).view(1, 1, 1, D)
            x = self.gamma_local.view(1,1,1,D) * local_x * alpha \
              + self.gamma_global.view(1,1,1,D) * global_x * (1 - alpha)
        elif self.fuse_mode == 'sum':
            x = (local_x + global_x)*1/math.sqrt(2)
        elif self.fuse_mode == 'concat':
            x = torch.cat([local_x, global_x], dim=-1)
            x = self.output_linear(x)
        else:
            raise NotImplementedError(f"Unknown fuse mode: {self.fuse_mode}")
        return x
    
    def __repr__(self):
        # return col_dim, row_dim, qk_col_dim, v_col_dim, num_col_heads, num_row_heads, fuse_mode, attention_mode
        # newline for better readability
        if self.fuse_mode == 'gated':
            return f"{self.__class__.__name__}(col_dim={self.matrix_attention.col_dim}, row_dim={self.matrix_attention.row_dim}, qk_col_dim={self.matrix_attention.qk_col_dim}, v_col_dim={self.matrix_attention.v_col_dim},\n" \
                   f" num_col_heads={self.matrix_attention.num_col_heads}, num_row_heads={self.matrix_attention.num_row_heads}, fuse_mode={self.fuse_mode}, attention_mode={self.vanilla_attention.attention_mode})"
        elif self.fuse_mode == 'sum':
            return f"{self.__class__.__name__}(col_dim={self.matrix_attention.col_dim}, row_dim={self.matrix_attention.row_dim}, qk_col_dim={self.matrix_attention.qk_col_dim}, v_col_dim={self.matrix_attention.v_col_dim},\n" \
                   f" num_col_heads={self.matrix_attention.num_col_heads}, num_row_heads={self.matrix_attention.num_row_heads}, fuse_mode={self.fuse_mode}, attention_mode={self.vanilla_attention.attention_mode})"
        elif self.fuse_mode == 'concat':
            return f"{self.__class__.__name__}(col_dim={self.matrix_attention.col_dim}, row_dim={self.matrix_attention.row_dim}, qk_col_dim={self.matrix_attention.qk_col_dim}, v_col_dim={self.matrix_attention.v_col_dim},\n" \
                   f" num_col_heads={self.matrix_attention.num_col_heads}, num_row_heads={self.matrix_attention.num_row_heads}, fuse_mode={self.fuse_mode}, attention_mode={self.vanilla_attention.attention_mode})"
        elif self.fuse_mode == 'local':
            return f"{self.__class__.__name__}(num_row_heads={self.vanilla_attention.num_heads}, attention_mode={self.vanilla_attention.attention_mode})"

#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings


#################################################################################
#                                 Core FusedMatLatte Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A FusedMatLatte tansformer block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
    

class FusedMatrixTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int, 
        col_dim: int, 
        qk_col_dim: int, 
        v_col_dim: int,
        num_col_heads: int, 
        num_row_heads: int,
        mlp_ratio=4.0, 
        **block_kwargs
    ):
        super().__init__()
        self.col_dim = col_dim
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = FusedMatrixAttention(
            row_dim=hidden_size,
            col_dim=col_dim,
            qk_col_dim=qk_col_dim,
            v_col_dim=v_col_dim,
            num_col_heads=num_col_heads,
            num_row_heads=num_row_heads,
            **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        """
        x: (B*N, F, D)
        c: (B*N, F, D)
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=-1)
        attn_x = self.attn(rearrange(modulate(self.norm1(x), shift_msa, scale_msa), '(b n) f d -> b f n d', n=self.col_dim))
        x = x + gate_msa * rearrange(attn_x, 'b f n d -> (b n) f d')
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """
    The final layer of FusedMatLatte.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class FrameDiT_H_IMG(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        hidden_size=1152,
        qk_col_dim=64,
        v_col_dim=64,
        depth=28,
        num_col_heads=16,
        num_row_heads=16,
        mlp_ratio=4.0,
        num_frames=16,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        extras=1,
        attention_mode='math',
        fuse_mode='gated',
        **kwargs
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.n_tokens_per_frames = (input_size // patch_size) ** 2
        self.num_col_heads = num_col_heads
        self.num_row_heads = num_row_heads
        self.extras = extras
        self.num_frames = num_frames

        self.x_embedder = PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
        self.t_embedder = TimestepEmbedder(hidden_size)

        if self.extras == 2:
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        if self.extras == 78: # timestep + text_embedding
            self.text_embedding_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(77 * 768, hidden_size, bias=True)
        )

        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)
        self.temp_embed = nn.Parameter(torch.zeros(1, num_frames, hidden_size), requires_grad=False)
        self.hidden_size =  hidden_size

        _transformer_block = lambda: TransformerBlock(hidden_size, num_row_heads, mlp_ratio=mlp_ratio, attention_mode=attention_mode)
        _mat_block = lambda: FusedMatrixTransformerBlock(
            hidden_size=hidden_size,
            col_dim=self.n_tokens_per_frames, 
            qk_col_dim=qk_col_dim,
            v_col_dim=v_col_dim,
            num_col_heads=num_col_heads,
            num_row_heads=num_row_heads,
            u_type=kwargs.get('u_type', 'param'),
            bias_type=kwargs.get('bias_type', 'matrix'),
            fuse_mode=fuse_mode,
        )

        self.blocks = nn.ModuleList([
            _transformer_block() if i % 2 == 0 else _mat_block() for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

        self.gradient_checkpointing = kwargs.get('gradient_checkpointing', False)

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, MatrixLinear):
                if isinstance(module.u, nn.Parameter):
                    torch.nn.init.xavier_uniform_(module.u)
                torch.nn.init.xavier_uniform_(module.w)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        temp_embed = get_1d_sincos_temp_embed(self.temp_embed.shape[-1], self.temp_embed.shape[-2])
        self.temp_embed.data.copy_(torch.from_numpy(temp_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        if self.extras == 2:
            # Initialize label embedding table:
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in MatLatte blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    # @torch.cuda.amp.autocast()
    # @torch.compile
    def forward(self, x, t, y=None, use_fp16=False, y_image=None, use_image_num=0):
        """
        Forward pass of FrameDiT_H_IMG.
        x: (N, F, C, H, W) tensor of video inputs
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        """
        if use_fp16:
            x = x.to(dtype=torch.float16)
        batches, frames, channels, high, weight = x.shape 
        x = rearrange(x, 'b f c h w -> (b f) c h w')
        x = self.x_embedder(x) + self.pos_embed  
        t = self.t_embedder(t, use_fp16=use_fp16)
        timestep_spatial = repeat(t, 'b d -> (b f) n d', n=self.pos_embed.shape[1], f=self.temp_embed.shape[1]+use_image_num) 
        timestep_temp = repeat(t, 'b d -> (b n) f d', n=self.pos_embed.shape[1], f=self.temp_embed.shape[1])

        if self.extras == 2:
            y = self.y_embedder(y, self.training)
            if self.training:
                y_image_emb = []
                for y_image_single in y_image:
                    y_image_single = y_image_single.unsqueeze(0) # 1, C, H, W
                    y_image_emb.append(self.y_embedder(y_image_single, self.training)) # 1, D
                y_image_emb = torch.cat(y_image_emb, dim=0)
                y_spatial = repeat(y, 'b d -> b f d', f=self.temp_embed.shape[1])
                y_spatial = torch.cat([y_spatial, y_image_emb], dim=1)
                y_spatial = repeat(y_spatial, 'b f d -> (b f) n d', n=self.pos_embed.shape[1])
            else:
                y_spatial = repeat(y, 'b d -> (b f) n d', n=self.pos_embed.shape[1], f=self.temp_embed.shape[1])
            y_temp = repeat(y, 'b d -> (b n) f d', n=self.pos_embed.shape[1], f=self.temp_embed.shape[1])
        elif self.extras == 78:
            text_embedding = self.text_embedding_projection(text_embedding.reshape(batches, -1))
            text_embedding_spatial = repeat(text_embedding, 'b d -> (b f) n d', n=self.pos_embed.shape[1], f=self.temp_embed.shape[1])
            text_embedding_temp = repeat(text_embedding, 'b d -> (b n) f d', n=self.pos_embed.shape[1], f=self.temp_embed.shape[1])

        for i in range(0, len(self.blocks), 2):
            spatial_block, temp_block = self.blocks[i:i+2]
            if self.extras == 2:
                c = timestep_spatial + y_spatial
            elif self.extras == 78:
                c = timestep_spatial + text_embedding_spatial
            else:
                c = timestep_spatial

            if self.gradient_checkpointing:
                x = checkpoint(spatial_block, x, c, use_reentrant=False)
            else:
                x = spatial_block(x, c)

            x = rearrange(x, '(b f) n d -> (b n) f d', b=batches)
            x_video = x[:, :(frames-use_image_num), :]
            x_image = x[:, (frames-use_image_num):, :]

            # Add Time Embedding
            if i == 0:
                x_video = x_video + self.temp_embed

            if self.extras == 2:
                c = timestep_temp + y_temp
            elif self.extras == 78:
                c = timestep_temp + text_embedding_temp
            else:
                c = timestep_temp

            x_video = temp_block(x_video, c)
            x = torch.cat([x_video, x_image], dim=1)

            x = rearrange(x, '(b n) f d -> (b f) n d', b=batches)

        if self.extras == 2:
            c = timestep_spatial + y_spatial
        else:
            c = timestep_spatial
        x = self.final_layer(x, c)               
        x = self.unpatchify(x)                  
        x = rearrange(x, '(b f) c h w -> b f c h w', b=batches)
        return x

    def forward_with_cfg(self, x, t, y=None, cfg_scale=7.0, use_fp16=False):
        """
        Forward pass of FrameDiT_H_IMG, but also batches the unconditional forward pass for classifier-free guidance.
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        if use_fp16:
            combined = combined.to(dtype=torch.float16)
        model_out = self.forward(combined, t, y=y, use_fp16=use_fp16)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # This can be done by uncommenting the following line and commenting-out the line following that.
        # eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        # eps, rest = model_out[:, :3], model_out[:, 3:]
        eps, rest = model_out[:, :, :4, ...], model_out[:, :, 4:, ...] 
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0) 
        return torch.cat([eps, rest], dim=2)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_1d_sincos_temp_embed(embed_dim, length):
    pos = torch.arange(0, length).unsqueeze(1)
    return get_1d_sincos_pos_embed_from_grid(embed_dim, pos)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0]) 
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1]) 

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega 

    pos = pos.reshape(-1)  
    out = np.einsum('m,d->md', pos, omega) 

    emb_sin = np.sin(out) 
    emb_cos = np.cos(out) 

    emb = np.concatenate([emb_sin, emb_cos], axis=1) 
    return emb


#============================================================================================
def FrameDiT_H_IMG_XL_2(**kwargs):
    return FrameDiT_H_IMG(depth=28, hidden_size=1152, patch_size=2, num_row_heads=16, **kwargs)

def FrameDiT_H_IMG_L_2(**kwargs):
    return FrameDiT_H_IMG(depth=24, hidden_size=1024, patch_size=2, num_row_heads=16, **kwargs)

def FrameDiT_H_IMG_B_2(**kwargs):
    return FrameDiT_H_IMG(depth=12, hidden_size=768, patch_size=2, num_row_heads=12, **kwargs)

def FrameDiT_H_IMG_S_2(**kwargs):
    return FrameDiT_H_IMG(depth=12, hidden_size=384, patch_size=2, num_row_heads=6, **kwargs)


def FrameDiT_H_IMG_S_64_256_2_softmax_u_concat(**kwargs):
    return FrameDiT_H_IMG_S_2(qk_col_dim=64, v_col_dim=256, num_col_heads=64, fuse_mode='concat', u_type='softmax', **kwargs)

def FrameDiT_H_IMG_B_64_256_2_softmax_u_concat(**kwargs):
    return FrameDiT_H_IMG_B_2(qk_col_dim=64, v_col_dim=256, num_col_heads=64, fuse_mode='concat', u_type='softmax', **kwargs)

def FrameDiT_H_IMG_L_64_256_2_softmax_u_concat(**kwargs):
    return FrameDiT_H_IMG_L_2(qk_col_dim=64, v_col_dim=256, num_col_heads=64, fuse_mode='concat', u_type='softmax', **kwargs)

def FrameDiT_H_IMG_XL_64_256_2_softmax_u_concat(**kwargs):
    return FrameDiT_H_IMG_XL_2(qk_col_dim=64, v_col_dim=256, num_col_heads=64, fuse_mode='concat', u_type='softmax', **kwargs)


def FrameDiT_H_IMG_S_64_256_2_concat(**kwargs):
    return FrameDiT_H_IMG_S_2(qk_col_dim=64, v_col_dim=256, num_col_heads=64, fuse_mode='concat', **kwargs)

def FrameDiT_H_IMG_B_64_256_2_concat(**kwargs):
    return FrameDiT_H_IMG_B_2(qk_col_dim=64, v_col_dim=256, num_col_heads=64, fuse_mode='concat', **kwargs)

def FrameDiT_H_IMG_L_64_256_2_concat(**kwargs):
    return FrameDiT_H_IMG_L_2(qk_col_dim=64, v_col_dim=256, num_col_heads=64, fuse_mode='concat', **kwargs)

def FrameDiT_H_IMG_XL_64_256_2_concat(**kwargs):
    return FrameDiT_H_IMG_XL_2(qk_col_dim=64, v_col_dim=256, num_col_heads=64, fuse_mode='concat', **kwargs)


def FrameDiT_H_IMG_S_128_512_2_softmax_u_concat(**kwargs):
    return FrameDiT_H_IMG_S_2(qk_col_dim=128, v_col_dim=512, num_col_heads=64, fuse_mode='concat', u_type='softmax', **kwargs)

def FrameDiT_H_IMG_B_128_512_2_softmax_u_concat(**kwargs):
    return FrameDiT_H_IMG_B_2(qk_col_dim=128, v_col_dim=512, num_col_heads=64, fuse_mode='concat', u_type='softmax', **kwargs)

def FrameDiT_H_IMG_L_128_512_2_softmax_u_concat(**kwargs):
    return FrameDiT_H_IMG_L_2(qk_col_dim=128, v_col_dim=512, num_col_heads=64, fuse_mode='concat', u_type='softmax', **kwargs)

def FrameDiT_H_IMG_XL_128_512_2_softmax_u_concat(**kwargs):
    return FrameDiT_H_IMG_XL_2(qk_col_dim=128, v_col_dim=512, num_col_heads=64, fuse_mode='concat', u_type='softmax', **kwargs)


def FrameDiT_H_IMG_S_128_512_2_concat(**kwargs):
    return FrameDiT_H_IMG_S_2(qk_col_dim=128, v_col_dim=512, num_col_heads=64, fuse_mode='concat', **kwargs)

def FrameDiT_H_IMG_B_128_512_2_concat(**kwargs):
    return FrameDiT_H_IMG_B_2(qk_col_dim=128, v_col_dim=512, num_col_heads=64, fuse_mode='concat', **kwargs)

def FrameDiT_H_IMG_L_128_512_2_concat(**kwargs):
    return FrameDiT_H_IMG_L_2(qk_col_dim=128, v_col_dim=512, num_col_heads=64, fuse_mode='concat', **kwargs)

def FrameDiT_H_IMG_XL_128_512_2_concat(**kwargs):
    return FrameDiT_H_IMG_XL_2(qk_col_dim=128, v_col_dim=512, num_col_heads=64, fuse_mode='concat', **kwargs)

#============================================================================================
FrameDiTHIMG_models = {
    'FrameDiTHIMG-S/64-256/2-softmax_u-concat': FrameDiT_H_IMG_S_64_256_2_softmax_u_concat,
    'FrameDiTHIMG-B/64-256/2-softmax_u-concat': FrameDiT_H_IMG_B_64_256_2_softmax_u_concat,
    'FrameDiTHIMG-L/64-256/2-softmax_u-concat': FrameDiT_H_IMG_L_64_256_2_softmax_u_concat,
    'FrameDiTHIMG-XL/64-256/2-softmax_u-concat': FrameDiT_H_IMG_XL_64_256_2_softmax_u_concat,

    'FrameDiTHIMG-S/64-256/2-concat': FrameDiT_H_IMG_S_64_256_2_concat,
    'FrameDiTHIMG-B/64-256/2-concat': FrameDiT_H_IMG_B_64_256_2_concat,
    'FrameDiTHIMG-L/64-256/2-concat': FrameDiT_H_IMG_L_64_256_2_concat,
    'FrameDiTHIMG-XL/64-256/2-concat': FrameDiT_H_IMG_XL_64_256_2_concat,

    'FrameDiTHIMG-S/128-512/2-softmax_u-concat': FrameDiT_H_IMG_S_128_512_2_softmax_u_concat,
    'FrameDiTHIMG-B/128-512/2-softmax_u-concat': FrameDiT_H_IMG_B_128_512_2_softmax_u_concat,
    'FrameDiTHIMG-L/128-512/2-softmax_u-concat': FrameDiT_H_IMG_L_128_512_2_softmax_u_concat,
    'FrameDiTHIMG-XL/128-512/2-softmax_u-concat': FrameDiT_H_IMG_XL_128_512_2_softmax_u_concat,

    'FrameDiTHIMG-S/128-512/2-concat': FrameDiT_H_IMG_S_128_512_2_concat,
    'FrameDiTHIMG-B/128-512/2-concat': FrameDiT_H_IMG_B_128_512_2_concat,
    'FrameDiTHIMG-L/128-512/2-concat': FrameDiT_H_IMG_L_128_512_2_concat,
    'FrameDiTHIMG-XL/128-512/2-concat': FrameDiT_H_IMG_XL_128_512_2_concat,

    'FusedMatLatteIMG-XL/128-512/2-concat': FrameDiT_H_IMG_XL_128_512_2_concat  
}
