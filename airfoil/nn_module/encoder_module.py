import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_
from .attention_module import pair, PreNorm, PostNorm,\
    StandardAttention, FeedForward, LinearAttention, ReLUFeedForward
from .cnn_module import PeriodicConv2d, PeriodicConv3d, UpBlock
#from .gnn_module import SmoothConvEncoder, SmoothConvDecoder, index_points
#from torch_scatter import scatter
# helpers


class TransformerCatNoCls(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim,  FeedForward(dim, mlp_dim, dropout=dropout)
                                  if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or not cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                   heads=heads, dim_head=dim_head, dropout=dropout,
                                                   relative_emb=True, scale=scale[d],
                                                   relative_emb_dim=relative_emb_dim,
                                                   min_freq=min_freq,
                                                   init_method=attention_init,
                                                   init_gain=init_gain,
                                                   use_ln=False,
                                                   )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([
                                        attn_module,
                                        FeedForward(dim, mlp_dim, dropout=dropout)
                                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                        )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x, pos_embedding):
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding) + x
                x = ln2(x)
                x = ffn(x) + x
        return x


class TransformerWithPad(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 mlp_dim,
                 attn_type,  # ['standard', 'galerkin', 'fourier']
                 use_ln=False,
                 scale=16,     # can be list, or an int
                 dropout=0.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 attention_init='orthogonal',
                 init_gain=None,
                 use_relu=False,
                 cat_pos=False,
                 ):
        super().__init__()
        assert attn_type in ['standard', 'galerkin', 'fourier']

        if isinstance(scale, int):
            scale = [scale] * depth
        assert len(scale) == depth

        self.layers = nn.ModuleList([])
        self.attn_type = attn_type
        self.use_ln = use_ln

        if attn_type == 'standard':
            for _ in range(depth):
                self.layers.append(
                    nn.ModuleList([
                    PreNorm(dim, StandardAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim,  FeedForward(dim, mlp_dim, dropout=dropout)
                                  if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout))]),
                )
        else:
            for d in range(depth):
                if scale[d] != -1 or not cat_pos:
                    attn_module = LinearAttention(dim, attn_type,
                                                   heads=heads, dim_head=dim_head, dropout=dropout,
                                                   relative_emb=True, scale=scale[d],
                                                   relative_emb_dim=relative_emb_dim,
                                                   min_freq=min_freq,
                                                   init_method=attention_init,
                                                   init_gain=init_gain,
                                                   use_ln=True
                                                   )
                else:
                    attn_module = LinearAttention(dim, attn_type,
                                                  heads=heads, dim_head=dim_head, dropout=dropout,
                                                  cat_pos=True,
                                                  pos_dim=relative_emb_dim,
                                                  relative_emb=False,
                                                  init_method=attention_init,
                                                  init_gain=init_gain
                                                  )
                if not use_ln:
                    self.layers.append(
                        nn.ModuleList([
                                        attn_module,
                                        FeedForward(dim, mlp_dim, dropout=dropout)
                                        if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout)
                        ]),
                        )
                else:
                    self.layers.append(
                        nn.ModuleList([
                            nn.LayerNorm(dim),
                            attn_module,
                            nn.LayerNorm(dim),
                            FeedForward(dim, mlp_dim, dropout=dropout)
                            if not use_relu else ReLUFeedForward(dim, mlp_dim, dropout=dropout),
                        ]),
                    )

    def forward(self, x, pos_embedding, pad_mask):
        # x in [b n c], pos_embedding in [b n 2]
        b, n, c = x.shape

        for layer_no, attn_layer in enumerate(self.layers):
            if not self.use_ln:
                [attn, ffn] = attn_layer

                x = attn(x, pos_embedding, padding_mask=pad_mask) + x
                x = ffn(x) + x
            else:
                [ln1, attn, ln2, ffn] = attn_layer
                x = ln1(x)
                x = attn(x, pos_embedding, padding_mask=pad_mask) + x
                x = ln2(x)
                x = ffn(x) + x
        return x


class SimpleAttentionEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 n_patch=16,
                 out_grid=64,
                 emb_dropout=0.1,           # dropout of embedding
                 attention_init='xavier',
                 ):
        super().__init__()
        self.n_patch = n_patch
        self.out_grid = out_grid

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.temp_embedding = nn.Parameter(
            torch.cat((torch.tensor([-1.]), torch.linspace(0, 1, t)), dim=0).view(1, t+1, 1), requires_grad=False)   # [b, t, 1]
        self.cls_token = nn.Parameter(torch.randn(1, 1, in_emb_dim), requires_grad=True)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, 2), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.st_transformer = STTransformerCat(in_emb_dim, 1, 4, 64, in_emb_dim, 'galerkin', attention_init=attention_init)

        self.s_transformer = TransformerCat(in_emb_dim*2, depth-1, 4, 64, in_emb_dim*2, 'galerkin', attention_init=attention_init)

        self.to_cls = nn.Sequential(
            nn.LayerNorm(2*in_emb_dim),
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=True)
            )

        self.shrink_temporal = nn.Sequential(
            Rearrange('b n t c -> b n (t c)'),
            nn.Linear(t*in_emb_dim, 2*in_emb_dim, bias=False),
        )
        self.expand_feat = nn.Linear(in_emb_dim, 2*in_emb_dim)

        self.project_to_latent = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, c, t, n]
                input_pos,  # [b, n, 2]
                ):
        x = rearrange(x, 'b c t n-> b t n c')
        x = self.to_embedding(x)
        x = self.dropout(x)

        x, x_cls = self.st_transformer.forward(x,
                                             self.cls_token,
                                             self.temp_embedding,
                                             input_pos)
        x = self.shrink_temporal(x)
        x_cls = self.expand_feat(x_cls)
        x, x_cls = self.s_transformer.forward(x, x_cls, input_pos, self.cls_emb)
        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma

        return x, x_cls


class NoSTAttentionEncoder(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 seq_len,                  # this should be the input sequence length
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.1,           # dropout of embedding
                 ):
        super().__init__()

        t = seq_len
        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.cls_token = nn.Parameter(torch.randn(1, 2*in_emb_dim), requires_grad=True)
        self.cls_emb = nn.Parameter(torch.randn(1, 1, 2), requires_grad=True)
        self.gamma = nn.Parameter(torch.tensor([0.1]), requires_grad=True)

        self.shrink_temporal = nn.Sequential(
            Rearrange('b t n c -> b n (t c)'),
            nn.Linear(t * in_emb_dim, 2 * in_emb_dim, bias=False),
        )
        self.s_transformer = TransformerCat(in_emb_dim*2, depth, 4, 64, in_emb_dim*2, 'galerkin', init_scale=32)

        self.to_cls = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.LayerNorm(out_seq_emb_dim))

        self.project_to_latent = nn.Sequential(
            nn.Linear(2*in_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(out_seq_emb_dim, out_seq_emb_dim, bias=False),
            nn.InstanceNorm1d(in_emb_dim))

    def forward(self,
                x,  # [b, c, t, n]
                input_pos,  # [b, n, 2]
                ):
        x = rearrange(x, 'b c t n-> b t n c')
        x = self.to_embedding(x)
        x = self.dropout(x)
        x = self.shrink_temporal(x)

        x, x_cls = self.s_transformer.forward(x, self.cls_token, input_pos, self.cls_emb)
        x, x_cls = self.project_to_latent(x), self.to_cls(x_cls) * self.gamma

        return x, x_cls


class SpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            # Rearrange('b c n -> b n c'),
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32, 16, 8, 8] +
                                                                             [1] * (depth - 4),
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=[32] + [16]*(depth-2) + [1],
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, t(*c)]
                input_pos,  # [b, n, 2]
                ):
        x = torch.cat((x, input_pos), dim=-1)
        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


class SpatialEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,           # dropout of embedding
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                 'galerkin',
                                                 use_relu=False,
                                                 use_ln=use_ln,
                                                 scale=[res, res//4] + [1]*(depth-2),
                                                 relative_emb_dim=2,
                                                 min_freq=1 / res,
                                                 dropout=0.03,
                                                 attention_init='orthogonal')

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos)
        x = self.to_out(x)

        return x


class Encoder1D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 emb_dropout=0.05,           # dropout of embedding
                 res=2048,
                 ):
        super().__init__()

        self.dropout = nn.Dropout(emb_dropout)

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.transformer = TransformerCatNoCls(in_emb_dim, depth, 1, in_emb_dim, in_emb_dim, 'fourier',
                                               scale=[8.] + [4.]*2 + [1.]*(depth-3),
                                               relative_emb_dim=1,
                                               min_freq=1/res,
                                               use_ln=True,
                                               dropout=emb_dropout,
                                               attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 1]
                ):
        x = torch.cat((x, input_pos/16.), dim=-1)
        x = self.to_embedding(x)
        # x = self.dropout(x)
        # x = torch.cat((x, input_pos), dim=-1)
        x = self.transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


# for ablation
class NoRelSpatialTemporalEncoder2D(nn.Module):
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_seq_emb_dim,          # embedding dim of encoded sequence      (how about 256)
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            # Rearrange('b c n -> b n c'),
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=-1,
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', True, scale=-1,
                                                     attention_init='orthogonal')

        self.project_to_latent = nn.Sequential(
            nn.Linear(in_emb_dim, out_seq_emb_dim, bias=False))

    def forward(self,
                x,  # [b, t(*c)+2, n]
                input_pos,  # [b, n, 2]
                ):

        x = self.to_embedding(x)
        x = self.s_transformer.forward(x, input_pos)
        x = self.project_to_latent(x)

        return x


class SpatialOperator2D(nn.Module):
    # this directly takes in the input and output solution
    # input and output are supposed to be on the same grid
    def __init__(self,
                 input_channels,           # how many channels
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_chanels,
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,           # dropout of embedding
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.s_transformer = TransformerWithPad(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                 'galerkin',
                                                 use_relu=False,
                                                 use_ln=use_ln,
                                                 scale=[res, res//4] + [1]*(depth-2),
                                                 relative_emb_dim=2,
                                                 min_freq=1 / res,
                                                 dropout=0.,
                                                 attention_init='orthogonal')

        self.ln = nn.LayerNorm(in_emb_dim)

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim+1, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, out_chanels, bias=False))

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 2]
                param,  # [b,]
                pad_mask,    # [b, n, 1]
                ):
        param = param.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], 1)  # [b, n, 1]
        x = torch.cat((x, param, input_pos), dim=-1)
        x = self.to_embedding(x)
        x = x.masked_fill(~pad_mask, 0.)

        x_skip = x
        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos, pad_mask)

        x = x.masked_fill(~pad_mask, 0.)

        x = self.ln(x+x_skip)
        x = torch.cat((x, param), dim=-1)
        x = self.to_out(x)
        x = x.masked_fill(~pad_mask, 0.)


        return x

    def denormalize(self,
                    x,   # [b, n, c]
                    dataset,
                    bound_mask,   # [b, n, 4]  left right bottom top
                    ):

        # impose Dirichlet boundary condition

        # velocity
        x[:, :, :2] = x[:, :, :2] * dataset.statistics['vel_std'] + dataset.statistics['vel_mean']
        all_bound = torch.any(bound_mask, dim=-1, keepdim=True)
        x[:, :, :2] = x[:, :, :2].masked_fill(all_bound, 0.)

        # pressure
        x[:, :, 2] = x[:, :, 2] * dataset.statistics['prs_std'] + dataset.statistics['prs_mean']

        # temperature
        x[:, :, 3] = x[:, :, 3] * dataset.statistics['temp_std'] + dataset.statistics['temp_mean']
        x[:, :, 3] = x[:, :, 3].masked_fill(bound_mask[..., 0], 1.)
        x[:, :, 3] = x[:, :, 3].masked_fill(bound_mask[..., 1], 0.)

        return x


class SpatialTemporalOperator2D(nn.Module):
    # this directly takes in the input and output solution
    # input and output are supposed to be on the same grid
    # currently designed for solving 2D multi-phase flow
    # only 1 steps forward at a time
    def __init__(self,
                 input_channels,           # how many channels
                 time_window,             # how many time steps to look back
                 in_emb_dim,               # embedding dim of token                 (how about 512)
                 out_chanels,
                 heads,
                 depth,                    # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,           # dropout of embedding
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Conv2d(input_channels, in_emb_dim,
                      kernel_size=(time_window, 1),
                      stride=(time_window, 1), padding=(0, 0), bias=False),
            Rearrange('b c 1 n -> b n c')
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                 'galerkin',
                                                 use_relu=False,
                                                 use_ln=use_ln,
                                                 scale=[res, res//4] + [1]*(depth-2),
                                                 relative_emb_dim=2,
                                                 min_freq=1 / res,
                                                 dropout=0.,
                                                 attention_init='orthogonal')

        self.ln = nn.LayerNorm(in_emb_dim)

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim+1, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.GELU(),
            nn.Linear(in_emb_dim, out_chanels, bias=False))

    def forward(self,
                x,  # [b, c, t, n]
                input_pos,  # [b, n, 2]
                param,  # [b,]
                ):
        param = rearrange(param, 'b -> b 1 1 1').repeat([1, 1, x.shape[2], x.shape[3]])
        input_pos_ = rearrange(input_pos, 'b n c -> b c 1 n').repeat([1, 1, x.shape[2], 1])
        x = torch.cat((x, param, input_pos_), dim=1)
        x = self.to_embedding(x)

        x_skip = x
        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos)

        x = self.ln(x+x_skip)
        param = rearrange(param, 'b c t n -> b n t c')[:, :, 0, :]
        x = torch.cat((x, param), dim=-1)
        x = self.to_out(x)  # [b, n, c]

        return x

    def denormalize(self,
                    x,   # [b, n, c]
                    dataset,
                    ):
        # b n c -> b h w c
        # impose Dirichlet boundary condition
        # Warning: reshape is currently hardcoded
        x = x.clone()
        x = rearrange(x, 'b (h w) c -> b h w c', h=dataset.statistics['height'], w=dataset.statistics['width'])
        # velocity
        x[:, :, :, :2] = x[:, :, :, :2] * dataset.statistics['vel_std'] + dataset.statistics['vel_mean']
        x[:, -1, :, :2] = 0.
        x[:, :, -1, :2] = 0.
        x[:, 0, :, :2] = 0.
        x[:, :, 0, :2] = 0.

        # pressure
        x[..., 2] = x[..., 2] * dataset.statistics['prs_std'] + dataset.statistics['prs_mean']

        # vof
        # normalize to [0, 1]
        x[..., 3] = F.tanh(x[..., 3]) * 0.5 + 0.5

        return x

    def impose_boundary(self, x, dataset):
        # Warning: reshape is currently hardcoded
        x = rearrange(x, 'b (h w) c -> b h w c', h=dataset.statistics['height'], w=dataset.statistics['width'])
        # velocity
        x[:, :, :, :2] = x[:, :, :, :2] * dataset.statistics['vel_std']
        bound_val = (0. - dataset.statistics['vel_mean']) / dataset.statistics['vel_std']
        x[:, :, -1, :2] = bound_val
        x[:, -1, :, :2] = bound_val
        x[:, :, 0, :2] = bound_val
        x[:, 0, :, :2] = bound_val

        # vof
        x[..., 3] = F.tanh(x[..., 3]) * 0.5 + 0.5

        x = rearrange(x, 'b h w c -> b c (h w)')
        return x


# ============================
# for neurips rebuttal
# ============================


class IrregSpatialEncoder2D(torch.nn.Module):
    # for steady state irregular geometries
    def __init__(self,
                 input_channels,  # how many channels
                 in_emb_dim,  # embedding dim of token                 (how about 512)
                 out_chanels,
                 heads,
                 depth,  # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,  # dropout of embedding
                 ):
        super().__init__()

        self.to_embedding = nn.Sequential(
            nn.Linear(input_channels, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
        )

        self.dropout = nn.Dropout(emb_dropout)

        self.s_transformer = TransformerWithPad(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                'galerkin',
                                                use_relu=True,
                                                use_ln=use_ln,
                                                scale=[res, res // 4] + [1] * (depth - 2),
                                                relative_emb_dim=2,
                                                min_freq=1 / res,
                                                dropout=0.,
                                                attention_init='orthogonal')

        # self.ln = nn.LayerNorm(in_emb_dim)

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, out_chanels, bias=False)
        )

    def forward(self,
                x,  # [b, n, c]
                input_pos,  # [b, n, 2]
                pad_mask,  # [b, n, 1]
                ):
        # randomly drop some node

        x = self.to_embedding(x)
        x = x.masked_fill(~pad_mask, 0.)

        # x_skip = x
        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos, pad_mask)

        x = x.masked_fill(~pad_mask, 0.)

        # x = self.ln(x + x_skip)

        x = self.to_out(x)
        x = x.masked_fill(~pad_mask, 0.)

        return x


class IrregSTEncoder2D(torch.nn.Module):
    # for time dependent airfoil
    def __init__(self,
                 input_channels,  # how many channels
                 time_window,
                 in_emb_dim,  # embedding dim of token                 (how about 512)
                 out_chanels,
                 max_node_type,
                 heads,
                 depth,  # depth of transformer / how many layers of attention    (4)
                 res,
                 use_ln=True,
                 emb_dropout=0.05,  # dropout of embedding
                 ):
        super().__init__()
        self.tw = time_window
        # here, assume the input is in the shape [b, t, n, c]
        self.to_embedding = nn.Sequential(
            Rearrange('b t n c -> b c t n'),
            nn.Conv2d(input_channels, in_emb_dim, kernel_size=(self.tw, 1), stride=(self.tw, 1), padding=(0, 0), bias=False),
            nn.GELU(),
            nn.Conv2d(in_emb_dim, in_emb_dim, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False),
            Rearrange('b c 1 n -> b n c'),
        )

        self.node_embedding = nn.Embedding(max_node_type, in_emb_dim)

        self.combine_embedding = nn.Linear(in_emb_dim*2, in_emb_dim, bias=False)

        self.dropout = nn.Dropout(emb_dropout)

        if depth > 4:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', use_ln,
                                                     scale=[32, 16, 8, 8] + [1] * (depth - 4),
                                                     min_freq=1/res,
                                                     attention_init='orthogonal')
        else:
            self.s_transformer = TransformerCatNoCls(in_emb_dim, depth, heads, in_emb_dim, in_emb_dim,
                                                     'galerkin', use_ln,
                                                     scale=[32] + [16] * (depth - 2) + [1],
                                                     min_freq=1 / res,
                                                     attention_init='orthogonal')

        self.ln = nn.LayerNorm(in_emb_dim)

        self.to_out = nn.Sequential(
            nn.Linear(in_emb_dim, in_emb_dim, bias=False),
            nn.ReLU(),
            nn.Linear(in_emb_dim, out_chanels, bias=False),
        )

    def forward(self,
                x,  # [b, t, n, c]
                node_type,  # [b, n, 1]
                input_pos,  # [b, n, 2]
                ):
        x = self.to_embedding(x)
        x_node = self.node_embedding(node_type.squeeze(-1))
        x = self.combine_embedding(torch.cat([x, x_node], dim=-1))
        x_skip = x

        x = self.dropout(x)

        x = self.s_transformer.forward(x, input_pos)

        x = self.ln(x + x_skip)

        x = self.to_out(x)

        return x





