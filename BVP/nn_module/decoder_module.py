import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
import numpy as np
from .attention_module import PreNorm, PostNorm, LinearAttention, CrossLinearAttention,\
    FeedForward, GeGELU, ProjDotProduct, ReLUFeedForward
from .cnn_module import UpBlock, PeriodicConv2d
from torch.nn.init import xavier_uniform_, orthogonal_
from copy import deepcopy

def copy_weights(model1,
                 model2
                 ):
    # copy weights of model2 to model1
    # two models need to be exactly the same
    for param_1, param_2 in zip(model1.parameters(), model2.parameters()):
        param_1.data = deepcopy(param_2.data)


class AttentionPropagator2D(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 attn_type,         # ['none', 'galerkin', 'fourier']
                 mlp_dim,
                 scale,
                 use_ln=True,
                 dropout=0.):
        super().__init__()
        assert attn_type in ['none', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])

        self.attn_type = attn_type
        self.use_ln = use_ln
        for d in range(depth):
            attn_module = LinearAttention(dim, attn_type,
                                          heads=heads, dim_head=dim_head, dropout=dropout,
                                          relative_emb=True, scale=scale,
                                          relative_emb_dim=2,
                                          min_freq=1/64,
                                          init_method='orthogonal'
                                          )
            if use_ln:
                self.layers.append(
                    nn.ModuleList([
                        nn.LayerNorm(dim),
                        attn_module,
                        nn.LayerNorm(dim),
                        nn.Linear(dim+2, dim),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                    ]),
                )
            else:
                self.layers.append(
                    nn.ModuleList([
                        attn_module,
                        nn.Linear(dim + 2, dim),
                        FeedForward(dim, mlp_dim, dropout=dropout)
                    ]),
                )

    def forward(self, x, pos):
        for layer_no, attn_layer in enumerate(self.layers):
            if self.use_ln:
                [ln1, attn, ln2, proj, ffn] = attn_layer
                x = attn(ln1(x), pos) + x
                x = ffn(
                    proj(torch.cat((ln2(x), pos), dim=-1))
                        ) + x
            else:
                [attn, proj, ffn] = attn_layer
                x = attn(x, pos) + x
                x = ffn(
                    proj(torch.cat((x, pos), dim=-1))
                        ) + x
        return x


class AttentionPropagator1D(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 heads,
                 dim_head,
                 attn_type,         # ['none', 'galerkin', 'fourier']
                 mlp_dim,
                 scale,
                 res,
                 dropout=0.):
        super().__init__()
        assert attn_type in ['none', 'galerkin', 'fourier']
        self.layers = nn.ModuleList([])

        self.attn_type = attn_type

        for d in range(depth):
            attn_module = LinearAttention(dim, attn_type,
                                          heads=heads, dim_head=dim_head, dropout=dropout,
                                          relative_emb=True,
                                          scale=scale,
                                          relative_emb_dim=1,
                                          min_freq=1 / res,
                                          )
            self.layers.append(
                nn.ModuleList([
                    attn_module,
                    FeedForward(dim, mlp_dim, dropout=dropout)
                ]),
            )

    def forward(self, x, pos):
        for layer_no, attn_layer in enumerate(self.layers):
            [attn, ffn] = attn_layer

            x = attn(x, pos) + x
            x = ffn(x) + x
        return x


class FourierPropagator(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 mode):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.latent_channels = dim

        for d in range(depth):
            self.layers.append(nn.Sequential(FourierConv2d(self.latent_channels, self.latent_channels,
                                                           mode, mode), nn.GELU()))

    def forward(self, z):
        for layer, f_conv in enumerate(self.layers):
            z = f_conv(z) + z
        return z


class MLPPropagator(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.latent_channels = dim

        for d in range(depth):
            layer = nn.Sequential(
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.GELU(),
                nn.Conv2d(dim, dim, 1, 1, 0, bias=False),
                nn.InstanceNorm2d(dim),
            )
            self.layers.append(layer)

    def forward(self, z):
        for layer, ffn in enumerate(self.layers):
            z = ffn(z) + z
        return z


class PointWiseMLPPropagator(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.latent_channels = dim

        for d in range(depth):
            if d == 0:
                layer = nn.Sequential(
                    nn.InstanceNorm1d(dim + 2),
                    nn.Linear(dim + 2, dim, bias=False),  # for position
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                )
            else:
                layer = nn.Sequential(
                    nn.InstanceNorm1d(dim),
                    nn.Linear(dim, dim, bias=False),
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                    nn.GELU(),
                    nn.Linear(dim, dim, bias=False),
                )
            self.layers.append(layer)

    def forward(self, z, pos):
        for layer, ffn in enumerate(self.layers):
            if layer == 0:
                z = ffn(torch.cat((z, pos), dim=-1)) + z
            else:
                z = ffn(z) + z
        return z


# code copied from: https://github.com/ndahlquist/pytorch-fourier-feature-networks
# author: Nic Dahlquist
class GaussianFourierFeatureTransform(torch.nn.Module):
    """
    An implementation of Gaussian Fourier feature mapping.
    "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains":
       https://arxiv.org/abs/2006.10739
       https://people.eecs.berkeley.edu/~bmild/fourfeat/index.html
    Given an input of size [batches, n, num_input_channels],
     returns a tensor of size [batches, n, mapping_size*2].
    """

    def __init__(self, num_input_channels, mapping_size=256, scale=10):
        super().__init__()

        self._num_input_channels = num_input_channels
        self._mapping_size = mapping_size
        self._B = nn.Parameter(torch.randn((num_input_channels, mapping_size)) * scale, requires_grad=False)

    def forward(self, x):

        batches, num_of_points, channels = x.shape

        # Make shape compatible for matmul with _B.
        # From [B, N, C] to [(B*N), C].
        x = rearrange(x, 'b n c -> (b n) c')

        x = x @ self._B.to(x.device)

        # From [(B*W*H), C] to [B, W, H, C]
        x = rearrange(x, '(b n) c -> b n c', b=batches)

        x = 2 * np.pi * x
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)


class CrossFormer(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                       heads=heads, dim_head=dim_head, dropout=dropout,
                                                       relative_emb=relative_emb,
                                                       scale=scale,

                                                       relative_emb_dim=relative_emb_dim,
                                                       min_freq=min_freq,
                                                       init_method='orthogonal',
                                                       cat_pos=cat_pos,
                                                       pos_dim=relative_emb_dim,
                                                  )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos)) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x


class CrossFormerWithPad(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,
                 heads,
                 dim_head,
                 mlp_dim,
                 residual=True,
                 use_ffn=True,
                 use_ln=False,
                 relative_emb=False,
                 scale=1.,
                 relative_emb_dim=2,
                 min_freq=1/64,
                 dropout=0.,
                 cat_pos=False,
                 ):
        super().__init__()

        self.cross_attn_module = CrossLinearAttention(dim, attn_type,
                                                       heads=heads, dim_head=dim_head, dropout=dropout,
                                                       relative_emb=relative_emb,
                                                       scale=scale,
                                                       relative_emb_dim=relative_emb_dim,
                                                       min_freq=min_freq,
                                                       init_method='orthogonal',
                                                       cat_pos=cat_pos,
                                                       pos_dim=relative_emb_dim,
                                                       use_ln=True,
                                                  )
        self.use_ln = use_ln
        self.residual = residual
        self.use_ffn = use_ffn

        if self.use_ln:
            self.ln1 = nn.LayerNorm(dim)
            self.ln2 = nn.LayerNorm(dim)

        if self.use_ffn:
            self.ffn = FeedForward(dim, mlp_dim, dropout)

    def forward(self, x, z, pad_mask, x_pos=None, z_pos=None):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        if self.use_ln:
            z = self.ln1(z)
            if self.residual:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos, padding_mask=pad_mask)) + x
            else:
                x = self.ln2(self.cross_attn_module(x, z, x_pos, z_pos, padding_mask=pad_mask))
        else:
            if self.residual:
                x = self.cross_attn_module(x, z, x_pos, z_pos, padding_mask=pad_mask) + x
            else:
                x = self.cross_attn_module(x, z, x_pos, z_pos, padding_mask=pad_mask)

        if self.use_ffn:
            x = self.ffn(x) + x

        return x


class BranchTrunkNet(nn.Module):
    def __init__(self,
                 dim,
                 branch_size,
                 branchnet_dim,
                 ):
        super().__init__()
        self.proj = nn.Sequential(
            Rearrange('b n c -> b c n'),
            nn.Linear(branch_size, branchnet_dim),
            nn.ReLU(),
            nn.Linear(branchnet_dim//2, branchnet_dim//2),
            nn.ReLU(),
            nn.Linear(branchnet_dim//2, 1),

        )
        self.net = ProjDotProduct(dim, dim, dim)

    def forward(self, x, z):
        # x in [b n1 c]
        # b, n1, c = x.shape   # coordinate encoding
        # b, n2, c = z.shape   # system encoding
        z = self.proj(z).squeeze(-1)
        return self.net(x, z)


class PointWiseDecoder2D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 propagator_depth,
                 scale=8,
                 dropout=0.,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels//2, 'galerkin', 4,
                                                self.latent_channels//2, self.latent_channels//2,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.expand_feat = nn.Linear(self.latent_channels//2, self.latent_channels)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.latent_channels),
               nn.Sequential(
                    nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
            for _ in range(propagator_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels * self.out_steps, bias=True))

    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos   # [b, n, 2]
                ):
        z = self.propagate(z, propagate_pos)
        u = self.decode(z)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z                # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps//self.out_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=-2)  # concatenate in temporal dimension
        return history  # [b, length_of_history*c, n]


class PointWiseDecoder1D(nn.Module):
    # for Burgers equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 decoding_depth,  # 4?
                 scale=8,
                 res=2048,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(1, self.latent_channels, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'fourier', 8,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=1,
                                                relative_emb_dim=1,
                                                min_freq=1/res)

        self.propagator = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                nn.GELU(),
                nn.Linear(self.latent_channels, self.latent_channels, bias=False),)
            for _ in range(decoding_depth)])

        self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def propagate(self, z):
        for num_l, layer in enumerate(self.propagator):
            z = z + layer(z)
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def init_propagator_params(self):
        for block in self.propagator:
            for layers in block:
                    for param in layers.parameters():
                        if param.ndim > 1:
                            in_c = param.size(-1)
                            orthogonal_(param[:in_c], gain=1/in_c)
                            param.data[:in_c] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))
                            if param.size(-2) != param.size(-1):
                                orthogonal_(param[in_c:], gain=1/in_c)
                                param.data[in_c:] += 1/in_c * torch.diag(torch.ones(param.size(-1), dtype=torch.float32))

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        z = self.propagate(z)
        z = self.decode(z)
        return z  # [b, n, c]


class PointWiseDecoder2DSimple(nn.Module):
    # for Darcy equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=211,
                 scale=0.5,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            # nn.Linear(2, self.latent_channels, bias=False),
            # nn.GELU(),
            # nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            # nn.GELU(),
            # nn.Linear(self.latent_channels * 2, self.latent_channels, bias=False),
            nn.Dropout(0.05),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                relative_emb=True,
                                                scale=16,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        # self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels+2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        z = self.decode(torch.cat((z, propagate_pos), dim=-1))
        return z  # [b, n, c]


class STPointWiseDecoder2D(nn.Module):
    # no time marching
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,
                 scale=8,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(3, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 1,
                                                self.latent_channels, self.latent_channels,
                                                residual=False,
                                                use_ffn=False,
                                                relative_emb=True,
                                                scale=1.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos,  # [b, tn, 3]
                input_pos,      # [b, n, 2]
                ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos[:, :, :-1], input_pos)
        z = self.decode(z)
        z = rearrange(z, 'b (t n) c -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return z


class BCDecoder1D(nn.Module):
    # for Burgers equation, using DeepONet formulation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 decoding_depth,  # 4?
                 scale=8,
                 res=2048,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(1, self.latent_channels, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = BranchTrunkNet(latent_channels,
                                                   res)

    def forward(self,
                z,  # [b, n, c]
                propagate_pos,  # [b, n, 1]
                ):
        propagate_pos = propagate_pos[0]
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z)

        return z  # [b, n, c]


class PieceWiseDecoder2DSimple(nn.Module):
    # for Darcy flow inverse problem
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=141,
                 scale=0.5,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            # nn.Linear(2, self.latent_channels, bias=False),
            # nn.GELU(),
            # nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            # nn.GELU(),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.Dropout(0.05),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                use_ffn=False,
                                                relative_emb=True,
                                                scale=16,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        # self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels+2, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        z = self.decode(torch.cat((z, propagate_pos), dim=-1))
        return z  # [b, n, c]


class NoRelPointWiseDecoder2D(nn.Module):
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 out_steps,  # 10
                 propagator_depth,
                 scale=8,
                 dropout=0.,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.out_steps = out_steps
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels//2, 'galerkin', 4,
                                                self.latent_channels//2, self.latent_channels//2,
                                                relative_emb=False,
                                                cat_pos=True,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.expand_feat = nn.Linear(self.latent_channels//2, self.latent_channels)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.latent_channels),
               nn.Sequential(
                    nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
            for _ in range(propagator_depth)])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels * self.out_steps, bias=True))

    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos   # [b, n, 2]
                ):
        z = self.propagate(z, propagate_pos)
        u = self.decode(z)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z                # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.expand_feat(z)

        # forward the dynamics in the latent space
        for step in range(forward_steps//self.out_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z)
            history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps))
        history = torch.cat(history, dim=-2)  # concatenate in temporal dimension
        return history  # [b, length_of_history*c, n]


class PointWiseDecoder2DTemporal(nn.Module):
    # for 2d convection diffusion equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 propagator_depth,
                 res=64,
                 scale=2,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            # nn.Linear(2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            # nn.GELU(),
            # nn.Linear(self.latent_channels * 2, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                relative_emb=True,
                                                scale=16,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        self.propagator = nn.ModuleList([
                           nn.Sequential(
                               nn.LayerNorm(self.latent_channels),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False))
            for _ in range(propagator_depth)])
        # self.init_propagator_params()
        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def init_propagator_params(self):
        for block in self.propagator:
            for layers in block:
                    for param in layers.parameters():
                        if param.ndim > 1:
                            in_c = param.size(-2)
                            orthogonal_(param, gain=1/in_c)
                            param.data[:, :in_c] += 1/in_c * torch.diag(torch.ones(in_c, dtype=torch.float32))


    def propagate(self, z, pos):
        for num_l, layer in enumerate(self.propagator):
            ffn = layer
           #  z = ffn(torch.cat((z, pos), dim=-1)) + z
            z = ffn(z) + z
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos=None,
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.propagate(z, propagate_pos)
        z = self.decode(z)
        return z  # [b, n, c]


class STDecoder1D(nn.Module):
    # for time-marching Burgers equation
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 scale=8,
                 res=2048,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(1, self.latent_channels, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'fourier', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=1,
                                                relative_emb_dim=1,
                                                min_freq=1/res)

        self.propagator = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(self.latent_channels),
                           nn.Sequential(
                               nn.Linear(self.latent_channels + 1, self.latent_channels, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
                                        ])

        self.init_propagator_params()
        self.to_out = nn.ModuleList([
            LinearAttention(self.latent_channels, 'fourier',
                            heads=1, dim_head=self.latent_channels,
                            pos_dim=1,
                            relative_emb=True,
                            relative_emb_dim=1,
                            min_freq=1/res,
                            ),
            nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))]
        )

    def propagate(self, z, pos):
        for num_l, layer in enumerate(self.propagator):
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos/16.), dim=-1)) + z
        return z

    def decode(self, z, pos):
        attn, ffn = self.to_out[0], self.to_out[1]
        z = attn(z, pos) + z
        z = ffn(z)
        return z

    def init_propagator_params(self):
        for layers in self.propagator[0][1]:
                for param in layers.parameters():
                    if param.ndim > 1:
                        in_c = param.size(-2)
                        orthogonal_(param, gain=1 / in_c)
                        param.data[:, :in_c] += 1 / in_c * torch.diag(torch.ones(in_c, dtype=torch.float32))

    def get_embedding(self,
                      z,  # [b, n c]
                      propagate_pos,  # [b, n, 2]
                      input_pos
                      ):
        x = self.coordinate_projection.forward(propagate_pos/16.)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        return z

    def forward(self,
                z,  # [b, n, c]
                propagate_pos  # [b, n, 2]
                ):
        x = self.coordinate_projection.forward(propagate_pos/16.)
        z = self.decoding_transformer.forward(x, z, propagate_pos)

        z = self.propagate(z, propagate_pos)
        u = self.decode(z, propagate_pos)
        u = rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=self.out_steps)
        return u, z  # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos/16.)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z_init = z

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z, propagate_pos)
            history.append(rearrange(u, 'b n (t c) -> b (t c) n', c=self.out_channels, t=1))
        history = torch.cat(history, dim=-2)  # concatenate in temporal dimension
        return z_init, history  # [b, length_of_history*c, n]


class STDecoder2D(nn.Module):
    # For Markovian Navier-Stokes / Reaction-diffusion
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 scale=8,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        self.propagator = nn.ModuleList([
               nn.ModuleList([nn.LayerNorm(self.latent_channels),
               nn.Sequential(
                    nn.Linear(self.latent_channels + 2, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False),
                    nn.GELU(),
                    nn.Linear(self.latent_channels, self.latent_channels, bias=False))])
            ])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def load_pretrained_weights(self, pretrained_decoder):
        copy_weights(self.coordinate_projection, pretrained_decoder.coordinate_projection)
        copy_weights(self.decoding_transformer, pretrained_decoder.decoding_transformer)


    def propagate(self, z, pos):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), pos), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,              # [b, n, c]
                propagate_pos   # [b, n, 2]
                ):
        z = self.propagate(z, propagate_pos)
        u = self.decode(z)
        if self.out_channels == 1:
            u = rearrange(u, 'b n t-> b t n')
        else:
            u = rearrange(u, 'b n (t c) -> b t c n', c=self.out_channels)
        return u, z                # [b c_out t n], [b c_latent t n]

    def rollout(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                forward_steps,
                input_pos):
        history = []
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagate(z, propagate_pos)
            u = self.decode(z)

            if self.out_channels == 1:
                u = rearrange(u, 'b n t-> b t n')
            else:
                u = rearrange(u, 'b n (t c) -> b t n c', c=self.out_channels)

            history.append(u)
        history = torch.cat(history, dim=1)  # concatenate in temporal dimension
        return history


class ReconSTDecoder2D(nn.Module):
    # For Navier-Stokes MAE
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 scale=8,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=16.,
                                                relative_emb_dim=2,
                                                min_freq=1/64)

        # simple reconstruction head
        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels // 2, self.latent_channels // 2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def forward(self,
                z,              # [b, n, c]
                propagate_pos,   # [b, n, 2]
                input_pos,
                ):
        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.to_out(z)
        return z                # [b n c_out]


class SpatialDecoder2D(nn.Module):
    # for Darcy equation or Heat-cavity
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=50,
                 scale=0.1,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels//2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
        )

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                use_ln=False,
                                                residual=True,
                                                relative_emb=True,
                                                scale=8,
                                                relative_emb_dim=2,
                                                min_freq=1/res)

        self.ln = nn.LayerNorm(self.latent_channels)

        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels+1, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels//2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels//2, self.out_channels, bias=True))

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 1]
                input_pos,
                param   # [b,]
                ):

        x = self.coordinate_projection.forward(propagate_pos)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.ln(z)
        param = param.view(-1, 1, 1).repeat([1, z.shape[0], 1])
        z = torch.cat((z, param), dim=-1)
        z = self.decode(z)
        return z  # [b, n, c]


# =============================================================================
# for neurips rebuttal
# =============================================================================

class IrregSpatialDecoder2D(nn.Module):
    # for steady state irregular geometry
    def __init__(self,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=10,
                 dropout=0.1,
                 scale=0.5,
                 **kwargs,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.coordinate_projection = nn.Sequential(
            # GaussianFourierFeatureTransform(3, self.latent_channels, scale=scale),
            nn.Linear(3, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.dropout = nn.Dropout(dropout)

        self.decoding_transformer = CrossFormerWithPad(self.latent_channels, 'galerkin', 4,
                                                        self.latent_channels, self.latent_channels,
                                                        use_ln=False,
                                                        residual=True,
                                                        relative_emb=True,
                                                        scale=1,
                                                        relative_emb_dim=2,
                                                        min_freq=1 / res)

        self.mix_layer = LinearAttention(self.latent_channels, 'galerkin',
                                           heads=1, dim_head=self.latent_channels,
                                           relative_emb=True, scale=4,
                                           relative_emb_dim=2,
                                           min_freq=1 / res,
                                           use_ln=True,
                                           )

        # self.ln = nn.LayerNorm(self.latent_channels)

        self.to_out = nn.Sequential(
            nn.Linear(self.latent_channels + 1, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False))

        self.scalar_head = nn.Sequential(
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, 1, bias=True))

        self.field_head = nn.Sequential(
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
            nn.ReLU(),
            nn.Linear(self.latent_channels, 2, bias=True))

        self.init_decoder_params()


    def init_decoder_params(self):
        for layers in self.to_out:
                for param in layers.parameters():
                    if param.ndim > 1:
                        in_c = param.size(-2)
                        xavier_uniform_(param, gain=1e-1)
                        # param.data[:, :in_c] += 1 / in_c * torch.diag(torch.ones(in_c, dtype=torch.float32))

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                input_pos,      # [b, n, 2]
                pad_mask,   # [b, n, 1]
                bound_mask, # [b, n, 1]
                ):
        x = self.coordinate_projection.forward(torch.cat((propagate_pos, bound_mask.float()), dim=-1))
        x = x.masked_fill(~pad_mask, 0.)
        z = self.dropout(z)
        z = self.decoding_transformer.forward(x, z, pad_mask, propagate_pos, input_pos.detach())
        z = z.masked_fill(~pad_mask, 0.)

        z = self.mix_layer.forward(z, propagate_pos, padding_mask=pad_mask) + z

        # z = self.ln(z)
        z = torch.cat((z, bound_mask.float()), dim=-1)
        z = self.to_out(z)
        z = z.masked_fill(~pad_mask, 0.)

        scalar_pred = self.scalar_head(z)
        field_pred = self.field_head(z)
        z = torch.cat((scalar_pred, field_pred), dim=-1)
        z = z.masked_fill(~pad_mask, 0.)

        return z  # [b, n, c]

    def impose_boundary(self,
                    x,  # [b, n, c]
                    train_set,
                    bound_mask,  # [b, n, 1]  left right bottom top
                    ):
        # denormalize
        # hard coded a scale factor for now
        x = x * 0.1
        # x[:, :, 0] = x[:, :, 0] * train_set.statistics['pot_std'] + train_set.statistics['pot_mean']
        # x[:, :, 1:] = x[:, :, 1:] * train_set.statistics['field_std'] + train_set.statistics['field_mean']
        # impose Dirichlet boundary condition

        # potential
        # x[:, :, 0:1] = x[:, :, 0:1].masked_fill(bound_mask, 0.)
        return x


class IrregSTDecoder2D(nn.Module):
    def __init__(self,
                 max_node_type,
                 latent_channels,  # 256??
                 out_channels,  # 1 or 2?
                 res=200,
                 scale=8,
                 **kwargs,
                 ):
        super().__init__()
        self.out_channels = out_channels
        self.latent_channels = latent_channels

        self.node_type_embedding = nn.Embedding(max_node_type, latent_channels)

        self.coordinate_projection = nn.Sequential(
            GaussianFourierFeatureTransform(2, self.latent_channels // 2, scale=scale),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.latent_channels, bias=False),
        )

        self.combine_layer = nn.Linear(self.latent_channels, self.latent_channels, bias=False)

        self.decoding_transformer = CrossFormer(self.latent_channels, 'galerkin', 4,
                                                self.latent_channels, self.latent_channels,
                                                relative_emb=True,
                                                scale=1.,
                                                relative_emb_dim=2,
                                                min_freq=1 / 200)

        self.mix_layer = LinearAttention(self.latent_channels, 'galerkin',
                                         heads=1, dim_head=self.latent_channels,
                                         relative_emb=True, scale=32,
                                         relative_emb_dim=2,
                                         min_freq=1 / res,
                                         use_ln=False,
                                         )

        self.propagator = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(self.latent_channels*2),
                           nn.Sequential(
                               nn.Linear(self.latent_channels*3, self.latent_channels*2, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels*2, self.latent_channels*2, bias=False),
                               nn.GELU(),
                               nn.Linear(self.latent_channels*2, self.latent_channels*2, bias=False))])
        ])

        self.to_out = nn.Sequential(
            nn.LayerNorm(self.latent_channels*2),
            nn.Linear(self.latent_channels*2, self.latent_channels*2, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels*2, self.latent_channels, bias=False),
            nn.GELU(),
            nn.Linear(self.latent_channels, self.out_channels, bias=True))

    def propagate(self, z, z_node):
        for layer in self.propagator:
            norm_fn, ffn = layer
            z = ffn(torch.cat((norm_fn(z), z_node), dim=-1)) + z
        return z

    def decode(self, z):
        z = self.to_out(z)
        return z

    def forward(self,
                z,  # [b, n c]
                propagate_pos,  # [b, n, 2]
                prop_node_type,  # [b, n, 1]
                forward_steps,
                input_pos):
        history = []
        x_node = self.node_type_embedding(prop_node_type)
        x = self.coordinate_projection.forward(propagate_pos)
        x = self.combine_layer(x+x_node)
        z = self.decoding_transformer.forward(x, z, propagate_pos, input_pos)
        z = self.mix_layer.forward(z, propagate_pos) + z

        # forward the dynamics in the latent space
        for step in range(forward_steps):
            z = self.propagate(z, x_node)
            u = self.decode(z)

            if self.out_channels == 1:
                u = rearrange(u, 'b n t-> b t n')
            else:
                u = rearrange(u, 'b n (t c) -> b t n c', c=self.out_channels)

            history.append(u)
        history = torch.cat(history, dim=1)  # concatenate in temporal dimension
        return history

    def denormalize(self,
                    x,  # [b, t, n, c]
                    train_set,
                    ):
        # denormalize
        # vel
        x[:, :, :, 0] = x[:, :, :, 0] * train_set.statistics['vel_x_std'] + train_set.statistics['vel_x_mean']
        x[:, :, :, 1] = x[:, :, :, 1] * train_set.statistics['vel_y_std'] + train_set.statistics['vel_y_mean']

        # dns
        x[:, :, :, 2] = x[:, :, :, 2] * train_set.statistics['dns_std'] + train_set.statistics['dns_mean']

        # prs
        x[:, :, :, 3] = x[:, :, :, 3] * train_set.statistics['prs_std'] + train_set.statistics['prs_mean']

        return x



