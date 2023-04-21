import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange
from torch.nn.init import xavier_uniform_, constant_, xavier_normal_, orthogonal_
# from torch_cluster import fps
# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class PostNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.norm(self.fn(x, **kwargs))


class GeGELU(nn.Module):
    """https: // paperswithcode.com / method / geglu"""
    def __init__(self):
        super().__init__()
        self.fn = nn.GELU()

    def forward(self, x):
        c = x.shape[-1]  # channel last arrangement
        return self.fn(x[..., :int(c//2)]) * x[..., int(c//2):]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim*2),
            GeGELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class ReLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


def masked_instance_norm(x, mask, eps = 1e-5):
    """
    x of shape: [batch_size (N), num_objects (L), features(C)]
    mask of shape: [batch_size (N), num_objects (L), 1]
    """
    mask = mask.float()  # (N,L,1)
    mean = (torch.sum(x * mask, 1) / torch.sum(mask, 1))   # (N,C)
    mean = mean.detach()
    var_term = ((x - mean.unsqueeze(1).expand_as(x)) * mask)**2  # (N,L,C)
    var = (torch.sum(var_term, 1) / torch.sum(mask, 1))  #(N,C)
    var = var.detach()
    mean_reshaped = mean.unsqueeze(1).expand_as(x)  # (N, L, C)
    var_reshaped = var.unsqueeze(1).expand_as(x)    # (N, L, C)
    ins_norm = (x - mean_reshaped) / torch.sqrt(var_reshaped + eps)   # (N, L, C)
    return ins_norm

# New position encoding module
# modified from https://github.com/lucidrains/x-transformers/blob/main/x_transformers/x_transformers.py
class RotaryEmbedding(nn.Module):
    def __init__(self, dim, min_freq=1/64, scale=1.):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.min_freq = min_freq
        self.scale = scale
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, coordinates, device):
        # coordinates [b, n]
        t = coordinates.to(device).type_as(self.inv_freq)
        t = t * (self.scale / self.min_freq)
        freqs = torch.einsum('... i , j -> ... i j', t, self.inv_freq)  # [b, n, d//2]
        return torch.cat((freqs, freqs), dim=-1)  # [b, n, d]


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j = 2)
    x1, x2 = x.unbind(dim = -2)
    return torch.cat((-x2, x1), dim = -1)


def apply_rotary_pos_emb(t, freqs):
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


def apply_2d_rotary_pos_emb(t, freqs_x, freqs_y):
    # split t into first half and second half
    # t: [b, h, n, d]
    # freq_x/y: [b, n, d]
    d = t.shape[-1]
    t_x, t_y = t[..., :d//2], t[..., d//2:]

    return torch.cat((apply_rotary_pos_emb(t_x, freqs_x),
                      apply_rotary_pos_emb(t_y, freqs_y)), dim=-1)


# https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/torch_encodings.py
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc


class StandardAttention(nn.Module):
    """Standard scaled dot product attention"""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., causal=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.causal = causal  # simple autogressive attention with upper triangular part being masked zero

    def forward(self, x, mask=None):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        if mask is not None:
            if not self.causal:
                raise Exception('Passing in mask while attention is not causal')
            mask_value = -torch.finfo(dots.dtype).max
            dots = dots.masked_fill(mask, mask_value)

        attn = self.attend(dots)     # similarity score

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class LinearAttention(nn.Module):
    """
    Contains following two types of attention, as discussed in "Choose a Transformer: Fourier or Galerkin"

    Galerkin type attention, with instance normalization on Key and Value
    Fourier type attention, with instance normalization on Query and Key
    """
    def __init__(self,
                 dim,
                 attn_type,                 # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',    # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1/64,             # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 use_ln=False
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type
        self.use_ln = use_ln

        self.heads = heads
        self.dim_head = dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = nn.InstanceNorm1d(dim_head)
                self.v_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.k_norm = nn.LayerNorm(dim_head)
                self.v_norm = nn.LayerNorm(dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = nn.InstanceNorm1d(dim_head)
                self.k_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.q_norm = nn.LayerNorm(dim_head)
                self.k_norm = nn.LayerNorm(dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim*heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            assert not cat_pos
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_qkv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    if self.attn_type == 'fourier':
                        # for v
                        init_fn(param[(self.heads * 2 + h) * self.dim_head:(self.heads * 2 + h + 1) * self.dim_head, :],
                                gain=self.init_gain)
                        param.data[(self.heads * 2 + h) * self.dim_head:(self.heads * 2 + h + 1) * self.dim_head,
                        :] += self.diagonal_weight * \
                              torch.diag(torch.ones(
                                  param.size(-1),
                                  dtype=torch.float32))
                    else: # for galerkin
                        # for q
                        init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                        #
                        param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                    torch.diag(torch.ones(
                                                                                        param.size(-1),
                                                                                        dtype=torch.float32))


    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, pos=None, not_assoc=False, padding_mask=None):
        # padding mask will be in shape [b, n, 1], it will indicates which point are padded and should be ignored
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        if pos is None and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')

        if padding_mask is None:
            if self.attn_type == 'galerkin':
                k = self.norm_wrt_domain(k, self.k_norm)
                v = self.norm_wrt_domain(v, self.v_norm)
            else:  # fourier
                q = self.norm_wrt_domain(q, self.q_norm)
                k = self.norm_wrt_domain(k, self.k_norm)
        else:
            grid_size = torch.sum(padding_mask, dim=[-1, -2]).view(-1, 1, 1, 1)  # [b, 1, 1]
            padding_mask = repeat(padding_mask, 'b n d -> (b h) n d', h=self.heads)  # [b, n, 1]

            if self.use_ln:
                if self.attn_type == 'galerkin':
                    k = self.k_norm(k)
                    v = self.v_norm(v)
                else:  # fourier
                    q = self.q_norm(q)
                    k = self.k_norm(k)
            else:

                if self.attn_type == 'galerkin':
                    k = rearrange(k, 'b h n d -> (b h) n d')
                    v = rearrange(v, 'b h n d -> (b h) n d')

                    k = masked_instance_norm(k, padding_mask)
                    v = masked_instance_norm(v, padding_mask)

                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
                    v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)
                else:  # fourier
                    q = rearrange(q, 'b h n d -> (b h) n d')
                    k = rearrange(k, 'b h n d -> (b h) n d')

                    q = masked_instance_norm(q, padding_mask)
                    k = masked_instance_norm(k, padding_mask)

                    q = rearrange(q, '(b h) n d -> b h n d', h=self.heads)
                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)

            padding_mask = rearrange(padding_mask, '(b h) n d -> b h n d', h=self.heads)  # [b, h, n, 1]


        if self.relative_emb:
            if self.relative_emb_dim == 2:
                freqs_x = self.emb_module.forward(pos[..., 0], x.device)
                freqs_y = self.emb_module.forward(pos[..., 1], x.device)
                freqs_x = repeat(freqs_x, 'b n d -> b h n d', h=q.shape[1])
                freqs_y = repeat(freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
                k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)
            elif self.relative_emb_dim == 1:
                assert pos.shape[-1] == 1
                freqs = self.emb_module.forward(pos[..., 0], x.device)
                freqs = repeat(freqs, 'b n d -> b h n d', h=q.shape[1])
                q = apply_rotary_pos_emb(q, freqs)
                k = apply_rotary_pos_emb(k, freqs)
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')

        elif self.cat_pos:
            assert pos.size(-1) == self.pos_dim
            pos = pos.unsqueeze(1)
            pos = pos.repeat([1, self.heads, 1, 1])
            q, k, v = [torch.cat([pos, x], dim=-1) for x in (q, k, v)]

        if not_assoc:
            # this is more efficient when n<<c
            score = torch.matmul(q, k.transpose(-1, -2))
            if padding_mask is not None:
                padding_mask = ~padding_mask
                padding_mask_arr = torch.matmul(padding_mask, padding_mask.transpose(-1, -2))  # [b, h, n, n]
                mask_value = 0.
                score = score.masked_fill(padding_mask_arr, mask_value)
                out = torch.matmul(score, v) * (1./grid_size)
            else:
                out = torch.matmul(score, v) * (1./q.shape[2])
        else:
            if padding_mask is not None:
                q = q.masked_fill(~padding_mask, 0)
                k = k.masked_fill(~padding_mask, 0)
                v = v.masked_fill(~padding_mask, 0)
                dots = torch.matmul(k.transpose(-1, -2), v)
                out = torch.matmul(q, dots) * (1. / grid_size)
            else:
                dots = torch.matmul(k.transpose(-1, -2), v)
                out = torch.matmul(q, dots) * (1./q.shape[2])
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class CrossLinearAttention(nn.Module):
    def __init__(self,
                 dim,
                 attn_type,  # ['fourier', 'galerkin']
                 heads=8,
                 dim_head=64,
                 dropout=0.,
                 init_params=True,
                 relative_emb=False,
                 scale=1.,
                 init_method='orthogonal',  # ['xavier', 'orthogonal']
                 init_gain=None,
                 relative_emb_dim=2,
                 min_freq=1 / 64,  # 1/64 is for 64 x 64 ns2d,
                 cat_pos=False,
                 pos_dim=2,
                 use_ln=False,
                 ):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.attn_type = attn_type
        self.use_ln = use_ln

        self.heads = heads
        self.dim_head = dim_head

        # query is the classification token
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)

        if attn_type == 'galerkin':
            if not self.use_ln:
                self.k_norm = nn.InstanceNorm1d(dim_head)
                self.v_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.k_norm = nn.LayerNorm(dim_head)
                self.v_norm = nn.LayerNorm(dim_head)

        elif attn_type == 'fourier':
            if not self.use_ln:
                self.q_norm = nn.InstanceNorm1d(dim_head)
                self.k_norm = nn.InstanceNorm1d(dim_head)
            else:
                self.q_norm = nn.LayerNorm(dim_head)
                self.k_norm = nn.LayerNorm(dim_head)

        else:
            raise Exception(f'Unknown attention type {attn_type}')

        if not cat_pos:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
            ) if project_out else nn.Identity()
        else:
            self.to_out = nn.Sequential(
                nn.Linear(inner_dim + pos_dim*heads, dim),
                nn.Dropout(dropout)
            )

        if init_gain is None:
            self.init_gain = 1. / dim_head
            self.diagonal_weight = 1. / dim_head
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain
        self.init_method = init_method
        if init_params:
            self._init_params()

        self.cat_pos = cat_pos
        self.pos_dim = pos_dim

        self.relative_emb = relative_emb
        self.relative_emb_dim = relative_emb_dim
        if relative_emb:
            self.emb_module = RotaryEmbedding(dim_head // self.relative_emb_dim, min_freq=min_freq, scale=scale)

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.to_kv.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for k
                    init_fn(param[h*self.dim_head:(h+1)*self.dim_head, :], gain=self.init_gain)
                    param.data[h*self.dim_head:(h+1)*self.dim_head, :] += self.diagonal_weight * \
                                                                          torch.diag(torch.ones(
                                                                              param.size(-1), dtype=torch.float32))

                    # for v
                    init_fn(param[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[(self.heads + h) * self.dim_head:(self.heads + h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                           torch.diag(torch.ones(
                                                                               param.size(-1), dtype=torch.float32))
                                                                               
        for param in self.to_q.parameters():
            if param.ndim > 1:
                for h in range(self.heads):
                    # for q
                    init_fn(param[h * self.dim_head:(h + 1) * self.dim_head, :], gain=self.init_gain)
                    param.data[h * self.dim_head:(h + 1) * self.dim_head, :] += self.diagonal_weight * \
                                                                                torch.diag(torch.ones(
                                                                                    param.size(-1), dtype=torch.float32))

    def norm_wrt_domain(self, x, norm_fn):
        b = x.shape[0]
        return rearrange(
            norm_fn(rearrange(x, 'b h n d -> (b h) n d')),
            '(b h) n d -> b h n d', b=b)

    def forward(self, x, z, x_pos=None, z_pos=None, padding_mask=None):
        # x (z^T z)
        # x [b, n1, d]
        # z [b, n2, d]
        n1 = x.shape[1]   # x [b, n1, d]
        n2 = z.shape[1]   # z [b, n2, d]
        if padding_mask is not None:
            grid_size = torch.sum(padding_mask, dim=1).view(-1, 1, 1, 1)

        q = self.to_q(x)

        kv = self.to_kv(z).chunk(2, dim=-1)
        k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), kv)

        if (x_pos is None or z_pos is None) and self.relative_emb:
            raise Exception('Must pass in coordinates when under relative position embedding mode')
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)

        if padding_mask is None:
            if self.attn_type == 'galerkin':
                k = self.norm_wrt_domain(k, self.k_norm)
                v = self.norm_wrt_domain(v, self.v_norm)
            else:  # fourier
                q = self.norm_wrt_domain(q, self.q_norm)
                k = self.norm_wrt_domain(k, self.k_norm)
        else:
            padding_mask = repeat(padding_mask, 'b n d -> (b h) n d', h=self.heads)  # [b, n, 1]
            if self.use_ln:
                if self.attn_type == 'galerkin':
                    k = self.k_norm(k)
                    v = self.v_norm(v)
                else:  # fourier
                    q = self.q_norm(q)
                    k = self.k_norm(k)
            else:


                if self.attn_type == 'galerkin':
                    k = rearrange(k, 'b h n d -> (b h) n d')
                    v = rearrange(v, 'b h n d -> (b h) n d')

                    k = masked_instance_norm(k, padding_mask)
                    v = masked_instance_norm(v, padding_mask)

                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)
                    v = rearrange(v, '(b h) n d -> b h n d', h=self.heads)
                else:  # fourier
                    q = rearrange(q, 'b h n d -> (b h) n d')
                    k = rearrange(k, 'b h n d -> (b h) n d')

                    q = masked_instance_norm(q, padding_mask)
                    k = masked_instance_norm(k, padding_mask)

                    q = rearrange(q, '(b h) n d -> b h n d', h=self.heads)
                    k = rearrange(k, '(b h) n d -> b h n d', h=self.heads)

            padding_mask = rearrange(padding_mask, '(b h) n d -> b h n d', h=self.heads)  # [b, h, n, 1]

        if self.relative_emb:
            if self.relative_emb_dim == 2:

                x_freqs_x = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs_y = self.emb_module.forward(x_pos[..., 1], x.device)
                x_freqs_x = repeat(x_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                x_freqs_y = repeat(x_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                z_freqs_x = self.emb_module.forward(z_pos[..., 0], z.device)
                z_freqs_y = self.emb_module.forward(z_pos[..., 1], z.device)
                z_freqs_x = repeat(z_freqs_x, 'b n d -> b h n d', h=q.shape[1])
                z_freqs_y = repeat(z_freqs_y, 'b n d -> b h n d', h=q.shape[1])

                q = apply_2d_rotary_pos_emb(q, x_freqs_x, x_freqs_y)
                k = apply_2d_rotary_pos_emb(k, z_freqs_x, z_freqs_y)

            elif self.relative_emb_dim == 1:
                assert x_pos.shape[-1] == 1 and z_pos.shape[-1] == 1
                x_freqs = self.emb_module.forward(x_pos[..., 0], x.device)
                x_freqs = repeat(x_freqs, 'b n d -> b h n d', h=q.shape[1])

                z_freqs = self.emb_module.forward(z_pos[..., 0], x.device)
                z_freqs = repeat(z_freqs, 'b n d -> b h n d', h=q.shape[1])

                q = apply_rotary_pos_emb(q, x_freqs)  # query from x domain
                k = apply_rotary_pos_emb(k, z_freqs)  # key from z domain
            else:
                raise Exception('Currently doesnt support relative embedding > 2 dimensions')
        elif self.cat_pos:
            assert x_pos.size(-1) == self.pos_dim and z_pos.size(-1) == self.pos_dim
            x_pos = x_pos.unsqueeze(1)
            x_pos = x_pos.repeat([1, self.heads, 1, 1])
            q = torch.cat([x_pos, q], dim=-1)

            z_pos = z_pos.unsqueeze(1)
            z_pos = z_pos.repeat([1, self.heads, 1, 1])
            k = torch.cat([z_pos, k], dim=-1)
            v = torch.cat([z_pos, v], dim=-1)

        if padding_mask is not None:
            q = q.masked_fill(~padding_mask, 0)
            k = k.masked_fill(~padding_mask, 0)
            v = v.masked_fill(~padding_mask, 0)
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1. / grid_size)
        else:
            dots = torch.matmul(k.transpose(-1, -2), v)
            out = torch.matmul(q, dots) * (1./n2)

        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


# helpers

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


# helper
def knn(x1, x2, k):
    # x1 [b, n1, c], x2 [b, n2, c]
    inner = -2 * torch.matmul(x1, rearrange(x2, 'b n c -> b c n'))  # [b n1 n2]
    xx = torch.sum(x1 ** 2, dim=-1, keepdim=True)  # [b, n1, 1]
    yy = torch.sum(x2 ** 2, dim=-1, keepdim=True)  # [b, n2, 1]
    pairwise_distance = -xx - inner - yy.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, n1, k)
    return idx


class AttentivePooling(nn.Module):
    """Use standard scaled-dot product (or say, fourier type attention)"""
    def __init__(self,
                 dim,
                 heads,
                 dim_head,
                 pooling_ratio=8,   # 8 -> 1
                 dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.pooling_ratio = pooling_ratio

        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim+2, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.emb_module = RotaryEmbedding(dim_head // 2, scale=32)

    def forward(self, x, pos_embedding):
        # x in [b, t, n, c]
        # pos_embedding in [b, n, 2], it's just coordinates of each point
        b, t, n, c = x.shape
        batch_idx = torch.arange(b, device=pos_embedding.device).view(-1, 1)
        batch_idx = rearrange(repeat(batch_idx, 'b () -> b n', n=n), 'b n -> (b n)')  # [b*n, ]
        pos_embedding = rearrange(pos_embedding, 'b n c -> (b n) c')    # flatten

        pivot_idx = fps(pos_embedding, batch_idx, ratio=1/self.pooling_ratio)   # [b*n*1/self.pooling_ratio, ]
        pivot_pos = rearrange(pos_embedding[pivot_idx], '(b n) c -> b n c', b=b)

        pos_embedding = rearrange(pos_embedding, '(b n) c -> b n c', b=b)
        nbr_idx = knn(pivot_pos, pos_embedding, k=self.pooling_ratio + 1) # [b, s, k]

        # duplicate indexes in the time dimension

        pos_embedding = repeat(pos_embedding, 'b n c -> (b t) n c', t=t)
        nbr_idx = repeat(nbr_idx, 'b n k -> (b t) n k', t=t)
        idx_base = torch.arange(0, b*t, device=x.device).view(-1, 1, 1) * n

        nbr_idx = nbr_idx + idx_base

        x = rearrange(x, 'b t n c -> (b t n) c')[nbr_idx.view(-1), :]  # [b*t*n*k, c]
        x = rearrange(x, '(bt n k) c -> bt n k c',
                      bt=b*t, n=int(n/self.pooling_ratio), k=self.pooling_ratio + 1)

        grouped_pos = rearrange(pos_embedding, 'bt n c -> (bt n) c')[nbr_idx.view(-1), :]  # [b*t*n*k, 3]
        grouped_pos = rearrange(grouped_pos, '(bt n k) c -> bt n k c',
                                bt=b*t, n=int(n/self.pooling_ratio), k=self.pooling_ratio + 1)
        grouped_pos = grouped_pos - repeat(
            rearrange(pivot_pos, 'b n c -> b n 1 c'), 'b n () c -> (b t) n k c', t=t, k=self.pooling_ratio + 1
        )
        x = rearrange(x, 'bt n k c -> (bt n) k c')          # [btn, k, c]
        grouped_pos = rearrange(grouped_pos, 'bt n k c -> (bt n) k c')      # [btn, k, 2]
        x = torch.cat((x, grouped_pos), dim=-1)

        freqs_x = self.emb_module.forward(grouped_pos[..., 0], x.device)
        freqs_y = self.emb_module.forward(grouped_pos[..., 1], x.device)

        # attention part
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'btn k (h d) -> btn h k d', h=self.heads), qkv)
        q = apply_2d_rotary_pos_emb(q, freqs_x, freqs_y)
        k = apply_2d_rotary_pos_emb(k, freqs_x, freqs_y)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)  # similarity score

        out = torch.matmul(attn, v)
        out = rearrange(out, 'btn h k d -> btn k (h d)')
        out = self.to_out(out)  # btn, k, d
        out = out.mean(dim=1)   # btn, d
        return rearrange(out, '(b t n) c -> b t n c', b=b, t=t),\
               pivot_pos  # [b, s, 2]


class ProjDotProduct(nn.Module):
    """
    Dot product that emulates the Branch and Trunk in DeepONet,
    implementation based on:
    https://github.com/devzhk/PINO/blob/97654eba0e3244322079d85d39fe673ceceade11/baselines/model.py#L22
    """
    def __init__(self,
                 branch_dim,
                 trunk_dim,
                 inner_dim,
                 init_params=True,
                 init_method='orthogonal',    # ['xavier', 'orthogonal']
                 init_gain=None,
                 ):
        super().__init__()

        self.branch_proj = nn.Linear(branch_dim, inner_dim, bias=False)
        self.trunk_proj = nn.Linear(trunk_dim, inner_dim, bias=False)

        self.to_out = nn.Identity()

        if init_gain is None:
            self.init_gain = 1. / inner_dim
            self.diagonal_weight = 1. / inner_dim
        else:
            self.init_gain = init_gain
            self.diagonal_weight = init_gain

        self.init_method = init_method
        if init_params:
            self._init_params()

    def _init_params(self):
        if self.init_method == 'xavier':
            init_fn = xavier_uniform_
        elif self.init_method == 'orthogonal':
            init_fn = orthogonal_
        else:
            raise Exception('Unknown initialization')

        for param in self.branch_proj.parameters():
            if param.ndim > 1:
                # for q
                init_fn(param, gain=self.init_gain)

        for param in self.trunk_proj.parameters():
            if param.ndim > 1:
                # for k
                init_fn(param, gain=self.init_gain)

    def forward(self, x, z):
        # x [n1, d]
        # z [b, d]

        q = self.trunk_proj(x)
        k = self.branch_proj(z)

        out = torch.einsum('bi,ni->bn', k, q)

        return self.to_out(out)



