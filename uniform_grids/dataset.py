import random

import torch
import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from scipy.spatial import KDTree
# from dgl.geometry import farthest_point_sampler
from einops import rearrange, repeat
# from nn_module.gnn_module import list2edge, stack_graph
import numba as nb
from types import SimpleNamespace
from torch.nn.utils.rnn import pad_sequence
import random

def normalize(u):
    return (u - np.min(u)) / (np.max(u) - np.min(u) + 1e-8)


class PDEData(Dataset):
    def __init__(self,
                 dataset_path,
                 input_steps,
                 future_steps,
                 seq_num,
                 seq_len,
                 case_prefix='pde',
                 interval=4,
                 use_position=True,
                 ):
        self.seq_num = seq_num
        self.seq_len = seq_len
        self.dataset_path = dataset_path
        self.case_prefix = case_prefix
        self.future_steps = future_steps
        self.input_steps = input_steps
        self.interval = interval
        self.use_position = use_position
        if use_position:
            x0, y0 = np.meshgrid(np.linspace(0, 1, 64),
                                 np.linspace(0, 1, 64))
            xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)
            self.grid = np.repeat(xs.reshape((2, 1, 64, 64)), self.input_steps, axis=1)

    def __len__(self):
        return self.seq_num #(self.seq_len - (self.input_steps + self.future_steps) * self.interval)

    def __getitem__(self, idx):
        seed_to_read = idx#idx // (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        sample_to_read = 0#idx % (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        fname = f'{self.case_prefix}_{seed_to_read}/all_solution.npy'
        data_path = os.path.join(self.dataset_path, fname)
        # try:
        #     data = np.load(data_path)
        # except ValueError:
        #     print(seed_to_read)
        #     raise Exception('Corrupted data encountered')
        data = np.load(data_path)
        data_mean = np.mean(data)
        data_std = np.std(data)

        data = (data - data_mean) / (data_std + 1e-8)  # normalize

        in_seq = data[sample_to_read:
                      sample_to_read + self.input_steps*self.interval:
                      self.interval][None, ...]
        random_idx = sample_to_read
        # while random_idx == sample_to_read:
        #     random_idx = np.random.randint(0, self.seq_len - self.input_steps * self.interval)

        another_in_seq = data[random_idx:
                          random_idx+ self.input_steps*self.interval:
                          self.interval][None, ...]
        gt_seq = data[sample_to_read + self.input_steps*self.interval:
                      sample_to_read + (self.input_steps + self.future_steps)*self.interval:
                      self.interval][None, ...]
        if self.use_position:
            in_seq = np.concatenate((in_seq, self.grid.copy()), axis=0)
            another_in_seq = np.concatenate((another_in_seq, self.grid.copy()), axis=0)
        return in_seq.astype(np.float32),\
               gt_seq.astype(np.float32),\
               another_in_seq.astype(np.float32),\
               data_mean, data_std


class PDEGraphData(Dataset):
    def __init__(self,
                 dataset_path,
                 input_steps,
                 future_steps,
                 seq_num,
                 seq_len,
                 case_prefix='pde',
                 interval=4,
                 use_position=True,
                 n_pivotal_points=256,
                 ):
        self.seq_num = seq_num
        self.seq_len = seq_len
        self.dataset_path = dataset_path
        self.case_prefix = case_prefix
        self.future_steps = future_steps
        self.input_steps = input_steps
        self.interval = interval
        self.use_position = use_position
        self.n_pivotal_points = n_pivotal_points

        x0, y0 = np.meshgrid(np.linspace(0, 1, 64),
                             np.linspace(0, 1, 64))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 64, 64]
        self.flat_grid_np = xs.transpose((1, 2, 0)).reshape(-1, 2) # [64, 64, 2]
        self.grid = np.repeat(xs.reshape((2, 1, 64, 64)), self.input_steps, axis=1)
        self.flat_grid_tensor = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').unsqueeze(0)
        self.tree_grid = KDTree(self.flat_grid_np, boxsize=1.+1e-8)

    def __len__(self):
        return self.seq_num #(self.seq_len - (self.input_steps + self.future_steps) * self.interval)

    def __getitem__(self, idx):
        seed_to_read = idx#idx // (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        sample_to_read = 0#idx % (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        fname = f'{self.case_prefix}_{seed_to_read}/all_solution.npy'
        data_path = os.path.join(self.dataset_path, fname)
        # try:
        #     data = np.load(data_path)
        # except ValueError:
        #     print(seed_to_read)
        #     raise Exception('Corrupted data encountered')
        data = np.load(data_path)
        data_mean = np.mean(data)
        data_std = np.std(data)

        data = (data - data_mean) / (data_std + 1e-8)  # normalize

        in_seq = data[sample_to_read:
                      sample_to_read + self.input_steps*self.interval:
                      self.interval][None, ...]

        gt_seq = data[sample_to_read + self.input_steps*self.interval:
                      sample_to_read + (self.input_steps + self.future_steps)*self.interval:
                      self.interval][None, ...]

        # graph construction
        # we first sample pivotal points from the input grid
        pivotal_point_idx = farthest_point_sampler(self.flat_grid_tensor, self.n_pivotal_points).squeeze(0)
        n_in_points = 64*64
        n_pivotal_points = self.n_pivotal_points
        n_prop_points = 16*16

        # use kd_tree to construct graph
        pivotal_points = self.flat_grid_np[pivotal_point_idx]
        tree2 = KDTree(pivotal_points, boxsize=1.+1e-8)
        indexes = tree2.query_ball_tree(self.tree_grid, r=1.1 / (np.sqrt(n_pivotal_points)))
        input2pivotal_graph = list2edge(indexes, remove_self_loop=False)

        # graph within input points
        indexes = self.tree_grid.query_ball_tree(self.tree_grid, r=1.5 / (np.sqrt(n_in_points)))
        input2input_graph = list2edge(indexes, remove_self_loop=True)

        # the node we want to propagate dynamics on is the same as input nodes in current form
        prop_points = self.flat_grid_np.copy()
        indexes = self.tree_grid.query_ball_tree(tree2, r=1.1 / (np.sqrt(n_pivotal_points)))
        pivotal2prop_graph = list2edge(indexes, remove_self_loop=False)

        if self.use_position:
            in_seq = np.concatenate((in_seq, self.grid.copy()), axis=0)
        in_seq = in_seq.reshape((in_seq.shape[0], in_seq.shape[1], -1))  # [c, t, n]
        gt_seq = gt_seq.reshape((gt_seq.shape[0], gt_seq.shape[1], -1))  # [c, t, n]

        data_pack = {
            'in_seq': in_seq.astype(np.float32),
            'gt_seq': gt_seq.astype(np.float32),
            'data_mean': data_mean,
            'data_std': data_std,

            'num_of_input': n_in_points,
            'num_of_pivot': n_pivotal_points,
            'num_of_prop': n_prop_points,

            'input_points': self.flat_grid_np.astype(np.float32),
            'pivotal_points': pivotal_points.astype(np.float32),
            'prop_points': prop_points.astype(np.float32),

            'input2pivotal_graph': input2pivotal_graph.astype(np.int64),
            'input2input_graph': input2input_graph.astype(np.int64),
            'pivotal2prop_graph': pivotal2prop_graph.astype(np.int64),
        }
        return data_pack


class PDEPatchData(Dataset):
    def __init__(self,
                 dataset_path,
                 input_steps,
                 future_steps,
                 seq_num,
                 seq_len,
                 case_prefix='pde',
                 interval=4,
                 use_position=True,
                 n_patch=16,
                 cache_seq_num=10000,
                 ):
        self.seq_num = seq_num
        self.seq_len = seq_len
        self.dataset_path = dataset_path
        self.case_prefix = case_prefix
        self.future_steps = future_steps
        self.input_steps = input_steps
        self.interval = interval
        self.use_position = use_position

        x0, y0 = np.meshgrid(np.linspace(0, 1, 64),
                             np.linspace(0, 1, 64))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 64, 64]
        self.flat_grid_np = xs.transpose((1, 2, 0)).reshape(-1, 2) # [64, 64, 2]
        self.grid = np.repeat(xs.reshape((2, 1, 64, 64)), self.input_steps, axis=1)
        self.flat_grid_tensor = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c') # [64*64, 2]

        self.cache_dict = {}
        self.cache_seq_num = cache_seq_num

        self.n_patch = n_patch
        delta = 1./n_patch
        x0_patch, y0_patch = np.meshgrid(np.linspace(0+delta/2, 1-delta/2, n_patch),
                                         np.linspace(0+delta/2, 1-delta/2, n_patch))
        xs_patch = np.concatenate((x0_patch[None, ...], y0_patch[None, ...]), axis=0)  # [2, n_patch, n_patch]
        self.flat_patch_grid_np = xs_patch.transpose((1, 2, 0)).reshape(-1, 2)  # [n_patch, n_patch, 2]
        self.flat_patch_grid_tensor = rearrange(torch.from_numpy(xs_patch), 'c h w -> (h w) c') # [n_patch * n_patch, 2]

    def __len__(self):
        return self.seq_num #(self.seq_len - (self.input_steps + self.future_steps) * self.interval)

    def __getitem__(self, idx):
        seed_to_read = idx#idx // (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        sample_to_read = 0#idx % (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        if seed_to_read in self.cache_dict.keys():
            data, data_mean, data_std = self.cache_dict[seed_to_read]
        else:
            fname = f'{self.case_prefix}_{seed_to_read}/all_solution.npy'
            data_path = os.path.join(self.dataset_path, fname)
            data = np.load(data_path)
            data_mean = np.mean(data)
            data_std = np.std(data)
            if len(self.cache_dict.keys()) > self.cache_seq_num:
                self.cache_dict.pop(random.choice(self.cache_dict.keys()))
            self.cache_dict[seed_to_read] = (data, data_mean, data_std)

        data = (data - data_mean) / (data_std + 1e-8)  # normalize

        in_seq = data[sample_to_read:
                      sample_to_read + self.input_steps*self.interval:
                      self.interval][None, ...]

        gt_seq = data[sample_to_read + self.input_steps*self.interval:
                      sample_to_read + (self.input_steps + self.future_steps)*self.interval:
                      self.interval][None, ...]

        # graph construction
        # we first sample pivotal points from the input grid
        input_points = self.flat_grid_tensor.clone()
        # x // dx_patch, y // dy_patch
        dx = dy = 1./self.n_patch
        # patch_idx in shape [num_of_input_points, ]
        ix = input_points[:, 0] / dx
        ix[ix >= self.n_patch] = self.n_patch - 1
        iy = input_points[:, 1] / dy
        iy[iy >= self.n_patch] = self.n_patch - 1
        patch_idx = ix.long() + iy.long() * self.n_patch

        # vector distance to patch_center
        # this is used as position embedding of first stage attention
        dist2patch_center = input_points - self.flat_patch_grid_tensor[patch_idx]

        if self.use_position:
            in_seq = np.concatenate((in_seq, self.grid.copy()), axis=0)
        in_seq = in_seq.reshape((in_seq.shape[0], in_seq.shape[1], -1))  # [c, t, n]
        gt_seq = gt_seq.reshape((gt_seq.shape[0], gt_seq.shape[1], -1))  # [c, t, n]

        data_pack = {
            'in_seq': in_seq.astype(np.float32),
            'gt_seq': gt_seq.astype(np.float32),
            'data_mean': data_mean,
            'data_std': data_std,

            'patch_idx': patch_idx,
            'dist2patch_center': dist2patch_center.float(),

            'input_pos': self.flat_grid_tensor.clone().float(),
            'prop_pos': self.flat_grid_tensor.clone().float(),
            'patch_center': self.flat_patch_grid_tensor.clone().float(),
        }
        return data_pack

class PDEDataNew(Dataset):
    def __init__(self,
                 dataset_path,
                 input_steps,
                 future_steps,
                 seq_num,
                 seq_len,
                 case_prefix='pde',
                 interval=4,
                 use_position=True,
                 use_grad=False,
                 cache_seq_num=8000,
                 input_space_sampling_ratio=0.75,
                 output_space_sampling_ratio=0.75,
                 normalize=True,
                 shuffle_points=False,
                 ):
        self.seq_num = seq_num
        self.seq_len = seq_len
        self.dataset_path = dataset_path
        self.case_prefix = case_prefix
        self.future_steps = future_steps
        self.input_steps = input_steps
        self.interval = interval
        self.use_position = use_position
        self.use_grad = use_grad
        self.input_space_sampling_ratio = input_space_sampling_ratio
        self.output_space_sampling_ratio = output_space_sampling_ratio
        self.shuffle_points = shuffle_points

        x0, y0 = np.meshgrid(np.linspace(0, 1, 64),
                             np.linspace(0, 1, 64))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 64, 64]
        self.flat_grid_np = xs.transpose((1, 2, 0)).reshape(-1, 2) # [64, 64, 2]
        # self.grid = np.repeat(xs.reshape((2, 1, 64, 64)), self.input_steps, axis=1)
        self.grid = xs
        self.flat_grid_tensor = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c') # [64*64, 2]
        self.dx = 1./64.

        self.cache_dict = {}
        self.cache_seq_num = cache_seq_num
        self.normalize = normalize

    def __len__(self):
        return self.seq_num #(self.seq_len - (self.input_steps + self.future_steps) * self.interval)

    def __getitem__(self, idx):
        seed_to_read = idx#idx // (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        sample_to_read = 0#idx % (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        if seed_to_read in self.cache_dict.keys():
            data, data_mean, data_std = self.cache_dict[seed_to_read]
        else:
            fname = f'{self.case_prefix}_{seed_to_read}/all_solution.npy'
            data_path = os.path.join(self.dataset_path, fname)
            data = np.load(data_path)
            data_mean = np.mean(data)
            data_std = np.std(data)
            if len(self.cache_dict.keys()) > self.cache_seq_num:
                self.cache_dict.pop(random.choice(list(self.cache_dict.keys())))
            self.cache_dict[seed_to_read] = (data, data_mean, data_std)

        if self.normalize:
            data = (data - data_mean) / (data_std + 1e-8)  # normalize
        else:
            data_mean, data_std = 0., 1.

        in_seq = data[sample_to_read:
                      sample_to_read + self.input_steps*self.interval:
                      self.interval]  # [t_in, h, w]

        gt_seq = data[sample_to_read + self.input_steps*self.interval:
                      sample_to_read + (self.input_steps + self.future_steps)*self.interval:
                      self.interval]     # [t_out, h, w]

        if self.use_grad:
            if self.output_space_sampling_ratio != 0:
                raise Exception('Cannot downsample output space when using gradient information')
            # pad with PBC
            gt_padded = np.pad(gt_seq, ((0, 0), (1, 1), (1, 1)), mode='wrap')  # [t_out, h+2, w+2]
            # calculate gradient, code borrowed from Galerkin Transformer, author: Shuhao Cao
            d, s = 2, 1
            grad_x = (gt_padded[:, s:-s, d:] - gt_padded[:, s:-s, :-d]) / (d*self.dx)  # (t, S_x, S_y)
            grad_y = (gt_padded[:, d:, s:-s] - gt_padded[:, :-d, s:-s]) / (d*self.dx)  # (t, S_x, S_y)

        if self.use_position:
            in_seq = np.concatenate((in_seq, self.grid.copy()), axis=0)  # [tc + 2, n]

        in_seq = in_seq.reshape((in_seq.shape[0], -1))  # [tc+2, n]
        gt_seq = gt_seq.reshape((gt_seq.shape[0], -1))  # [tc, n]

        if self.input_space_sampling_ratio == 0:
            random_idx_input = np.arange(in_seq.shape[-1])
        else:
            random_idx_input = np.random.choice(in_seq.shape[-1], int(self.input_space_sampling_ratio *
                                                                      in_seq.shape[-1]),
                                                replace=False)

        if self.output_space_sampling_ratio == 0:
            random_idx_query = np.arange(gt_seq.shape[-1])
        else:
            random_idx_query = np.random.choice(gt_seq.shape[-1], int(self.output_space_sampling_ratio
                                                                      * gt_seq.shape[-1]),
                                                replace=False)

        if self.shuffle_points:
            np.random.shuffle(random_idx_input)

        data_pack = {
            'in_seq': in_seq.astype(np.float32)[..., random_idx_input],
            'gt_seq': gt_seq.astype(np.float32)[..., random_idx_query],
            'data_mean': data_mean,
            'data_std': data_std,

            'input_pos': self.flat_grid_tensor.clone().float()[random_idx_input, ...],
            'prop_pos': self.flat_grid_tensor.clone().float()[random_idx_query, ...],
        }
        if self.use_grad:
            data_pack['grad_x'] = grad_x.astype(np.float32)
            data_pack['grad_y'] = grad_y.astype(np.float32)  # unsqueeze channel dimension

        return data_pack


def graph_collate_fn(data_lst):
    batch_data_pack = data_lst[0]
    input2pivotal_graph_lst = []
    input2input_graph_lst = []
    pivotal2prop_graph_lst = []

    input_nodes_lst = []
    pivotal_nodes_lst = []
    prop_nodes_lst = []

    for k, v in batch_data_pack.items():
        if k in ['in_seq', 'gt_seq', 'data_mean', 'data_std']:
            batch_data_pack[k] = torch.as_tensor(v).unsqueeze(0)  # create an additional batch dimension for them
        elif k in ['input_points', 'pivotal_points', 'prop_points']:
            batch_data_pack[k] = torch.as_tensor(v)
        elif k in ['num_of_pivot']:
            pivotal_nodes_lst += [torch.as_tensor(v).unsqueeze(0)]
        elif k in ['num_of_input']:
            input_nodes_lst += [torch.as_tensor(v).unsqueeze(0)]
        elif k in ['num_of_prop']:
            prop_nodes_lst += [torch.as_tensor(v).unsqueeze(0)]
        elif k in ['input2pivotal_graph']:
            input2pivotal_graph_lst += [torch.as_tensor(v)]
        elif k in ['input2input_graph']:
            input2input_graph_lst += [torch.as_tensor(v)]
        elif k in ['pivotal2prop_graph']:
            pivotal2prop_graph_lst += [torch.as_tensor(v)]
        else:
            raise Exception(f'Unknown data key {k}')

    for data in data_lst[1:]:
        for k, v in data.items():
            if k in ['in_seq', 'gt_seq', 'data_mean', 'data_std']:
                batch_data_pack[k] = torch.cat(
                    (batch_data_pack[k], torch.as_tensor(v).unsqueeze(0)), dim=0)  # concatenate in the batch dimension
            elif k in ['input_points', 'pivotal_points', 'prop_points']:
                batch_data_pack[k] = torch.cat((batch_data_pack[k], torch.as_tensor(v)), dim=0)
            elif k in ['num_of_pivot']:
                pivotal_nodes_lst += [torch.as_tensor(v).unsqueeze(0)]
            elif k in ['num_of_input']:
                input_nodes_lst += [torch.as_tensor(v).unsqueeze(0)]
            elif k in ['num_of_prop']:
                prop_nodes_lst += [torch.as_tensor(v).unsqueeze(0)]
            elif k in ['input2pivotal_graph']:
                input2pivotal_graph_lst += [torch.as_tensor(v)]
            elif k in ['input2input_graph']:
                input2input_graph_lst += [torch.as_tensor(v)]
            elif k in ['pivotal2prop_graph']:
                pivotal2prop_graph_lst += [torch.as_tensor(v)]
            else:
                raise Exception(f'Unknown data key {k}')
    num_pivotal_nodes = torch.cat(pivotal_nodes_lst, dim=0)
    num_input_nodes = torch.cat(input_nodes_lst, dim=0)
    num_prop_nodes = torch.cat(prop_nodes_lst, dim=0)

    input2pivotal_graph = stack_graph(input2pivotal_graph_lst, num_input_nodes, num_pivotal_nodes)
    input2input_graph = stack_graph(input2input_graph_lst, num_input_nodes, num_input_nodes)
    pivotal2prop_graph = stack_graph(pivotal2prop_graph_lst, num_pivotal_nodes, num_prop_nodes)

    batch_data_pack['input2pivotal_graph'] = input2pivotal_graph
    batch_data_pack['input2input_graph'] = input2input_graph
    batch_data_pack['pivotal2prop_graph'] = pivotal2prop_graph

    return batch_data_pack


def grouping(pos, feat, patch_idx):
    """
    Args:
        pos: position in shape [n, 2]
        feat: feature in shape [n, c]
        patch_idx:   which patch each points belong to, in shape [n, ]

    Returns:
        grouped_pos:
        in shape [p, n_p, 2]  n_p is the number of points fall into each patch,
        for a patch that doesn't have that much of point, it will be padded with zeros

        grouped_feat:
        in shape [p, n_p, c]  similar to grouped_pos

        num_of_points_per_patch:
        [p, ]   how many points are in each patch
    """
    num_patch = patch_idx.max() + 1     # p
    _, num_of_points_per_patch = torch.unique(patch_idx, return_counts=True)

    points_in_each_patch = [pos[patch_idx == patch] for patch in range(num_patch)]
    feat_in_each_patch = [feat[patch_idx == patch] for patch in range(num_patch)]

    grouped_pos = pad_sequence(points_in_each_patch, batch_first=True)  # [p, n_p, 2]
    grouped_feat = pad_sequence(feat_in_each_patch, batch_first=True)  # [p, n_p, c]
    return grouped_pos, grouped_feat, num_of_points_per_patch


def temporal_grouping(feat, patch_idx):
    """

    Args:
        feat: feature in shape [c, t, n]
        patch_idx:  which patch each points belong to, in shape [n, ]

    Returns:
        grouped_feat: feature in shape [t, p, n_p, c]
    """
    c, t = feat.shape[0:2]
    num_patch = patch_idx.max() + 1     # p
    feat = rearrange(feat, 'c t n -> n (t c)')
    feat_in_each_patch = [feat[patch_idx == patch] for patch in range(num_patch)]

    grouped_feat = pad_sequence(feat_in_each_patch, batch_first=True)  # [p, n_p, c]
    grouped_feat = rearrange(grouped_feat, 'p n (t c) -> t p n c', c=c, t=t)
    return grouped_feat


def unpad(padded_patchified_pos, padded_patchified_seq, num_of_points_per_patch):
    """
    use very slow for loop, just to check correctness
    Args:
        padded_patchified_pos: padded and patchified pos, in shape [b, p, n_p, 2]
        padded_patchified_seq: padded and patchified sequence, in shape [b, t, p, n_p, c]
        num_of_points_per_patch: [b, p]
    Returns:
        unpadded_seq: [b, t, n, c]  (image structure is not preserved)
    """
    t, c = padded_patchified_seq.shape[1], padded_patchified_seq.shape[-1]
    padded_patchified_pos = repeat(padded_patchified_pos, 'b p n_p c -> (b repeat) p n_p c', repeat=t)
    padded_patchified_seq = rearrange(padded_patchified_seq, 'b t p n_p c -> (b t) p n_p c')
    num_of_points_per_patch = repeat(num_of_points_per_patch, 'b p -> (b repeat) p', repeat=t)  # [(b t), p, n_p]

    batch_unpadded_seq = []
    batch_unpadded_pos = []
    for i_b in range(padded_patchified_seq.shape[0]):
        unpadded_seq = []
        unpadded_pos = []
        for i_p in range(padded_patchified_seq.shape[1]):
            unpadded_pos += [padded_patchified_pos[i_b, i_p][:num_of_points_per_patch[i_b, i_p]].unsqueeze(0)]
            unpadded_seq += [padded_patchified_seq[i_b, i_p][:num_of_points_per_patch[i_b, i_p]].unsqueeze(0)] # [bt, n, c]
        unpadded_seq = torch.cat(unpadded_seq, dim=1)
        unpadded_pos = torch.cat(unpadded_pos, dim=1)
        batch_unpadded_pos += [unpadded_pos]
        batch_unpadded_seq += [unpadded_seq]
    batch_unpadded_seq = torch.cat(batch_unpadded_seq, dim=0)
    batch_unpadded_pos = torch.cat(batch_unpadded_pos, dim=0)

    batch_unpadded_seq = rearrange(batch_unpadded_seq, '(b t) n c -> b t n c', t=t)
    batch_unpadded_pos = rearrange(batch_unpadded_pos, '(b t) n c -> b t n c', t=t)

    return batch_unpadded_pos, batch_unpadded_seq


def patch_collate_fn(data_lst):
    batch_data_pack = {}
    for i, data in enumerate(data_lst):
        for k, v in data.items():
            if i == 0:
                if k in ['in_seq', 'gt_seq', 'data_mean', 'data_std', 'prop_pos', 'patch_center']:
                    batch_data_pack[k] = torch.as_tensor(v).unsqueeze(0)
            else:
                if k in ['in_seq', 'gt_seq', 'data_mean', 'data_std', 'prop_pos', 'patch_center']:
                    batch_data_pack[k] = torch.cat(
                        (batch_data_pack[k], torch.as_tensor(v).unsqueeze(0)), dim=0)  # concatenate in the batch dimension

        dist2patch_center = torch.as_tensor(data['dist2patch_center'])
        input_pos = torch.as_tensor(data['input_pos'])
        patch_idx = torch.as_tensor(data['patch_idx'])
        input_pos, dist2patch_center, num_of_points_per_patch = \
            grouping(input_pos, dist2patch_center, patch_idx)
        if i == 0:
            batch_data_pack['dist2patch_center'] = dist2patch_center.unsqueeze(0)  # [b, p, n_p, 2]
            batch_data_pack['input_pos'] = input_pos.unsqueeze(0) # [b, p, n_p, 2]
            batch_data_pack['num_of_points_per_patch'] = num_of_points_per_patch.unsqueeze(0) # [b, p]
        else:
            batch_data_pack['dist2patch_center'] = torch.cat((batch_data_pack['dist2patch_center'], dist2patch_center.unsqueeze(0)), dim=0)
            batch_data_pack['input_pos'] = torch.cat((batch_data_pack['input_pos'], input_pos.unsqueeze(0)), dim=0)
            batch_data_pack['num_of_points_per_patch'] = torch.cat((batch_data_pack['num_of_points_per_patch'], num_of_points_per_patch.unsqueeze(0)), dim=0)

        in_seq = torch.as_tensor(data['in_seq'])
        patchified_in_seq = temporal_grouping(in_seq, patch_idx)
        if i == 0:
            batch_data_pack['patchified_in_seq'] = patchified_in_seq.unsqueeze(0)  # [b, t, p, n_p, c]
        else:
            batch_data_pack['patchified_in_seq'] = torch.cat((batch_data_pack['patchified_in_seq'], patchified_in_seq.unsqueeze(0)), dim=0)

    return batch_data_pack


class PDE_COEF_data(Dataset):
    def __init__(self,
                 id_path,        # ids of sequences to be used, an npy file
                 dataset_path,
                 input_steps,
                 future_steps,
                 seq_len,
                 coefficients_key,
                 case_prefix='pde',
                 interval=1,
                 use_position=True,
                 ):
        self.ids = np.load(id_path)
        self.seq_len = seq_len
        self.dataset_path = dataset_path
        self.case_prefix = case_prefix
        self.future_steps = future_steps
        self.input_steps = input_steps
        self.interval = interval
        self.use_position = use_position
        self.coefficients_key = coefficients_key
        if use_position:
            x0, y0 = np.meshgrid(np.linspace(0, 1, 64),
                                 np.linspace(0, 1, 64))
            xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)
            self.grid = np.repeat(xs.reshape((2, 1, 64, 64)), self.input_steps, axis=1)

    def __len__(self):
        return self.ids.shape[0] #(self.seq_len - (self.input_steps + self.future_steps) * self.interval)

    def __getitem__(self, idx):
        seed_to_read = self.ids[idx]#idx // (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        sample_to_read = 0#idx % (self.seq_len - (self.input_steps + self.future_steps) * self.interval)
        fname = f'{self.case_prefix}_{seed_to_read}/all_solution.npy'
        data_path = os.path.join(self.dataset_path, fname)

        data = np.load(data_path)
        data_mean = np.mean(data)
        data_std = np.std(data)

        info_fname = f'{self.case_prefix}_{seed_to_read}/system_info.npz'
        info_path = os.path.join(self.dataset_path, info_fname)
        coefficients = []
        with np.load(info_path) as info_data:
            coefficient = info_data[self.coefficients_key]
            coefficients += [coefficient]

        data = (data - data_mean) / (data_std + 1e-8)  # normalize

        in_seq = data[sample_to_read:
                      sample_to_read + self.input_steps*self.interval:
                      self.interval][None, ...]
        random_idx = sample_to_read
        # while random_idx == sample_to_read:
        #     random_idx = np.random.randint(0, self.seq_len - self.input_steps * self.interval)

        another_in_seq = data[random_idx:
                          random_idx+ self.input_steps*self.interval:
                          self.interval][None, ...]
        gt_seq = data[sample_to_read + self.input_steps*self.interval:
                      sample_to_read + (self.input_steps + self.future_steps)*self.interval:
                      self.interval][None, ...]
        if self.use_position:
            in_seq = np.concatenate((in_seq, self.grid.copy()), axis=0)
            another_in_seq = np.concatenate((another_in_seq, self.grid.copy()), axis=0)
        return in_seq.astype(np.float32),\
               gt_seq.astype(np.float32),\
               another_in_seq.astype(np.float32),\
               data_mean, data_std, coefficients


def get_data_loader(opt, dataset_path, seq_num, seq_len, train=True):
    dataset = PDEData(dataset_path, opt.in_seq_len, opt.out_seq_len, seq_num, seq_len, interval=opt.interval)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size if train else opt.batch_size*4,
                            shuffle=True if train else False, num_workers=0, pin_memory=False)

    return dataloader


def get_graph_data_loader(opt, dataset_path, seq_num, seq_len, train=True):
    dataset = PDEGraphData(dataset_path, opt.in_seq_len, opt.out_seq_len, seq_num, seq_len, interval=opt.interval)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size if train else opt.batch_size*2,
                            shuffle=True if train else False, num_workers=2, pin_memory=False,
                            collate_fn=graph_collate_fn)

    return dataloader


def get_patch_data_loader(opt, dataset_path, seq_num, seq_len, train=True):
    dataset = PDEPatchData(dataset_path, opt.in_seq_len, opt.out_seq_len, seq_num, seq_len, interval=opt.interval)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size if train else opt.batch_size*2,
                            shuffle=True if train else False, num_workers=4, pin_memory=True,
                            collate_fn=patch_collate_fn)

    return dataloader


def get_new_data_loader(opt, dataset_path, seq_num, seq_len, train=True, use_grad=False):
    if not use_grad:
        dataset = PDEDataNew(dataset_path, opt.in_seq_len, opt.out_seq_len, seq_num, seq_len,
                             interval=opt.interval,
                             input_space_sampling_ratio=opt.sampling_ratio if train else 0,
                             output_space_sampling_ratio=0,                                                   # opt.sampling_ratio if train else 0,
                             use_grad=False,
                             shuffle_points=True if train else False,
                             )
    else:
        dataset = PDEDataNew(dataset_path, opt.in_seq_len, opt.out_seq_len, seq_num, seq_len,
                             interval=opt.interval,
                             input_space_sampling_ratio=opt.sampling_ratio if train else 0,
                             output_space_sampling_ratio=0,
                             use_grad=True,
                             shuffle_points=True if train else False,
                             )
    dataloader = DataLoader(dataset, batch_size=opt.batch_size if train else opt.batch_size*2,
                            shuffle=True if train else False, num_workers=1, pin_memory=True)

    return dataloader


def get_data_loader_with_coefficient(opt, dataset_path, id_path, coefficients_key, seq_num, seq_len, train=True):
    dataset1 = PDEData(dataset_path, opt.in_seq_len, opt.out_seq_len, seq_num, seq_len)
    dataloader1 = DataLoader(dataset1, batch_size=opt.batch_size if train else opt.batch_size*4,
                            shuffle=True if train else False, num_workers=0, pin_memory=False)
    dataset2 = PDE_COEF_data(id_path,
                             dataset_path, opt.in_seq_len, opt.out_seq_len, seq_len, coefficients_key)
    dataloader2 = DataLoader(dataset2, batch_size=opt.batch_size if train else opt.batch_size * 4,
                            shuffle=True if train else False, num_workers=0, pin_memory=False)
    return dataloader1, dataloader2



if __name__ == '__main__':
    # dat = PDEPatchData('./ns2d_unit_test_data', 10, 190, 10, 200)
    # dataloader = DataLoader(dat, batch_size=4,
    #                         shuffle=True, num_workers=2, pin_memory=False, collate_fn=patch_collate_fn)
    # data_iter = iter(dataloader)
    # data = next(data_iter)
    # # print(data)
    # # print(data['in_seq'].shape)
    # # plt.imshow(data['in_seq'][0, 0, 9].reshape(64, 64))
    # # plt.show()
    # print(data['input_pos'].shape)
    # print(data['num_of_points_per_patch'])
    # print(data['patchified_in_seq'].shape)
    # unpad_pos, unpad_seq = unpad(data['input_pos'], data['patchified_in_seq'], data['num_of_points_per_patch'])
    # plt.gca().invert_yaxis()
    #
    # plt.scatter(unpad_pos[0, -1, :, 0].numpy(), unpad_pos[0, -1, :, 1].numpy(),
    #             c=unpad_seq[0, -1, :, 0].numpy(), s=10)
    # plt.show()
    # plt.imshow(data['in_seq'][0, 0, 9].reshape(64, 64))
    # plt.show()
    # data_pack = dat[0]
    # for i in range(data_pack.pivotal_points.shape[0]):
    #     plt.scatter(data_pack.pivotal_points[i, 0],
    #                 data_pack.pivotal_points[i, 1])
    # plt.show()
    # plt.imshow(in_seq[0, 0])
    # plt.show()
    # in_seq, gt, another_in_seq, m, s = dat[16]
    # plt.imshow(in_seq[0, 0])
    # plt.show()
    res = 3
    x0, y0 = np.meshgrid(np.linspace(0, 1, res),
                         np.linspace(0, 1, res))
    xs = np.concatenate((x0[..., None], y0[..., None]), axis=-1)  # [res, res, 2]
    grid1 = xs.reshape((3, 3, 2))  # [res, res, 2]
    grid1 = torch.from_numpy(grid1).float()

    # gridx = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    # gridx = gridx.reshape(res, 1, 1).repeat([1, res, 1])
    # gridy = torch.tensor(np.linspace(0, 1, res), dtype=torch.float32)
    # gridy = gridy.reshape(1, res, 1).repeat([res, 1, 1])
    # grid2 = torch.cat((gridx, gridy), dim=-1).reshape((3, 3, 2))
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    x, y = np.meshgrid(x, y)
    grid = np.stack([x, y], axis=-1)
    grid2 = np.c_[x.ravel(), y.ravel()]

    print(grid1)
    print(grid2)