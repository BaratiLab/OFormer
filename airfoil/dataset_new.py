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
import glob


def get_position_noise(sequence: np.ndarray, noise_std, use_rw=False):
    """
    Returns random-walk noise in the velocity applied to the position.
    """
    if use_rw:
        velocity_sequence = sequence[1:] - sequence[:-1]
        velocity_sequence_noise = np.random.randn(*(velocity_sequence.shape)) * noise_std
        velocity_sequence_noise = np.cumsum(velocity_sequence_noise, axis=0)
        sequence_noise = np.concatenate([
            np.zeros_like(velocity_sequence_noise[0:1]),
            np.cumsum(velocity_sequence_noise, axis=0)], axis=0)
    else:
        sequence_noise = np.random.randn(*(sequence.shape)) * noise_std

    return sequence_noise


class BurgersData(Dataset):
    def __init__(self,
                 dataset,
                 timewindow,
                 nsteps,
                 interval,
                 res=200,
                 pushforward=2,
                 transpose_output=False,
                 ):
        n, n_t, n_x = dataset.shape
        self.dataset = dataset
        self.seq_num = n
        self.seq_len = n_t // interval
        self.nsteps = nsteps
        self.interval = interval
        self.timewindow = timewindow
        self.pushforward = pushforward
        self.transpose_output = transpose_output

        grid = np.linspace(0, 16., res, dtype=np.float32)
        self.grid = grid.reshape((res, 1))

    def __len__(self):
        return self.seq_num * (self.seq_len - (self.timewindow + self.pushforward * self.nsteps))

    def __getitem__(self, idx):

        seed_to_read = idx // (self.seq_len - (self.timewindow + self.pushforward * self.nsteps))
        tstart = idx % (self.seq_len - (self.timewindow + self.pushforward * self.nsteps))

        sequence = self.dataset[seed_to_read,
                   tstart*self.interval:(tstart+self.timewindow + self.pushforward * self.nsteps)*self.interval:self.interval].copy()

        in_seq = sequence[:self.timewindow].transpose(1, 0)
        out_seq = sequence[self.timewindow:]

        if self.transpose_output:
            out_seq = out_seq.transpose(1, 0)
        return in_seq.astype(np.float32), out_seq.astype(np.float32), self.grid.copy(), tstart*self.interval


class NoisyBurgersData(Dataset):
    def __init__(self,
                 dataset,
                 nsteps,
                 interval,
                 res=200,
                 transpose_output=False,
                 add_noise=False,
                 noise_level=(0.003, 1),
                 num_levels=30,
                 ):
        n, n_t, n_x = dataset.shape
        self.dataset = dataset
        self.seq_num = n
        self.seq_len = n_t // interval
        self.nsteps = nsteps
        self.interval = interval
        self.transpose_output = transpose_output
        self.add_noise = add_noise
        self.noise_level = noise_level
        self.noise_stds = np.exp(np.linspace(np.log(noise_level[0]), np.log(noise_level[1]), num_levels)).astype(np.float32)
        print(self.noise_stds)
        grid = np.linspace(0, 16., res, dtype=np.float32)
        self.grid = grid.reshape((res, 1))
        if self.add_noise:
            print('Noise will be added to data')

    def __len__(self):
        return self.seq_num * (self.seq_len - self.nsteps)

    def __getitem__(self, idx):

        seed_to_read = idx // (self.seq_len - self.nsteps)
        tstart = idx % (self.seq_len - self.nsteps)

        clean_sequence = self.dataset[seed_to_read,
                   tstart*self.interval:(tstart + self.nsteps)*self.interval:self.interval].copy().astype(np.float32)

        noise_std = np.random.choice(self.noise_stds, size=1)
        noise = get_position_noise(self.dataset[seed_to_read], noise_std, use_rw=False)
        sequence = clean_sequence + noise[tstart*self.interval:(tstart + self.nsteps)*self.interval:self.interval].astype(np.float32)

        return sequence, clean_sequence, self.grid.copy(), noise_std


class NavierStokesData(Dataset):
    def __init__(self,
                 dataset,
                 timewindow,
                 nsteps,
                 interval,
                 res=64,
                 pushforward=2,
                 ):
        n_t, _, _, n = dataset.shape
        self.dataset = dataset
        self.seq_num = n
        self.seq_len = n_t // interval
        self.timewindow = timewindow
        self.nsteps = nsteps
        self.interval = interval
        self.res = res
        self.pushforward = pushforward

        x0, y0 = np.meshgrid(np.linspace(0, 1, res),
                             np.linspace(0, 1, res))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 64, 64]
        self.grid = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').float()  # [64*64, 2]

    def __len__(self):
        return self.seq_num * (self.seq_len - (self.timewindow + self.pushforward * self.nsteps))

    def __getitem__(self, idx):
        seed_to_read = idx // (self.seq_len - (self.timewindow + self.pushforward * self.nsteps))
        tstart = idx % (self.seq_len - (self.timewindow + self.pushforward * self.nsteps))

        sequence = self.dataset[
                   tstart*self.interval:
                   (tstart + self.timewindow + self.nsteps * self.pushforward)*self.interval:
                   self.interval, ..., seed_to_read].copy()
        in_seq = sequence[:self.timewindow].reshape(-1, self.res*self.res)
        out_seq = sequence[self.timewindow:].reshape(-1, self.res*self.res)

        return in_seq.astype(np.float32), out_seq.astype(np.float32), self.grid.clone()


class NavierStokesData2D(Dataset):
    def __init__(self,
                 dataset,
                 timewindow,
                 nsteps,
                 interval,
                 res=64,
                 pushforward=2,
                 ):
        n_t, _, _, n = dataset.shape
        self.dataset = dataset
        self.seq_num = n
        self.seq_len = n_t // interval
        self.timewindow = timewindow
        self.nsteps = nsteps
        self.interval = interval
        self.res = res
        self.pushforward = pushforward

        x0, y0 = np.meshgrid(np.linspace(0, 1, res),
                             np.linspace(0, 1, res))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 64, 64]
        self.grid = torch.from_numpy(xs).float()  # [2, 64*64]

    def __len__(self):
        return self.seq_num * (self.seq_len - (self.timewindow + self.pushforward * self.nsteps))

    def __getitem__(self, idx):
        seed_to_read = idx // (self.seq_len - (self.timewindow + self.pushforward * self.nsteps))
        tstart = idx % (self.seq_len - (self.timewindow + self.pushforward * self.nsteps))

        sequence = self.dataset[
                   tstart*self.interval:
                   (tstart + self.timewindow + self.nsteps * self.pushforward)*self.interval:
                   self.interval, ..., seed_to_read].copy()
        in_seq = sequence[:self.timewindow].reshape(-1, self.res, self.res).transpose(1, 2, 0)
        out_seq = sequence[self.timewindow:].reshape(-1, self.res, self.res).transpose(1, 2, 0)

        return in_seq.astype(np.float32), out_seq.astype(np.float32), self.grid.clone(), tstart*self.interval


class ReactionDiffusion2DData(Dataset):
    def __init__(self,
                 dataset,
                 timewindow,
                 nsteps,
                 interval,
                 res=64,
                 ):
        n, n_t, _, _, c = dataset.shape
        self.dataset = dataset
        self.seq_num = n
        self.seq_len = n_t // interval
        self.timewindow = timewindow
        self.nsteps = nsteps
        self.interval = interval
        self.res = res
        x0, y0 = np.meshgrid(np.linspace(0, 1, res),
                             np.linspace(0, 1, res))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 64, 64]
        self.grid = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').float()  # [64*64, 2]

    def __len__(self):
        return self.seq_num * (self.seq_len - (self.timewindow + 2 * self.nsteps))

    def __getitem__(self, idx):
        seed_to_read = idx // (self.seq_len - (self.timewindow + 2 * self.nsteps))
        tstart = idx % (self.seq_len - (self.timewindow + 2 * self.nsteps))

        sequence = self.dataset[seed_to_read,
                   tstart*self.interval:
                   (tstart + self.timewindow + self.nsteps*2)*self.interval:
                   self.interval].copy()
        in_seq = sequence[:self.timewindow].reshape(-1, self.res*self.res, 2)  # [t_in, n*n, 2]
        out_seq = sequence[self.timewindow:].reshape(-1, self.res*self.res, 2)  # [t_out*2, n*n, 2]

        return in_seq.astype(np.float32), out_seq.astype(np.float32), self.grid.clone()


class NavierStokesReconsData(Dataset):
    def __init__(self,
                 dataset,
                 timewindow,
                 interval,
                 sample_ratio=0.25,
                 res=64,
                 test_mode=False                # return an array with points' value not sampled set to zero
                 ):
        n_t, _, _, n = dataset.shape
        self.dataset = dataset
        self.seq_num = n
        self.seq_len = n_t // interval
        self.timewindow = timewindow
        self.interval = interval
        self.res = res
        self.sample_ratio = sample_ratio

        self.test_mode = test_mode

        x0, y0 = np.meshgrid(np.linspace(0, 1, res),
                             np.linspace(0, 1, res))
        xs = np.concatenate((x0[None, ...], y0[None, ...]), axis=0)  # [2, 64, 64]
        self.grid = rearrange(torch.from_numpy(xs), 'c h w -> (h w) c').float()  # [64*64, 2]

    def __len__(self):
        return self.seq_num * (self.seq_len - self.timewindow)

    def __getitem__(self, idx):
        seed_to_read = idx // (self.seq_len - self.timewindow)
        tstart = idx % (self.seq_len - self.timewindow)

        sequence = self.dataset[
                   tstart*self.interval:
                   (tstart + self.timewindow)*self.interval:
                   self.interval, ..., seed_to_read].copy()

        in_seq = sequence[:self.timewindow].reshape(-1, self.res*self.res)

        random_idx_input = np.random.choice(in_seq.shape[-1],
                                            int(self.sample_ratio * in_seq.shape[-1]),
                                            replace=False)
        in_seq_sampled = in_seq[..., random_idx_input].copy()
        input_pos = self.grid.clone()[random_idx_input]

        if self.test_mode:
            masked_in_seq = in_seq.copy()
            masked_in_seq = (masked_in_seq - masked_in_seq.min()) / (masked_in_seq.max() - masked_in_seq.min())
            mask = np.zeros_like(masked_in_seq)
            mask[..., random_idx_input] = 1.  # [b t n]
            masked_in_seq *= mask
            return in_seq_sampled.astype(np.float32),\
                   np.transpose(in_seq.astype(np.float32), (1, 0)),\
                   input_pos, \
                   self.grid.clone(), \
                np.transpose(masked_in_seq, (1, 0))

        return in_seq_sampled.astype(np.float32),\
               np.transpose(in_seq.astype(np.float32), (1, 0)),\
               input_pos,\
               self.grid.clone()


class HeatCavityData(Dataset):
    def __init__(self,
                 data_dir,
                 num_case,
                 test_mode=False                # return an array with points' value not sampled set to zero
                 ):
        self.data_dir = data_dir
        self.num_case = num_case
        idxs = np.arange(0, self.num_case)
        np.random.seed(44)   # deterministic
        np.random.shuffle(idxs)
        if not test_mode:
            print('Using training data')
            self.idxs = idxs[:int(self.num_case * 0.9)]
        else:
            print('Using testing data')
            self.idxs = idxs[int(self.num_case * 0.9):]

        self.cache = {}
        self.prepare_data()
        self.max_num_points = 0

    def __len__(self):
        return len(self.idxs)

    def prepare_data(self):
        vel_all = []
        prs_all = []
        temp_all = []
        ra_all = []
        num_points = []

        for seed_to_read in range(self.num_case):
            path = os.path.join(self.data_dir, f'case_{seed_to_read}')

            prs_init = np.load(os.path.join(path, 'press_init.npy'))
            prs_vals = np.load(os.path.join(path, 'press_vals.npy'))

            vel_init = np.load(os.path.join(path, 'vel_init.npy'))
            vel_vals = np.load(os.path.join(path, 'vel_vals.npy'))

            temp_init = np.load(os.path.join(path, 'temperature_init.npy'))
            temp_vals = np.load(os.path.join(path, 'temperature_vals.npy'))

            param = np.load(os.path.join(path, 'info.npz'))['Ra'].item()

            coords = np.load(os.path.join(path, 'coords_arr.npy'))

            vel_all.append(vel_vals.reshape(-1))
            prs_all.append(prs_vals.reshape(-1))
            temp_all.append(temp_vals.reshape(-1))
            ra_all.append(param)
            num_points.append(coords.shape[0])

            x = np.concatenate((vel_init, prs_init[..., None], temp_init[..., None]), axis=1)
            y = np.concatenate((vel_vals, prs_vals[..., None], temp_vals[..., None]), axis=1)

            self.cache[seed_to_read] = (x, y, coords, param)

        vel_all = np.concatenate(vel_all, axis=0)
        prs_all = np.concatenate(prs_all, axis=0)
        temp_all = np.concatenate(temp_all, axis=0)
        self.max_num_points = np.max(num_points)

        print(f'Max num of points: {self.max_num_points}')

        # perform normalization
        temp_mean, temp_std = np.mean(temp_all), np.std(temp_all)
        vel_mean, vel_std = np.mean(vel_all), np.std(vel_all)
        prs_mean, prs_std = np.mean(prs_all), np.std(prs_all)
        param_min, param_max = np.min(ra_all), np.max(ra_all)

        self.statistics = {'temp_mean': temp_mean,
                           'temp_std': temp_std,
                           'vel_mean': vel_mean,
                           'vel_std': vel_std,
                           'prs_mean': prs_mean,
                           'prs_std': prs_std,
                           'param_min': param_min,
                           'param_max': param_max}

        for seed_to_read in range(self.num_case):
            x, y, coords, param = self.cache[seed_to_read]
            x[:, 0:2] = (x[:, 0:2] - vel_mean) / vel_std
            # y[:, 0:2] = (y[:, 0:2] - vel_mean) / vel_std
            x[:, 2] = (x[:, 2] - prs_mean) / prs_std
            # y[:, 2] = (y[:, 2] - prs_mean) / prs_std
            x[:, 3] = (x[:, 3] - temp_mean) / temp_std
            # y[:, 3] = (y[:, 3] - temp_mean) / temp_std
            param = (param - param_min) / (param_max - param_min)

            bound_mask = self.get_bound_mask(coords)
            x, y, coords, pad_mask, bound_mask = self.pad_data(x, y, coords, bound_mask)
            self.cache[seed_to_read] = (x, y, coords, param, pad_mask, bound_mask)

    def get_bound_mask(self, coords):
        # bound mask order: left, right, bottom, top
        eps = 1e-6
        bound_mask = np.zeros([coords.shape[0], 4])

        bound_mask[coords[:, 0] < eps, 0] = 1  # left bound
        bound_mask[np.abs(coords[:, 0] - 1) < eps, 1] = 1  # right bound
        bound_mask[coords[:, 1] < eps, 2] = 1  # bottom bound
        bound_mask[np.abs(coords[:, 1] - 1) < eps, 3] = 1  # top bound

        return bound_mask

    def pad_data(self, x, y, coords, bound_mask):
        num_points = coords.shape[0]
        if num_points > self.max_num_points:
            raise ValueError('Number of points is larger than max_num_points')
        pad_mask = np.zeros((self.max_num_points, 1))
        pad_mask[:num_points] = 1

        pad_size = self.max_num_points - num_points
        pad_coords = np.zeros((pad_size, 2))
        pad_x = np.zeros((pad_size, x.shape[1]))
        pad_y = np.zeros((pad_size, y.shape[1]))
        pad_bound = np.zeros((pad_size, bound_mask.shape[1]))

        x = np.concatenate((x, pad_x), axis=0)
        y = np.concatenate((y, pad_y), axis=0)
        coords = np.concatenate((coords, pad_coords), axis=0)
        bound_mask = np.concatenate((bound_mask, pad_bound), axis=0)

        return x, y, coords, pad_mask, bound_mask

    def __getitem__(self, idx):
        seed_to_read = self.idxs[idx]
        x, y, coords, param, pad_mask, bound_mask = self.cache[seed_to_read]
        # transform all data into float tensor
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        coords = coords.astype(np.float32)
        param = param.astype(np.float32)
        pad_mask = pad_mask.astype(np.bool)
        bound_mask = bound_mask.astype(np.bool)

        return x, y, coords, param, pad_mask, bound_mask


class ElectroStatData(Dataset):
    def __init__(self,
                 data_path,               # path to train data, a pickle file
                 ):
        self.data_path = data_path
        self.data = pickle.load(open(data_path, 'rb'))

        self.cache = {}
        self.prepare_data()
        self.max_num_points = 0

    def __len__(self):
        return len(self.data)

    def prepare_data(self):
        # feat_all = []
        pot_all = []
        field_all = []
        num_points = []

        for case_no, dat in enumerate(self.data):
            feat = dat['data_x']
            label = dat['data_y']

            coords = feat[:, :2]
            bound_mask = np.abs(feat[:, 3:4] - 1) < 1e-10   # keepdim
            num_points.append(coords.shape[0])
            # feat_all.append(feat[:, :2])
            bound_mask_ = bound_mask.astype(np.bool)[:, 0]
            pot_all.append(label[~bound_mask_, 0])
            field_all.append(label[~bound_mask_, 1:])
            x = feat#[:, 2:]
            y = label

            self.cache[case_no] = (x, y, coords, bound_mask)

        # feat_all = np.concatenate(feat_all, axis=0)
        pot_all = np.concatenate(pot_all, axis=0)
        field_all = np.concatenate(field_all, axis=0)

        self.max_num_points = np.max(num_points)

        print(f'Max num of points: {self.max_num_points}')
        print(f'Min num of points: {np.min(num_points)}')
        print(f'Mean num of points: {np.mean(num_points)}')

        for case_no in range(len(self.data)):
            x, y, coords, bound_mask = self.cache[case_no]
            x, y, coords, pad_mask, bound_mask = self.pad_data(x, y, coords, bound_mask)
            self.cache[case_no] = (x, y, coords, pad_mask, bound_mask)

        # perform normalization
        self.statistics = {}

        # for j in range(2, feat_all.shape[1]):
        #     self.statistics[f'feat_{j}_mean'] = np.mean(feat_all[:, j])
        #     self.statistics[f'feat_{j}_std'] = np.std(feat_all[:, j])

        self.statistics['pot_mean'] = np.mean(pot_all)
        self.statistics['pot_std'] = np.std(pot_all)
        self.statistics['pot_min'] = np.min(pot_all)
        self.statistics['pot_max'] = np.max(pot_all)

        self.statistics['field_mean'] = np.mean(field_all)
        self.statistics['field_std'] = np.std(field_all)
        self.statistics['field_min'] = np.min(field_all)
        self.statistics['field_max'] = np.max(field_all)

        print(self.statistics)

    def pad_data(self, x, y, coords, bound_mask):
        num_points = coords.shape[0]
        if num_points > self.max_num_points:
            raise ValueError('Number of points is larger than max_num_points')
        pad_mask = np.zeros((self.max_num_points, 1))
        pad_mask[:num_points] = 1

        pad_size = self.max_num_points - num_points
        pad_coords = np.zeros((pad_size, 2))
        pad_x = np.zeros((pad_size, x.shape[1]))
        pad_y = np.zeros((pad_size, y.shape[1]))
        pad_bound = np.zeros((pad_size, bound_mask.shape[1]))

        x = np.concatenate((x, pad_x), axis=0)
        y = np.concatenate((y, pad_y), axis=0)
        coords = np.concatenate((coords, pad_coords), axis=0)
        bound_mask = np.concatenate((bound_mask, pad_bound), axis=0)

        return x, y, coords, pad_mask, bound_mask

    def __getitem__(self, idx):
        x, y, coords, pad_mask, bound_mask = self.cache[idx]
        # transform all data into float tensor
        x = x.astype(np.float32)
        y = y.astype(np.float32)
        coords = coords.astype(np.float32)
        pad_mask = pad_mask.astype(np.bool)
        bound_mask = bound_mask.astype(np.bool)

        return x, y, coords, pad_mask, bound_mask


class AirfoilData(Dataset):
    # dataset courtesy of https://sites.google.com/view/meshgraphnets#h.i6eb12yvkyfc
    NODE_TYPE_MAP = {0: 0, 2: 1, 4: 2}
    # 0: normal
    # 2: airfoil boundary
    # 4: open area boundary
    def __init__(self,
                data_path,               # path to train data, a pickle file
                interval=4,              # interval of data to be used
                tw=5,
                use_normalized=True,
                return_cells=False,      # return cells (triangles) for visualization
                markovian=False,
                pushforward=2,
                dry_run=False,
                ):
        self.interval = interval
        self.tw = tw
        self.data_path = data_path
        self.fname_lst = glob.glob(data_path + '/*.npz')
        self.return_cells = return_cells
        self.use_normalized = use_normalized
        self.dry_run = dry_run

        self.markovian = markovian
        self.pushforward = pushforward

        self.cache = {}
        self.prepare_data()

    def __len__(self):
        return len(self.fname_lst)

    def prepare_data(self):
        node_type_all = []
        dns_all = []
        prs_all = []
        vel_all = []
        pos_all = []
        cell_all = []

        for fname in self.fname_lst:
            data = np.load(fname)
            node_type = data['node_type'][None, ::self.interval]  # [0, 2, 4]
            pos_seq = data['pos'][::self.interval].astype(np.float32)
            cell_seq = data['cells'][::self.interval].astype(np.float32)
            assert np.sum(pos_seq[-1] - pos_seq[0]) < 1e-5
            assert np.sum(node_type[-1] - node_type[0]) < 1e-5

            dns = data['dns'][None, ::self.interval].astype(np.float32)  # [1, time, n, 1]
            prs = data['prs'][None, ::self.interval].astype(np.float32)
            vel = data['vel'][None, ::self.interval].astype(np.float32)

            dns_all.append(dns)
            prs_all.append(prs)
            vel_all.append(vel)
            node_type_all.append(node_type[0, 0])
            pos_all.append(pos_seq[0:1])
            cell_all.append(cell_seq[0])

            del data

        dns_all = np.concatenate(dns_all, axis=0)
        prs_all = np.concatenate(prs_all, axis=0)
        vel_all = np.concatenate(vel_all, axis=0)
        pos_all = np.concatenate(pos_all, axis=0)  # [b, n, 2]

        print(f'dns shape: {dns_all.shape}')
        print(f'prs shape: {prs_all.shape}')
        print(f'vel shape: {vel_all.shape}')
        print(f'pos shape: {pos_all.shape}')

        # perform normalization
        self.statistics = {}
        self.statistics['dns_mean'] = np.mean(dns_all)
        self.statistics['dns_std'] = np.std(dns_all)
        self.statistics['prs_mean'] = np.mean(prs_all)
        self.statistics['prs_std'] = np.std(prs_all)
        self.statistics['vel_x_mean'] = np.mean(vel_all[..., 0])
        self.statistics['vel_x_std'] = np.std(vel_all[..., 0])
        self.statistics['vel_y_mean'] = np.mean(vel_all[..., 1])
        self.statistics['vel_y_std'] = np.std(vel_all[..., 1])

        self.statistics['pos_x_min'] = np.min(pos_all[..., 0])  # left bound
        self.statistics['pos_x_max'] = np.max(pos_all[..., 0])  # right bound
        self.statistics['pos_y_min'] = np.min(pos_all[..., 1])  # lower bound
        self.statistics['pos_y_max'] = np.max(pos_all[..., 1])  # upper bound

        self.statistics['x_len'] = self.statistics['pos_x_max'] - self.statistics['pos_x_min']
        self.statistics['y_len'] = self.statistics['pos_y_max'] - self.statistics['pos_y_min']

        for key, value in self.statistics.items():
            print(f'{key}: {value}')

        if self.dry_run:
            del dns_all, prs_all, vel_all, pos_all, cell_all
            return

        for seq_no in range(len(node_type_all)):
            node_type = node_type_all[seq_no]
            node_type_mapped = np.vectorize(self.NODE_TYPE_MAP.__getitem__)(node_type)
            dns = dns_all[seq_no]
            prs = prs_all[seq_no]
            vel = vel_all[seq_no]
            pos = pos_all[seq_no]

            dns_norm = (dns - self.statistics['dns_mean']) / self.statistics['dns_std']
            prs_norm = (prs - self.statistics['prs_mean']) / self.statistics['prs_std']
            vel_norm = np.concatenate(
                ((vel[..., 0] - self.statistics['vel_x_mean'])[..., None] / self.statistics['vel_x_std'],
                 (vel[..., 1] - self.statistics['vel_y_mean'])[..., None] / self.statistics['vel_y_std']), axis=2)

            pos_norm = np.concatenate(((pos[..., 0] - self.statistics['pos_x_min'])[..., None] / self.statistics['x_len'],
                                        (pos[..., 1] - self.statistics['pos_y_min'])[..., None] / self.statistics['y_len']), axis=1)
            if self.use_normalized:
                dns = dns_norm
                prs = prs_norm
                vel = vel_norm

            pos_offseted = np.concatenate(((pos[..., 0] - self.statistics['pos_x_min'])[..., None],
                                           (pos[..., 1] - self.statistics['pos_y_min'])[..., None]), axis=1)

            if self.markovian:
                x = np.concatenate((vel_norm,
                                    dns_norm,
                                    prs_norm,
                                    pos_norm[None, ...].repeat(vel_norm.shape[0], axis=0)), axis=2)  # [seq_len, n, 6]
                node_type = node_type_mapped
                self.cache[seq_no] = (x.astype(np.float32),
                                      node_type.astype(np.int64),
                                      pos_offseted, cell_all[seq_no])
            else:
                x = np.concatenate((vel_norm[:self.tw],
                                    dns_norm[:self.tw],
                                    prs_norm[:self.tw],
                                    pos_norm[None, ...].repeat(self.tw, axis=0)), axis=2)  # [seq_len, n, 6]
                node_type = node_type_mapped
                y = np.concatenate((vel[self.tw:],
                                    dns[self.tw:],
                                    prs[self.tw:]), axis=2)  # unnormalized  [seq_len, n, 4]
                self.cache[seq_no] = (x.astype(np.float32), y.astype(np.float32), node_type.astype(np.int64),
                                      pos_offseted, cell_all[seq_no])

    def __getitem__(self, idx):
        if self.markovian:
            x, node_type, pos, cell = self.cache[idx]
            start_idx = np.random.randint(0, x.shape[0] - self.pushforward - self.tw)
            x, y = x[start_idx:start_idx + self.tw], x[start_idx + self.tw:start_idx + self.tw + self.pushforward, ..., :4]
        else:
            x, y, node_type, pos, cell = self.cache[idx]

        if self.return_cells:
            return x, y, node_type, pos, cell
        else:
            return x, y, node_type, pos

    def get_statistics(self):
        return self.statistics


if __name__ == '__main__':
    train_data_path = '../pde_data2/prcoessed_airfoil_train_data_dt6'
    train_data = AirfoilData(train_data_path, dry_run=True)
    data_dict = train_data.get_statistics()
    np.savez('af_train_data_statistics.npz', **data_dict)























