import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from einops import rearrange, repeat
from utils.util_mesh import CustomCoodGenerator, RandomCustomCoodGenerator
from math import ceil

def get_grid3d(S, T, time_scale=1.0, device='cpu'):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt


class ShallowWater2DDataset(Dataset):
    """
    Dataset: Shallow Water 2D Dataset
    Source: https://doi.org/10.18419/darus-2986
    This is a folder dataset (preprocessed).
    Folder:
    - Files (*.npz):
        - Sample (x1)

    """
    def __init__(self, dataset_params, split):
        # load configuration
        self.split = split  # train, val, test
        dataset_params = dataset_params[split]
        self.dataset_params = dataset_params

        # path to dataset folder
        data_folder_path = dataset_params['dataset_path']

        # dataset number and split configuration
        self.n_all_samples = dataset_params['n_all_samples']

        # resolution and domain range configuration
        self.reduced_resolution = dataset_params['reduced_resolution']
        self.reduced_resolution_t = dataset_params['reduced_resolution_t']

        # task specific parameters
        self.in_seq = dataset_params['in_seq']
        self.out_seq = dataset_params['out_seq']

        # load data list
        self.data_paths_all = sorted(glob(os.path.join(data_folder_path, '*.npz')))
        if self.n_all_samples != len(self.data_paths_all):
            print("Warning: n_all_samples is not equal to the number of files in the folder")
        self.data_paths_all = self.data_paths_all[:self.n_all_samples]

        # dataset has been split into train, test (folder)
        self.n_samples = self.n_all_samples
        self.data_paths = self.data_paths_all

        # load an example
        with np.load(self.data_paths[0]) as f:
            data = torch.from_numpy(f['data'][()])  # (res_t, res_x, res_y, 2)
            u = data[:, :, :, 0:1]

            grid_t = torch.from_numpy(f['grid_t'][()])  # res_t = 101
            grid_x = torch.from_numpy(f['grid_x'][()])  # res_x = 128
            grid_y = torch.from_numpy(f['grid_y'][()])  # res_y = 128

        data_res_t, data_res_x, data_res_y, _ = u.shape  # (res_t, res_x, res_y, 1)

        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution
        self.data_res_t = data_res_t
        self.res_time = ceil(self.data_res_t / self.reduced_resolution_t)

        # mesh grid
        grid_x_sampled = grid_x[::self.reduced_resolution]
        grid_y_sampled = grid_y[::self.reduced_resolution]
        grid_t_sampled = grid_t[::self.reduced_resolution_t]
        gx, gy = torch.meshgrid(grid_x_sampled, grid_y_sampled)
        self.S = self.res_grid
        self.mesh = torch.stack((gy, gx), dim=-1)
        self.grid = self.mesh.reshape(-1, 2)

        g_x = self.mesh[:, :, 0].unsqueeze(-1).repeat([1, 1, self.res_time]).unsqueeze(-1)
        g_y = self.mesh[:, :, 1].unsqueeze(-1).repeat([1, 1, self.res_time]).unsqueeze(-1)
        g_t = grid_t_sampled.unsqueeze(0).unsqueeze(0).repeat([self.S, self.S, 1]).unsqueeze(-1)
        self.grid_3d = torch.cat((g_x, g_y, g_t), dim=-1)  # S x S x T x 3

    def _prepare(self):

        # for one sample
        t = self.grid_t
        u = self.u

        # set the value & key position (encoder)
        u_sampled = u.clone()

        u_sampled = rearrange(u_sampled, 't x y a -> x y t a')

        # undersample on time and space
        u_sampled = u_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        t_sampled = t[::self.reduced_resolution_t]

        self.u_sampled = u_sampled


    def _load_sample(self, data_path):

        # load data
        with np.load(data_path) as f:
            self.data = torch.from_numpy(f['data'][()])  # (res_t, res_x, res_y, 1)
            self.u = self.data[:, :, :, :]
            self.grid_t = torch.from_numpy(f['grid_t'][()])  # res_t = 101
            self.grid_x = torch.from_numpy(f['grid_x'][()])  # res_x = 128
            self.grid_y = torch.from_numpy(f['grid_y'][()])  # res_y = 128

        self._prepare()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):

        data_path = self.data_paths[idx]

        self._load_sample(data_path)

        u_sampled_inseq = self.u_sampled[:, :, :self.in_seq, :]  # H, W, Tin, 1

        a = torch.cat((u_sampled_inseq.repeat(1, 1, self.res_time, 1),
                       self.grid_3d),
                      dim=-1)  # (H, W, T, 4)

        u = self.u_sampled  # (H, W, T, 1)

        return a, u


class ShallowWater2DDatasetBaseline(Dataset):
    """
    Dataset: Shallow Water 2D Dataset
    Source: https://doi.org/10.18419/darus-2986
    This is a folder dataset (preprocessed).
    Folder:
    - Files (*.npz):
        - Sample (x1)

    """

    def __init__(self, dataset_params, split):
        # load configuration
        self.split = split  # train, val, test
        dataset_params = dataset_params[split]
        self.dataset_params = dataset_params

        # path to dataset folder
        data_folder_path = dataset_params['dataset_path']

        # dataset number and split configuration
        self.n_all_samples = dataset_params['n_all_samples']

        # resolution and domain range configuration
        self.reduced_resolution = dataset_params['reduced_resolution']
        self.reduced_resolution_t = dataset_params['reduced_resolution_t']

        # task specific parameters
        self.in_seq = dataset_params['in_seq']
        self.out_seq = dataset_params['out_seq']

        # load data list
        self.data_paths_all = sorted(glob(os.path.join(data_folder_path, '*.npz')))
        if self.n_all_samples != len(self.data_paths_all):
            print("Warning: n_all_samples is not equal to the number of files in the folder")
        self.data_paths_all = self.data_paths_all[:self.n_all_samples]

        # dataset has been split into train, test (folder)
        self.n_samples = self.n_all_samples
        self.data_paths = self.data_paths_all

        # load an example
        with np.load(self.data_paths[0]) as f:
            data = torch.from_numpy(f['data'][()])  # (res_t, res_x, res_y, 2)
            u = data[:, :, :, 0:1]

            grid_t = torch.from_numpy(f['grid_t'][()])  # res_t = 101
            grid_x = torch.from_numpy(f['grid_x'][()])  # res_x = 128
            grid_y = torch.from_numpy(f['grid_y'][()])  # res_y = 128

        data_res_t, data_res_x, data_res_y, _ = u.shape  # (res_t, res_x, res_y, 1)

        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution
        self.data_res_t = data_res_t
        self.res_time = ceil(self.data_res_t / self.reduced_resolution_t)

        # mesh grid
        grid_x_sampled = grid_x[::self.reduced_resolution]
        grid_y_sampled = grid_y[::self.reduced_resolution]
        grid_t_sampled = grid_t[::self.reduced_resolution_t]
        gx, gy = torch.meshgrid(grid_x_sampled, grid_y_sampled)
        self.S = self.res_grid
        self.mesh = torch.stack((gy, gx), dim=-1)
        self.grid = self.mesh.reshape(-1, 2)

        g_x = self.mesh[:, :, 0].unsqueeze(-1).repeat([1, 1, self.res_time]).unsqueeze(-1)
        g_y = self.mesh[:, :, 1].unsqueeze(-1).repeat([1, 1, self.res_time]).unsqueeze(-1)
        g_t = grid_t_sampled.unsqueeze(0).unsqueeze(0).repeat([self.S, self.S, 1]).unsqueeze(-1)
        self.grid_3d = torch.cat((g_x, g_y, g_t), dim=-1)  # S x S x T x 3

        xaxis = np.ravel(g_x)
        yaxis = np.ravel(g_y)
        taxis = np.ravel(g_t)
        points = np.stack([xaxis, yaxis, taxis], axis=0).T
        self.xyt = torch.tensor(points, dtype=torch.float)  # (S*S*T, 3)

    def _prepare(self):

        # for one sample
        t = self.grid_t
        u = self.u

        # set the value & key position (encoder)
        u_sampled = u.clone()

        u_sampled = rearrange(u_sampled, 't x y a -> x y t a')

        # undersample on time and space
        u_sampled = u_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        t_sampled = t[::self.reduced_resolution_t]

        self.u_sampled = u_sampled

    def _load_sample(self, data_path):

        # load data
        with np.load(data_path) as f:
            self.data = torch.from_numpy(f['data'][()])  # (res_t, res_x, res_y, 1)
            self.u = self.data[:, :, :, :]
            self.grid_t = torch.from_numpy(f['grid_t'][()])  # res_t = 101
            self.grid_x = torch.from_numpy(f['grid_x'][()])  # res_x = 128
            self.grid_y = torch.from_numpy(f['grid_y'][()])  # res_y = 128

        self._prepare()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):

        data_path = self.data_paths[idx]

        self._load_sample(data_path)

        u_sampled_inseq = self.u_sampled[:, :, :self.in_seq, :]  # H, W, Tin, 1

        a = u_sampled_inseq  # (H, W, Tin, 1)

        u = self.u_sampled  # (H, W, T, 1)

        return a, u


