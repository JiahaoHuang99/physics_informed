import os
import numpy as np
import torch
from torch.utils.data import Dataset
from glob import glob
from einops import rearrange, repeat
from utils.util_mesh import SquareMeshGenerator


def get_grid3d(S, T, time_scale=1.0, device='cpu'):
    gridx = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridx = gridx.reshape(1, S, 1, 1, 1).repeat([1, 1, S, T, 1])
    gridy = torch.tensor(np.linspace(0, 1, S + 1)[:-1], dtype=torch.float, device=device)
    gridy = gridy.reshape(1, 1, S, 1, 1).repeat([1, S, 1, T, 1])
    gridt = torch.tensor(np.linspace(0, 1 * time_scale, T), dtype=torch.float, device=device)
    gridt = gridt.reshape(1, 1, 1, T, 1).repeat([1, S, S, 1, 1])
    return gridx, gridy, gridt


class IncompressibleNavierStokes2DDataset(Dataset):
    """
    Dataset: Incompressible Navier Stokes2D Dataset
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

        self.real_space_range = dataset_params['real_space_range']  # [0, 1]
        self.real_space = [[self.real_space_range[0], self.real_space_range[1]],
                           [self.real_space_range[0], self.real_space_range[1]]]
        self.real_time_range = dataset_params['real_time_range']  # [0, 5]

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
        with np.load(self.data_paths[0]) as data:
            velocity = torch.from_numpy(data['velocity'].astype(np.float32))  # (res_t, res_x, res_y, 2)
            t = torch.from_numpy(data['t'].astype(np.float32))

        data_res_t, data_res_x, data_res_y, _ = velocity.shape  # (res_t, res_x, res_y, 2)

        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution
        self.data_res_t = data_res_t
        self.res_time = self.data_res_t // self.reduced_resolution_t

        self.meshgenerator = SquareMeshGenerator(real_space=self.real_space,
                                                 mesh_size=self.mesh_size,
                                                 downsample_rate=self.reduced_resolution,
                                                 is_diag=False)

        self.grid = self.meshgenerator.get_grid()  # (n_sample, 2)
        self.mesh = rearrange(self.grid, '(x y) c -> x y c', x=self.res_grid, y=self.res_grid)
        self.S = self.res_grid

        grid_x = self.mesh[:, :, 0].unsqueeze(-1).repeat([1, 1, self.res_time]).unsqueeze(-1)
        grid_y = self.mesh[:, :, 1].unsqueeze(-1).repeat([1, 1, self.res_time]).unsqueeze(-1)
        grid_t = t[::self.reduced_resolution_t].unsqueeze(0).unsqueeze(0).repeat([self.S, self.S, 1]).unsqueeze(-1)
        self.grid_3d = torch.cat((grid_x, grid_y, grid_t), dim=-1)  # S x S x T x 3

    def _prepare(self):

        # for one sample
        force = self.force
        particles = self.particles
        velocity = self.velocity
        t = self.t

        # set the value & key position (encoder)
        force_sampled = force.clone()
        particles_sampled = particles.clone()
        velocity_sampled = velocity.clone()

        force_sampled = rearrange(force_sampled, 'x y a -> x y a')
        particles_sampled = rearrange(particles_sampled, 't x y a -> x y t a')
        velocity_sampled = rearrange(velocity_sampled, 't x y a -> x y t a')

        # undersample on time and space
        force_sampled = force_sampled[::self.reduced_resolution, ::self.reduced_resolution, :]
        particles_sampled = particles_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        velocity_sampled = velocity_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        t_sampled = t[::self.reduced_resolution_t]

        self.force_sampled = force_sampled
        self.particles_sampled = particles_sampled
        self.velocity_sampled = velocity_sampled


    def _load_sample(self, data_path):

        # load data
        # Keys: ['force', 'particles', 't', 'velocity']
        with np.load(data_path) as f:
            self.force = torch.from_numpy(f['force'])  # (res_x, res_y, 2)
            self.particles = torch.from_numpy(f['particles'].astype(np.float32))  # (res_t, res_x, res_y, 1)
            self.velocity = torch.from_numpy(f['velocity'].astype(np.float32))  # (res_t, res_x, res_y, 2)
            self.t = torch.from_numpy(f['t'].astype(np.float32))

        data_res_t, data_res_x, data_res_y, _ = self.velocity.shape  # (res_t, res_x, res_y, 2)
        assert data_res_x == data_res_y
        assert _ == 2
        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution
        self.data_res_t = data_res_t
        self.res_time = self.data_res_t // self.reduced_resolution_t
        assert self.in_seq + self.out_seq == self.res_time

        self._prepare()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):

        data_path = self.data_paths[idx]

        self._load_sample(data_path)

        velocity_sampled_inseq = self.velocity_sampled[:, :, :self.in_seq, :]  # H, W, Tin, 2
        particles_sampled_inseq = self.particles_sampled[:, :, :self.in_seq, :]  # H, W, Tin, 1
        force_sampled = self.force_sampled.unsqueeze(-2)  # H, W, T, 1 (add time channel)

        a = torch.cat((velocity_sampled_inseq.repeat(1, 1, self.res_time, 1),
                       particles_sampled_inseq.repeat(1, 1, self.res_time, 1),
                       force_sampled.repeat(1, 1, self.res_time, 1),
                       self.grid_3d),
                      dim=-1)  # (H, W, T, 8)

        u = self.velocity_sampled

        return a, u


class IncompressibleNavierStokes2DDataseta(Dataset):
    """
    Dataset: Incompressible Navier Stokes2D Dataset
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
        self.reduced_resolution = dataset_params['reduced_resolution_pde'] if 'reduced_resolution_pde' in dataset_params else dataset_params['reduced_resolution']
        self.reduced_resolution_t = dataset_params['reduced_resolution_t']

        self.real_space_range = dataset_params['real_space_range']  # [0, 1]
        self.real_space = [[self.real_space_range[0], self.real_space_range[1]],
                           [self.real_space_range[0], self.real_space_range[1]]]
        self.real_time_range = dataset_params['real_time_range']  # [0, 5]

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
        with np.load(self.data_paths[0]) as data:
            velocity = torch.from_numpy(data['velocity'].astype(np.float32))  # (res_t, res_x, res_y, 2)
            t = torch.from_numpy(data['t'].astype(np.float32))

        data_res_t, data_res_x, data_res_y, _ = velocity.shape  # (res_t, res_x, res_y, 2)

        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution
        self.data_res_t = data_res_t
        self.res_time = self.data_res_t // self.reduced_resolution_t

        self.meshgenerator = SquareMeshGenerator(real_space=self.real_space,
                                                 mesh_size=self.mesh_size,
                                                 downsample_rate=self.reduced_resolution,
                                                 is_diag=False)

        self.grid = self.meshgenerator.get_grid()  # (n_sample, 2)
        self.mesh = rearrange(self.grid, '(x y) c -> x y c', x=self.res_grid, y=self.res_grid)
        self.S = self.res_grid

        grid_x = self.mesh[:, :, 0].unsqueeze(-1).repeat([1, 1, self.res_time]).unsqueeze(-1)
        grid_y = self.mesh[:, :, 1].unsqueeze(-1).repeat([1, 1, self.res_time]).unsqueeze(-1)
        grid_t = t[::self.reduced_resolution_t].unsqueeze(0).unsqueeze(0).repeat([self.S, self.S, 1]).unsqueeze(-1)
        self.grid_3d = torch.cat((grid_x, grid_y, grid_t), dim=-1)  # S x S x T x 3

    def _prepare(self):

        # for one sample
        force = self.force
        particles = self.particles
        velocity = self.velocity
        t = self.t

        # set the value & key position (encoder)
        force_sampled = force.clone()
        particles_sampled = particles.clone()
        velocity_sampled = velocity.clone()

        force_sampled = rearrange(force_sampled, 'x y a -> x y a')
        particles_sampled = rearrange(particles_sampled, 't x y a -> x y t a')
        velocity_sampled = rearrange(velocity_sampled, 't x y a -> x y t a')

        # undersample on time and space
        force_sampled = force_sampled[::self.reduced_resolution, ::self.reduced_resolution, :]
        particles_sampled = particles_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        velocity_sampled = velocity_sampled[::self.reduced_resolution, ::self.reduced_resolution, ::self.reduced_resolution_t, :]
        t_sampled = t[::self.reduced_resolution_t]

        self.force_sampled = force_sampled
        self.particles_sampled = particles_sampled
        self.velocity_sampled = velocity_sampled


    def _load_sample(self, data_path):

        # load data
        # Keys: ['force', 'particles', 't', 'velocity']
        with np.load(data_path) as f:
            self.force = torch.from_numpy(f['force'])  # (res_x, res_y, 2)
            self.particles = torch.from_numpy(f['particles'].astype(np.float32))  # (res_t, res_x, res_y, 1)
            self.velocity = torch.from_numpy(f['velocity'].astype(np.float32))  # (res_t, res_x, res_y, 2)
            self.t = torch.from_numpy(f['t'].astype(np.float32))

        data_res_t, data_res_x, data_res_y, _ = self.velocity.shape  # (res_t, res_x, res_y, 2)
        assert data_res_x == data_res_y
        assert _ == 2
        self.res_full = data_res_x
        self.mesh_size = [self.res_full, self.res_full]
        self.res_grid = self.res_full // self.reduced_resolution
        self.data_res_t = data_res_t
        self.res_time = self.data_res_t // self.reduced_resolution_t
        assert self.in_seq + self.out_seq == self.res_time

        self._prepare()

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return self.n_samples

    def __getitem__(self, idx):

        data_path = self.data_paths[idx]

        self._load_sample(data_path)

        velocity_sampled_inseq = self.velocity_sampled[:, :, :self.in_seq, :]  # H, W, Tin, 2
        particles_sampled_inseq = self.particles_sampled[:, :, :self.in_seq, :]  # H, W, Tin, 1
        force_sampled = self.force_sampled.unsqueeze(-2)  # H, W, T, 1 (add time channel)

        a = torch.cat((velocity_sampled_inseq.repeat(1, 1, self.res_time, 1),
                       particles_sampled_inseq.repeat(1, 1, self.res_time, 1),
                       force_sampled.repeat(1, 1, self.res_time, 1),
                       self.grid_3d),
                      dim=-1)  # (H, W, T, 8)

        u0 = velocity_sampled_inseq

        return a, u0, force_sampled, self.particles_sampled

