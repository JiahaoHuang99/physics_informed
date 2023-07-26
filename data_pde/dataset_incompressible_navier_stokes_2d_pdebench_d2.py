import os
import dgl
import time
import numpy as np
import sklearn
import hashlib
import networkx as nx
from math import ceil
from scipy import sparse as sp
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from glob import glob
from einops import rearrange, repeat

from utils.util_mesh import RandomMeshGenerator, SquareMeshGenerator


class IncompressibleNavierStokes2DDataset(Dataset):
    """
    Dataset: Incompressible Navier Stokes2D Dataset
    Source: https://doi.org/10.18419/darus-2986
    This is a folder dataset (preprocessed).
    Folder:
    - Files (*.npz):
        - Sample (x1):

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

        # FIXME: CHECK
        return torch.cat([self.a_sampled.unsqueeze(2), self.mesh], dim=2), self.u_sampled
