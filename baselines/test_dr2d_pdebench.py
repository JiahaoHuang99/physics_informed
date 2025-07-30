from tqdm import tqdm
import wandb
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from baselines.model import DeepONetDR
from train_utils.losses import LpLoss
from train_utils.utils import save_checkpoint
from baselines.data import DeepONetCPNS
from data_pde.dataset_diffusion_reaction_2d_pdebench_d2 import DiffusionReaction2DDatasetBaseline
import utils.util_metrics
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


@torch.no_grad()
def eval_dr2d_deeponet(model,
                       grid,
                       val_loader,
                       metrics_list,
                       device='cpu'):

    model.eval()

    epoch_metrics_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_dict[metric_name] = []
    for x, y in val_loader:
        bs, S, _, T, __ = y.shape
        Tin = x.shape[3]
        x = x.to(device)  # initial condition, (batchsize, S, S, Tin, 2)
        y = y.to(device)  # ground truth, (batchsize, S, S, T, 2)

        grid = grid.to(device)  # grid value, (S*S*T, 3)

        x = x.reshape(bs, -1, 2)  # (batchsize, S*S*Tin, 2)

        pred = model(x, grid)  # (batchsize, S*S*T, 2)
        pred = pred.reshape(bs, S, S, T, 2)  # (batchsize, S, S, T, 2)

        pred[:, :, :, :Tin, :] = y[:, :, :, :Tin, :]

        step_eval_dict = utils.util_metrics.eval_dr2d(pred, y, metrics_list)
        for metric_name in metrics_list:
            epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

    epoch_metrics_ave_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_ave_dict[metric_name] = np.mean(epoch_metrics_dict[metric_name])

    return epoch_metrics_dict, epoch_metrics_ave_dict


def test_deeponet_dr2d_pdebench(config, device='cuda:0'):
    '''
    Train DeepONet for Incompressible Navier Stokes 2D (PDEBench)
    '''
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    batch_size = config['train']['batchsize']

    dataset = DiffusionReaction2DDatasetBaseline(dataset_params=data_config, split='train')
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=True,
                            shuffle=True)

    val_dataset = DiffusionReaction2DDatasetBaseline(dataset_params=data_config, split='test')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                drop_last=False,
                                shuffle=False)

    u0_dim = dataset.S ** 2 * dataset.in_seq
    model = DeepONetDR(branch_layer=[u0_dim] + config['model']['branch_layers'],
                       trunk_layer=[3] + config['model']['trunk_layers']).to(device)


    log_dict = {}

    x, y = next(iter(dataloader))
    bs, S, _, T, __ = y.shape
    assert S == _
    assert __ == 2
    assert bs == batch_size

    grid = dataset.xyt
    grid = grid.to(device)  # grid value, (S*S*T, 3)

    epoch_metrics_dict, epoch_metrics_ave_dict = eval_dr2d_deeponet(model,
                                                                    grid,
                                                                    val_dataloader,
                                                                    config['data']['test']['metrics_list'],
                                                                    device=device
                                                                    )

    for metric_name in config['data']['test']['metrics_list']:
        log_dict[f'TEST METRICS/{metric_name}'] = epoch_metrics_ave_dict[metric_name]


