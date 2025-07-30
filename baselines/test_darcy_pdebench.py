from tqdm import tqdm
import wandb
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from baselines.model import DeepONetCP
from train_utils.losses import LpLoss
from train_utils.utils import save_checkpoint
# from baselines.data import DarcyFlow
from data_pde.dataset_darcy_flow_pdebench_d2 import DarcyFlowDatasetBaseline as DarcyFlow
import utils.util_metrics
import numpy as np

try:
    import wandb
except ImportError:
    wandb = None


@torch.no_grad()
def eval_darcy_deeponet(model,
                        grid,
                        val_loader,
                        metrics_list,
                        device='cpu'):

    model.eval()

    epoch_metrics_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_dict[metric_name] = []
    for x, y in val_loader:
        x = x.to(device)  # initial condition, (batchsize, u0_dim)
        y = y.to(device)  # ground truth, (batchsize, SxS)
        pred = model(x, grid)
        step_eval_dict = utils.util_metrics.eval_darcy(pred.unsqueeze(-1).unsqueeze(-1), y.unsqueeze(-1).unsqueeze(-1), metrics_list)
        for metric_name in metrics_list:
            epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

    epoch_metrics_ave_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_ave_dict[metric_name] = np.mean(epoch_metrics_dict[metric_name])

    return epoch_metrics_dict, epoch_metrics_ave_dict


def test_deeponet_darcy_pdebench(config, device='cuda:0'):
    '''
    Train DeepONet for Darcy Flow (PDEBench)
    '''
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    batch_size = config['train']['batchsize']

    dataset = DarcyFlow(dataset_params=data_config, split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = DarcyFlow(dataset_params=data_config, split='test')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    u0_dim = dataset.S ** 2
    model = DeepONetCP(branch_layer=[u0_dim] + config['model']['branch_layers'],
                       trunk_layer=[2] + config['model']['trunk_layers']).to(device)

    # load weight
    model_path = config['data']['test']['weight_path']
    if os.path.exists(model_path):
        print(f'Loading model from {model_path}')
        model.load_state_dict(torch.load(model_path, map_location=device)["model"])
    else:
        raise FileNotFoundError(f'Model path {model_path} does not exist.')

    log_dict = {}

    grid = dataset.mesh
    grid = grid.reshape(-1, 2).to(device)  # grid value, (SxS, 2)

    epoch_metrics_dict, epoch_metrics_ave_dict = eval_darcy_deeponet(model,
                                                                     grid,
                                                                     val_dataloader,
                                                                     config['data']['test']['metrics_list'],
                                                                     device=device
                                                                     )

    for metric_name in config['data']['test']['metrics_list']:
        log_dict[f'TEST METRICS/{metric_name}'] = epoch_metrics_ave_dict[metric_name]

        print(f'TEST METRICS/{metric_name}: {epoch_metrics_ave_dict[metric_name]}')


