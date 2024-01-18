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


def train_deeponet_darcy_pdebench(config, device='cuda:0'):
    '''
    Train DeepONet for Darcy Flow (PDEBench)
    '''
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    batch_size = config['train']['batchsize']
    save_epoch = config['train']['save_epoch']
    eval_epoch = config['train']['eval_epoch']

    dataset = DarcyFlow(dataset_params=data_config, split='train')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    val_dataset = DarcyFlow(dataset_params=data_config, split='val')
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    u0_dim = dataset.S ** 2
    model = DeepONetCP(branch_layer=[u0_dim] + config['model']['branch_layers'],
                       trunk_layer=[2] + config['model']['trunk_layers']).to(device)
    optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
    scheduler = MultiStepLR(optimizer,
                            milestones=config['train']['milestones'],
                            gamma=config['train']['scheduler_gamma'])

    # set up wandb
    if wandb:
        os.environ['WANDB_MODE'] = config['log']['wandb_mode']
        run = wandb.init(project=config['log']['project'],
                         entity=config['log']['entity'],
                         config=config,
                         reinit=True,
                         )

    pbar = range(config['train']['epochs'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.1)
    # myloss = LpLoss(size_average=True)
    myloss = torch.nn.MSELoss(reduction='mean')
    model.train()
    grid = dataset.mesh
    grid = grid.reshape(-1, 2).to(device)  # grid value, (SxS, 2)
    for e in pbar:
        log_dict = {}
        train_loss = 0.0
        for x, y in dataloader:
            x = x.to(device)  # initial condition, (batchsize, u0_dim)
            y = y.to(device)  # ground truth, (batchsize, SxS)

            pred = model(x, grid)
            loss = myloss(pred, y)

            model.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.shape[0]
        train_loss /= len(dataset)
        log_dict['train_loss'] = train_loss
        lr = scheduler.get_last_lr()[0]
        log_dict['learning_rate'] = lr

        scheduler.step()

        pbar.set_description(
            (
                f'Epoch: {e}; Averaged train loss: {train_loss:.5f}; '
            )
        )
        if e % save_epoch == 0:
            print(f'Epoch: {e}, averaged train loss: {train_loss:.5f}')
            save_checkpoint(config['train']['save_dir'],
                            config['train']['save_name'].replace('.pt', f'_{e}.pt'),
                            model, optimizer)

        if e % eval_epoch == 0:
            epoch_metrics_dict, epoch_metrics_ave_dict = eval_darcy_deeponet(model,
                                                                             grid,
                                                                             val_dataloader,
                                                                             config['data']['val']['metrics_list'],
                                                                             device=device
                                                                             )

            for metric_name in config['data']['val']['metrics_list']:
                log_dict[f'VAL METRICS/{metric_name}'] = epoch_metrics_ave_dict[metric_name]

        if wandb:
            wandb.log(log_dict)

    save_checkpoint(config['train']['save_dir'],
                    config['train']['save_name'],
                    model,
                    optimizer)

    if wandb:
        run.finish()

