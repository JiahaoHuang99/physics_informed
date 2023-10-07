from tqdm import tqdm
import wandb
import os
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from baselines.model import DeepONetNS
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

        x = x.to(device)  # initial condition, (batchsize, S, S, 1, 2)
        y = y.to(device)  # ground truth, (batchsize, S, S, T, 2)

        grid = grid.to(device)  # grid value, (S*S*T, 3)

        x = x.reshape(bs, -1, 2)  # (batchsize, S*S, 2)

        pred = model(x, grid)  # (batchsize, S*S*T, 2)
        pred = pred.reshape(bs, S, S, T, 2)  # (batchsize, S, S, T, 2)

        # remove the first time point
        pred = pred[:, :, :, 1:, :]
        y = y[:, :, :, 1:, :]

        step_eval_dict = utils.util_metrics.eval_dr2d(pred, y, metrics_list)
        for metric_name in metrics_list:
            epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

    epoch_metrics_ave_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_ave_dict[metric_name] = np.mean(epoch_metrics_dict[metric_name])

    return epoch_metrics_dict, epoch_metrics_ave_dict


def train_deeponet_dr2d_pdebench(config, device='cuda:0'):
    '''
    Train DeepONet for Incompressible Navier Stokes 2D (PDEBench)
    '''
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_config = config['data']
    batch_size = config['train']['batchsize']
    eval_epoch = config['train']['eval_epoch']
    save_epoch = config['train']['save_epoch']

    dataset = DiffusionReaction2DDatasetBaseline(dataset_params=data_config, split='train')
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            drop_last=True,
                            shuffle=True)

    val_dataset = DiffusionReaction2DDatasetBaseline(dataset_params=data_config, split='val')
    val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                drop_last=False,
                                shuffle=False)

    u0_dim = dataset.S ** 2
    model = DeepONetNS(branch_layer=[u0_dim * 2] + config['model']['branch_layers'],
                       trunk_layer=[3] + config['model']['trunk_layers']).to(device)
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
    myloss = LpLoss(size_average=True)
    model.train()

    for e in pbar:
        log_dict = {}
        train_loss = 0.0
        for x, y in dataloader:
            bs, S, _, T, __ = y.shape
            assert S == _
            assert __ == 2
            assert bs == batch_size

            x = x.to(device)  # initial condition, (batchsize, S, S, 1, 2)
            y = y.to(device)  # ground truth, (batchsize, S, S, T, 2)
            grid = dataset.xyt
            grid = grid.to(device)  # grid value, (S*S*T, 3)

            x = x.reshape(bs, -1, 2)  # (batchsize, S*S, 2)

            pred = model(x, grid)  # (batchsize, S*S*T, 2)
            pred = pred.reshape(bs, S, S, T, 2)  # (batchsize, S, S, T, 2)

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
                            model,
                            optimizer,
                            relative_path=False)

        if e % eval_epoch == 0:
            epoch_metrics_dict, epoch_metrics_ave_dict = eval_dr2d_deeponet(model,
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

