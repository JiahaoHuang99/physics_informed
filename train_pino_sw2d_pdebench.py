import os
import yaml
import random
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

import utils.util_metrics
# from utils.util_torch_fdm import gradient_xx_scalar, gradient_yy_scalar, gradient_t
from models import FNO3d_SW2D

from train_utils.losses import LpLoss
from train_utils.datasets import sample_data
from train_utils.utils import save_ckpt, count_params, dict2str

from data_pde.dataset_shallow_water_2d_pdebench_d2 import ShallowWater2DDataset

try:
    import wandb
except ImportError:
    wandb = None


@torch.no_grad()
def eval_sw(model,
            val_loader,
            metrics_list,
            device='cpu'):
    model.eval()

    epoch_metrics_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_dict[metric_name] = []
    for a, u in val_loader:
        a, u = a.to(device), u.to(device)
        out = model(a)

        # remove the first time point
        u = u[:, :, :, 1:, :]
        out = out[:, :, :, 1:, :]

        step_eval_dict = utils.util_metrics.eval_sw2d(out, u, metrics_list)
        for metric_name in metrics_list:
            epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

    epoch_metrics_ave_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_ave_dict[metric_name] = np.mean(epoch_metrics_dict[metric_name])

    return epoch_metrics_dict, epoch_metrics_ave_dict


def train_sw(model,
             train_u_loader,        # training data
             train_a_loader,        # initial conditions
             val_loader,            # validation data
             optimizer, 
             scheduler,
             device, config, args):

    save_step = config['train']['save_step']
    eval_step = config['train']['eval_step']

    ic_weight = config['train']['ic_loss']
    f_weight = config['train']['f_loss']
    xy_weight = config['train']['xy_loss']

    # set up directory
    base_dir = os.path.join('exp', config['log']['logdir'])
    ckpt_dir = os.path.join(base_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    # loss fn
    lploss = LpLoss(size_average=True)

    # set up wandb
    if wandb and args.log:
        os.environ['WANDB_MODE'] = config['log']['wandb_mode']
        run = wandb.init(project=config['log']['project'],
                         entity=config['log']['entity'],
                         config=config,
                         reinit=True,
                         name=config['log']['name'])

    pbar = range(config['train']['num_iter'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)

    u_loader = sample_data(train_u_loader)
    a_loader = sample_data(train_a_loader)

    for e in pbar:
        log_dict = {}

        optimizer.zero_grad()
        # data loss
        if xy_weight > 0:
            a, u = next(u_loader)
            a = a.to(device)
            u = u.to(device)
            out = model(a)
            data_loss = lploss(out, u)
        else:
            data_loss = torch.zeros(1, device=device)

        # pde loss
        if f_weight > 0:
            a, u = next(a_loader)
            a = a.to(device)
            u = u.to(device)
            out = model(a)
            ic_loss, f_loss = PINO_SW2D_loss(out)

        else:
            f_loss = torch.zeros(1, device=device)
            ic_loss = torch.zeros(1, device=device)

        loss = data_loss * xy_weight + f_loss * f_weight + ic_loss * ic_weight

        loss.backward()
        optimizer.step()
        scheduler.step()

        log_dict['train loss'] = loss.item()
        log_dict['data'] = data_loss.item()
        log_dict['ic'] = ic_loss.item()
        log_dict['pde'] = f_loss.item()

        if e % eval_step == 0:
            epoch_metrics_dict, epoch_metrics_ave_dict = eval_sw(model, val_loader, config['data']['val']['metrics_list'], device)
            for metric_name in config['data']['val']['metrics_list']:
                log_dict[f'VAL METRICS/{metric_name}'] = epoch_metrics_ave_dict[metric_name]

        logstr = dict2str(log_dict)
        pbar.set_description((logstr))

        if wandb and args.log:
            wandb.log(log_dict)
        if e % save_step == 0 and e > 0:
            ckpt_path = os.path.join(ckpt_dir, f'model-{e}.pt')
            save_ckpt(ckpt_path, model, optimizer, scheduler)

    # clean up wandb
    if wandb and args.log:
        run.finish()


def PINO_SW2D_loss(uv, space_range=2, time_range=5):
    raise NotImplementedError
    device = uv.device
    batchsize, nx, ny, nt, _ = uv.shape
    assert _ == 2

    dx = space_range / nx
    dy = space_range / ny
    dt = time_range / (nt - 1)

    u = uv[..., :1]
    v = uv[..., 1:]

    lploss = LpLoss(size_average=True)

    # pde term
    k = 5e-3
    Du = 1e-3
    Dv = 5e-3

    Ru = u - u ** 3 - k - v
    Rv = u - v

    Res_u = FDM_DR_2D(u=u, D=Du, dx=dx, dy=dy, dt=dt,)
    Res_v = FDM_DR_2D(u=v, D=Dv, dx=dx, dy=dy, dt=dt,)

    loss_ic = torch.zeros(1, device=device)

    loss_f = lploss.rel(Res_u, Ru) + lploss.rel(Res_v, Rv)

    return loss_ic, loss_f


def FDM_SW_2D(u, D, dx, dy, dt,):
    raise NotImplementedError
    u_t = gradient_t(u, dt=dt)
    gradxx_u = gradient_xx_scalar(u, dx=dx)
    gradyy_u = gradient_yy_scalar(u, dy=dy)

    res = u_t - D * gradxx_u - D * gradyy_u

    return res


def subprocess(args):
    with open(args.config, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # set random seed
    config['seed'] = args.seed
    seed = args.seed
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # create model 
    model = FNO3d_SW2D(modes1=config['model']['modes1'],
                       modes2=config['model']['modes2'],
                       modes3=config['model']['modes3'],
                       fc_dim=config['model']['fc_dim'],
                       layers=config['model']['layers'],
                       act=config['model']['act'],
                       in_dim=config['model']['in_dim'],
                       out_dim=config['model']['out_dim'],
                       pad_ratio=config['model']['pad_ratio']).to(device)

    num_params = count_params(model)
    config['num_params'] = num_params
    print(f'Number of parameters: {num_params}')
    # Load from checkpoint
    if args.ckpt:
        ckpt_path = args.ckpt
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt['model'])
        print('Weights loaded from %s' % ckpt_path)
    
    if args.test:
        batchsize = config['test']['batchsize']
        testset = ShallowWater2DDataset(dataset_params=config['data'], split='test')
        testloader = DataLoader(testset, batch_size=batchsize, num_workers=4)
        criterion = LpLoss()
        test_err, std_err = eval_sw(model, testloader, criterion, device)
        print(f'Averaged test relative L2 error: {test_err}; Standard error: {std_err}')
    else:
        # training set
        batchsize = config['train']['batchsize']
        u_set = ShallowWater2DDataset(dataset_params=config['data'], split='train')
        u_loader = DataLoader(u_set, batch_size=batchsize, num_workers=4, shuffle=True)

        a_set = ShallowWater2DDataset(dataset_params=config['data'], split='train')
        a_loader = DataLoader(a_set, batch_size=batchsize, num_workers=4, shuffle=True)

        # val set
        valset = ShallowWater2DDataset(dataset_params=config['data'], split='val')
        val_loader = DataLoader(valset, batch_size=batchsize, num_workers=4)

        print(f'Train set: {len(u_set)}; Test set: {len(valset)}.')
        optimizer = Adam(model.parameters(), lr=config['train']['base_lr'])
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
                                                         milestones=config['train']['milestones'], 
                                                         gamma=config['train']['scheduler_gamma'])
        if args.ckpt:
            ckpt = torch.load(ckpt_path)
            optimizer.load_state_dict(ckpt['optim'])
            scheduler.load_state_dict(ckpt['scheduler'])

        train_sw(model,
                 u_loader,
                 a_loader,
                 val_loader, 
                 optimizer,
                 scheduler,
                 device, 
                 config, args)

    print('Done!')
        
        

if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    # parse options
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config', type=str, default='configs/fno/FNO-SW2D-PDEBench-debug.yaml', help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test', action='store_true', help='Test')
    parser.add_argument('--device', default='cuda:0')

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    subprocess(args)