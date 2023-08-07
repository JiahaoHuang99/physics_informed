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
from utils.util_torch_fdm import gradient_xy_vector, gradient_xy_scalar, gradient_t, laplacian
from models import FNO3d_incomNS2D

from train_utils.losses import LpLoss
from train_utils.datasets import sample_data
from train_utils.utils import save_ckpt, count_params, dict2str

from data_pde.dataset_incompressible_navier_stokes_2d_pdebench_d2 import IncompressibleNavierStokes2DDataset, IncompressibleNavierStokes2DDataseta

try:
    import wandb
except ImportError:
    wandb = None


@torch.no_grad()
def eval_ns(model,
            val_loader,
            metrics_list,
            device='cpu'):
    model.eval()

    epoch_metrics_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_dict[metric_name] = []
    for a_in, u in val_loader:  # FIXME
        a_in, u = a_in.to(device), u.to(device)
        out = model(a_in).squeeze(dim=-1)

        # remove the first time point
        u = u[:, :, :, 1:, :]
        out = out[:, :, :, 1:, :]

        step_eval_dict = utils.util_metrics.eval_incom_ns2d(out, u, metrics_list)
        for metric_name in metrics_list:
            epoch_metrics_dict[metric_name].append(step_eval_dict[metric_name])

    epoch_metrics_ave_dict = {}
    for metric_name in metrics_list:
        epoch_metrics_ave_dict[metric_name] = np.mean(epoch_metrics_dict[metric_name])

    return epoch_metrics_dict, epoch_metrics_ave_dict


def train_ns(model, 
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
                         )

    pbar = range(config['train']['num_iter'])
    pbar = tqdm(pbar, dynamic_ncols=True, smoothing=0.2)

    u_loader = sample_data(train_u_loader)
    a_loader = sample_data(train_a_loader)

    for e in pbar:
        log_dict = {}

        optimizer.zero_grad()
        # data loss
        if xy_weight > 0:
            a_in, u = next(u_loader)
            a_in = a_in.to(device)
            u = u.to(device)
            out = model(a_in)
            data_loss = lploss(out, u)
        else:
            data_loss = torch.zeros(1, device=device)

        # pde loss
        if f_weight > 0:
            a_in, u0, force, particles = next(a_loader)
            a_in = a_in.to(device)
            u0 = u0.to(device)
            force = force.to(device)
            particles = particles.to(device)
            out = model(a_in)
            ic_loss, f_loss = PINO_IncomNS2D_loss(out, u0, force, particles)

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
            epoch_metrics_dict, epoch_metrics_ave_dict = eval_ns(model, val_loader, config['data']['val']['metrics_list'], device)
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


def PINO_IncomNS2D_loss(u, u0, force, particles, space_range=1, time_range=5):
    batchsize, nx, ny, nt, _ = u.shape
    assert _ == 2

    dx = space_range / nx
    dy = space_range / ny
    dt = time_range / nt

    lploss = LpLoss(size_average=True)

    u0_pred = u[:, :, :, 0:1, :]
    loss_ic = lploss(u0_pred, u0)

    Du = FDM_NS_2D(v=u, p=particles, dx=dx, dy=dy, dt=dt)
    force = force.repeat(1, 1, 1, nt, 1)

    loss_f = lploss.rel(Du, force)

    return loss_ic, loss_f


def FDM_NS_2D(v, p, dx, dy, dt, rho=0.01, eta=0.01):

    v_t = gradient_t(v, dt=dt)
    grad_v = gradient_xy_vector(v, dx=dx, dy=dy)
    v_grad_v = torch.einsum('bxyti,bxyti->bxyt', v, grad_v).unsqueeze(-1)

    grad_p = gradient_xy_scalar(p, dx=dx, dy=dy)
    lap_v = laplacian(v, dx=dx, dy=dy).unsqueeze(-1)

    Du = rho * v_t + rho * v_grad_v + grad_p - eta * lap_v

    return Du

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
    model = FNO3d_incomNS2D(modes1=config['model']['modes1'],
                            modes2=config['model']['modes2'],
                            modes3=config['model']['modes3'],
                            fc_dim=config['model']['fc_dim'],
                            layers=config['model']['layers'],
                            act=config['model']['act'],
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
        testset = IncompressibleNavierStokes2DDataset(dataset_params=config['data'], split='test')
        testloader = DataLoader(testset, batch_size=batchsize, num_workers=4)
        criterion = LpLoss()
        test_err, std_err = eval_ns(model, testloader, criterion, device)
        print(f'Averaged test relative L2 error: {test_err}; Standard error: {std_err}')
    else:
        # training set
        batchsize = config['train']['batchsize']
        u_set = IncompressibleNavierStokes2DDataset(dataset_params=config['data'], split='train')
        u_loader = DataLoader(u_set, batch_size=batchsize, num_workers=4, shuffle=True)

        a_set = IncompressibleNavierStokes2DDataseta(dataset_params=config['data'], split='train')
        a_loader = DataLoader(a_set, batch_size=batchsize, num_workers=4, shuffle=True)

        # val set
        valset = IncompressibleNavierStokes2DDataset(dataset_params=config['data'], split='val')
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

        train_ns(model, 
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
    parser.add_argument('--config', type=str, default='configs/pino/PINO-IncomNS2D-PDEPbence-debug.yaml', help='Path to the configuration file')
    parser.add_argument('--log', action='store_true', help='Turn on the wandb')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--test', action='store_true', help='Test')
    parser.add_argument('--device', default='cuda:3')

    args = parser.parse_args()
    if args.seed is None:
        args.seed = random.randint(0, 100000)
    subprocess(args)