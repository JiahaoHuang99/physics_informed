import yaml
from argparse import ArgumentParser
from baselines.train_dr2d_pdebench import train_deeponet_dr2d_pdebench



if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='configs/deeponet/DeepONet-DR2D-PDEBench-debug.yaml', help='Path to the configuration file')
    parser.add_argument('--mode', type=str, default='train', help='Train or test')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if args.mode == 'train':
        print('Start training DeepONet Cartesian Product')
        train_deeponet_dr2d_pdebench(config, args.device)
    else:
        raise NotImplementedError
    print('Done!')