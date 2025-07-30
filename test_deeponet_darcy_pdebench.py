import yaml
from argparse import ArgumentParser
# from baselines.test_darcy import test_deeponet_darcy
from baselines.test_darcy_pdebench import test_deeponet_darcy_pdebench


if __name__ == '__main__':
    parser = ArgumentParser(description='Basic paser')
    parser.add_argument('--config_path', type=str, default='configs/deeponet/DeepONet-DarcyFlow-PDEBench-debug.yaml', help='Path to the configuration file')
    parser.add_argument('--mode', type=str, default='train', help='Train or test')
    parser.add_argument('--device', default='cuda:3')
    args = parser.parse_args()

    config_file = args.config_path
    with open(config_file, 'r') as stream:
        config = yaml.load(stream, yaml.FullLoader)

    if args.mode == 'train':
        print('Start training DeepONet Cartesian Product')
        if 'name' in config['data'] and config['data']['name'] == 'Darcy':
            assert NotImplementedError
            # test_deeponet_darcy(config)
        elif 'name' in config['data'] and config['data']['name'] == 'DarcyFlow-PDEBench':
            test_deeponet_darcy_pdebench(config, device=args.device)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    print('Done!')