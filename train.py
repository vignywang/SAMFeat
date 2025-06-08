# 
# Created  on 2020/08/25
#
import os
import yaml
from pathlib import Path

import argparse
import torch
import numpy as np

from utils.logger import get_logger
from trainers import get_trainer


def setup_seed():
    # make the result reproducible
    torch.manual_seed(3928)
    torch.cuda.manual_seed_all(2342)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(2933)


def write_config(logger, prefix, config):
    for k, v in config.items():
        if isinstance(v, dict):
            logger.info('{}: '.format(k))
            write_config(logger, prefix+' '*4, v)
        else:
            logger.info('{}{}: {}'.format(prefix, k, v))


def main():
    setup_seed()
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', type=str, default='0')
    parser.add_argument('--configs', type=str, required=True)
    parser.add_argument('--indicator', type=str, required=True)
    args = parser.parse_args()

    # read configs
    with open(args.configs, 'r') as f:
        config = yaml.load(f)

    # initialize ckpt_path
    ckpt_path = Path('ckpt', config['name']+'_'+args.indicator)
    ckpt_path.mkdir(parents=True, exist_ok=True)
    config['ckpt_path'] = str(ckpt_path)

    # initialize logger
    log_path = Path('log', config['name']+'_'+args.indicator)
    log_path.mkdir(parents=True, exist_ok=True)
    config['logger'] = get_logger(str(log_path))

    # write config
    write_config(config['logger'], '', config)

    # set gpu devices
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    args.gpus = [i for i in range(len(args.gpus.split(',')))]
    config['logger'].info("Set CUDA_VISIBLE_DEVICES to %s" % args.gpus)

    # initialize trainer and train
    with get_trainer(config['trainer'])(**config) as trainer:
        trainer.train()


if __name__ == '__main__':
    main()
