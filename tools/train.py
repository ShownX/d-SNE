"""
Main function to train the d-SNE model
"""

import argparse
from dsne import get_model, update_config


def parse_args():
    parser = argparse.ArgumentParser(description='d-SNE: Domain Adaptation using Stochastic Neighbourhood Embedding')
    parser.add_argument('--cfg', type=str, default='configs/DIGITS/MT-MM.yaml', help='configuration file')
    parser.add_argument('--model', type=str, default='B', help='train source domain')
    parser.add_argument('--no-src', dest='train_src', action='store_false', help='train source domain')
    parser.add_argument('--no-tgt', dest='train_tgt', action='store_false', help='train target domain')
    parser.add_argument('--session', type=str, default='', help='train target domain')
    parser.add_argument('--ckpt', type=str, default='', help='checkpoint path')
    parser.set_defaults(train_src=True, train_tgt=True)
    args = parser.parse_args()

    cfg = update_config(args)

    return cfg


if __name__ == '__main__':
    cfg = parse_args()
    model = get_model(cfg)
    model.train()

