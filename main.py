import os
import torch
import argparse
from torch.backends import cudnn
from models.LAIBNet import build_LAIBNet
from train import _train

def main(args):
    # CUDNN
    cudnn.benchmark = True
    model = build_LAIBNet(16)
    if torch.cuda.is_available():
        model.cuda()
        _train(model, args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Directories
    parser.add_argument('--data_dir', type=str, default='./dataset')
    parser.add_argument('--mode', default='train', choices=['train'], type=str)
    # Train
    parser.add_argument('--resume', type=str, default='')#
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--learning_rate', type=float, default=0.0005)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--num_epoch', type=int, default=320)
    parser.add_argument('--num_worker', type=int, default=6)
    parser.add_argument('--gamma', type=float, default=0.8)
    parser.add_argument('--lr_steps', type=list, default=[(x+1) * 40 for x in range(320//40)]) 
    #model  Directories
    parser.add_argument('--model_save_dir', type=str, default='./weights')
    args = parser.parse_args()
    main(args)
