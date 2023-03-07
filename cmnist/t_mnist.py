import argparse
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from pathlib import Path
from au_ import make_environment, MyMLP, mean_nll, mean_accuracy, pretty_print
from vae.beta_vae import BetaVAE_B, BetaVAE_H
from vae.solver import Solver, reconstruction_loss, kl_divergence
from torchvision import datasets
from vae.utils import str2bool
from torch import optim
import pickle
from time import time
import cv2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colored MNIST')
    parser.add_argument('--hidden_dim', type=int, default=390)
    parser.add_argument('--steps', type=int, default=30001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.00001)
    parser.add_argument('--path', default='results/', help='The path results to be saved.')
    parser.add_argument('--method', default='Rex', help='ERM, IRM, CoCO')
    parser.add_argument('--grayscale_model', default=False)
    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--model_type', default='VAE', help='VAE, AE, Linear_vae, Linear_ae')


    #IRM
    parser.add_argument('--penalty_anneal_iters', type=int, default=190)
    parser.add_argument('--l2_regularizer_weight', type=float,default=0.00110794568 )
    parser.add_argument('--penalty_weight', type=float, default=91257.18613115903)
    #CoCo
    parser.add_argument('--coco_weight', type=float, default=500)
    #beta_vae
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1000000, type=float, help='maximum training iteration')
    parser.add_argument('--classify_max_iter', default=500, type=float, help='maximum training iteration for classify')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--env_num', default=2, type=int, help='the number of envs')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=2, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--objective', default='H', type=str,
                        help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='Linear_vae', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=1000, type=float,
                        help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='G:/data/mnist/mnist_dataset_color/baseset/img_64',
                        type=str, help='dataset directory')
    parser.add_argument('--dset_dir1', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/env1', type=str, help='dataset directory')
    parser.add_argument('--dset_dir2', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/env2', type=str, help='dataset directory')
    parser.add_argument('--dset_dir_test', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/test', type=str, help='dataset directory')


    parser.add_argument('--dataset', default='colored_mnist', type=str, help='dataset name')
    parser.add_argument('--image_size', default=28, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=0, type=int, help='dataloader num_workers')
    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=100, type=int,
                        help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=10000, type=int,
                        help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=20000, type=int,
                        help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints_linear_vae', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--classifier_ckpt_dir', default='checkpoints_classifier_new/300', type=str, help='checkpoint directory')

    flags = parser.parse_args()

    if flags.model_type == 'VAE':
        net = Solver(flags)
        # net.train()
        net.train_classify()
        net.test_classify()
    elif flags.model_type == 'AE':
        net = Solver(flags)
        net.train_ae()
        net.train_ae_classify()
        net.test_ae()














