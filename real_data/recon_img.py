import numpy as np
import warnings
import argparse

warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom
import time
import cv2

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image
import torch.nn as nn
import torch.nn.init as init
import matplotlib.pyplot as plt

from vae.beta_vae import kaiming_init, View, reparametrize
from vae.utils import cuda, grid2gif
from vae.dataset import return_data, return_data1, return_data2, return_data_test

from au_ import LR
from resnet_vae import my_VAE, VAE
from vae.utils import str2bool
from solver import Solver

class DataGather(object):
    def __init__(self):
        self.data = self.get_empty_data_dict()

    def get_empty_data_dict(self):
        return dict(iter=[],
                    recon_loss=[],
                    total_kld=[],
                    dim_wise_kld=[],
                    mean_kld=[],
                    mu=[],
                    var=[],
                    images=[],)

    def insert(self, **kwargs):
        for key in kwargs:
            self.data[key].append(kwargs[key])

    def flush(self):
        self.data = self.get_empty_data_dict()


def viz_reconstruction(x, recon_x, viz, args):
    gather = DataGather()
    gather.insert(images=x.data)
    gather.insert(images=F.sigmoid(recon_x).data)
    x = gather.data['images'][0][:args.batch_size]
    x = make_grid(x, normalize=True)
    x_recon = gather.data['images'][1][:args.batch_size]
    x_recon = make_grid(x_recon, normalize=True)
    images = torch.stack([x, x_recon], dim=0).cpu()
    viz.images(images, env='img_reconstruction',
                    opts=dict(title=str(1)), nrow=10)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colored MNIST')
    parser.add_argument('--hidden_dim', type=int, default=390)
    parser.add_argument('--steps', type=int, default=30001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--path', default='results/', help='The path results to be saved.')
    parser.add_argument('--method', default='Rex', help='ERM, IRM, CoCO')
    parser.add_argument('--grayscale_model', default=False)
    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--model_type', default='VAE', help='VAE, AE, Linear_vae, Linear_ae')
    parser.add_argument('--n_env', type=int, default=2)

    # beta_vae
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=50000, type=float, help='maximum training iteration')
    parser.add_argument('--classify_max_iter', default=500, type=float, help='maximum training iteration for classify')
    parser.add_argument('--batch_size', default=48, type=int, help='batch size')
    parser.add_argument('--env_num', default=2, type=int, help='the number of envs')

    parser.add_argument('--z_dim', default=20, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=2, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--objective', default='resnet', type=str,
                        help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='resnet_vae', type=str,
                        help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=1000, type=float,
                        help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='G:/data/new_NICO/all_train',
                        type=str, help='dataset directory')
    parser.add_argument('--dset_dir1', default='G:/data/new_NICO/in forest', type=str, help='dataset directory')
    parser.add_argument('--dset_dir2', default='G:/data/new_NICO/in water', type=str, help='dataset directory')
    parser.add_argument('--dset_dir_test', default='G:/data/new_NICO/on snow', type=str, help='dataset directory')

    parser.add_argument('--dataset', default='colored_mnist', type=str, help='dataset name')
    parser.add_argument('--image_size', default=224, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=1, type=int, help='dataloader num_workers')
    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=100, type=int,
                        help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=1000, type=int,
                        help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=2000, type=int,
                        help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints_resnet_vae_realdata_size224', type=str,
                        help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--classifier_ckpt_dir', default='classifier_checkpoints', type=str,
                        help='checkpoint directory')
    parser.add_argument('--classifier_ckpt_dir_realdata', default='classifier_checkpoints', type=str,
                        help='checkpoint directory')
    args = parser.parse_args()

    viz = visdom.Visdom()

    for x, _ in return_data(args):
        x = Variable(cuda(x, True))
        recon_model = VAE(20, 3).cuda()
        # 检验z的每个维度
        # recon_model = cuda(recon_model(z_dim=10, nc=3, masked_dim=0, masked_value=i), True)
        # recon_model = cuda(recon_model(z_dim=10, nc=3), True)
        net = Solver(args)

        # 检验重建效果
        # recon_model = cuda(recon_model(z_dim=10, nc=3), True)
        checkpoints = torch.load('checkpoints_resnet_vae_realdata_size224/main/last')
        recon_model.load_state_dict(checkpoints['model_states']['net'])
        recon_x, _, _ = recon_model(x, )
        viz_reconstruction(x, recon_x, viz, args)
