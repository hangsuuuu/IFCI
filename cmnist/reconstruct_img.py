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


class AutoEncoder(nn.Module):
    def __init__(self, z_dim=10, nc=3, masked_dim=0, masked_value=0):
        super(AutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 7, 7
            nn.ReLU(True),
            View((-1, 32 * 7 * 7)),  # B, 1568
            nn.Linear(32 * 7 * 7, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim),  # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32 * 7 * 7),  # B, 1568
            nn.ReLU(True),
            View((-1, 32, 7, 7)),  # B,  32,  7,  7
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 28, 28
        )
        # self.masked_dim = masked_dim
        # self.masked_value = masked_value

    def forward(self, x):
        z = self._encode(x)
        # z[:, self.masked_dim] = self.masked_value
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

class BetaVAE_H(nn.Module):
    """Model proposed in original beta-VAE paper(Higgins et al, ICLR, 2017)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_H, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 14, 14
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 7, 7
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 3, 2, 1),  # B,  64, 4, 4
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 256, 4, 1),  # B,  256, 1, 1
        #     nn.ReLU(True),
        #     View((-1, 256 * 1 * 1)),  # B, 256
        #     nn.Linear(256 * 1 * 1, z_dim * 2),  # B, z_dim*2
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim, 256),  # B, 256
        #     View((-1, 256, 1, 1)),  # B, 256,  1,  1
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 64, 4, 1),  # B,  64, 4, 4
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 3, 2, 1),  # B,  32, 7, 7
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 28, 28
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256, z_dim * 2),  # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        )

        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        # z = mu
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)

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
    parser.add_argument('--dset_dir', default='G:/data/mnist/mnist_dataset_color/baseset/img_64',
                        type=str, help='dataset directory')
    parser.add_argument('--dset_dir1', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/env1', type=str,
                        help='dataset directory')
    parser.add_argument('--dset_dir2', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/env2', type=str,
                        help='dataset directory')
    parser.add_argument('--dset_dir_test', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/test', type=str,
                        help='dataset directory')

    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--dataset', default='colored_mnist', type=str, help='dataset name')
    parser.add_argument('--image_size', default=64, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')
    args = parser.parse_args()



    viz = visdom.Visdom()

    for x, _ in return_data(args):
        x = Variable(cuda(x, True))
        for i in range(-2, 3):
            # recon_model = AutoEncoder
            recon_model = BetaVAE_H
            # 检验z的每个维度
            # recon_model = cuda(recon_model(z_dim=10, nc=3, masked_dim=0, masked_value=i), True)
            recon_model = cuda(recon_model(z_dim=10, nc=3), True)

            # 检验重建效果
            # recon_model = cuda(recon_model(z_dim=10, nc=3), True)
            checkpoints = torch.load('checkpoints_linear_vae/main/last')
            recon_model.load_state_dict(checkpoints['model_states']['net'])
            recon_x, _, _ = recon_model(x)
            viz_reconstruction(x, recon_x, viz, args)

