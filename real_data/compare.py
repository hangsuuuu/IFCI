import warnings
warnings.filterwarnings("ignore")

import os
from tqdm import tqdm
import visdom
import time
import cv2

import argparse

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.utils import make_grid, save_image

from vae.utils import cuda, grid2gif
from vae.beta_vae import BetaVAE_H, BetaVAE_B, Simplified_VAE, AutoEncoder, Simplified_AutoEncoder, Linear_vae, Linear_ae
from vae.dataset import return_data, return_data1, return_data2, return_data_test

from au_ import LR, LR_realdata
from gmm.casual_infer import casual_inference
from resnet_vae import VAE, loss_func
from vae.utils import str2bool

import torchvision.models as models
import torch.nn as nn


class ResNet18Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet18Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet18 = models.resnet18(pretrained=True)
        self.num_feature = self.ResNet18.fc.in_features
        self.ResNet18.fc = nn.Linear(self.num_feature, self.z_dim)
        self.Linear = nn.Linear(self.z_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ResNet18(x)
        x = self.Linear(x)
        x = self.sigmoid(x)
        return x


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
    parser.add_argument('--n_env', type=int, default=2)

    #beta_vae
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=50000, type=float, help='maximum training iteration')
    parser.add_argument('--classify_max_iter', default=500, type=float, help='maximum training iteration for classify')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--env_num', default=2, type=int, help='the number of envs')

    parser.add_argument('--z_dim', default=20, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=2, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--objective', default='resnet', type=str,
                        help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='resnet_vae', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
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

    parser.add_argument('--ckpt_dir', default='checkpoints_resnet_vae_realdata_size224', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--classifier_ckpt_dir', default='classifier_checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--classifier_ckpt_dir_realdata', default='classifier_checkpoints', type=str,
                        help='checkpoint directory')

    flags = parser.parse_args()

    if os.path.exists('compare'):
        print('classifier had been trained')

    else:
        model = ResNet18Enc().cuda()
        loss = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
        optimizer.zero_grad()

        wind = visdom.Visdom()
        wind.line([0.],  # Y的第一个点的坐标
                  [0.],  # X的第一个点的坐标
                  env='train_loss',
                  win='train_loss',
                  opts=dict(title='train_loss'))
        for i in range(0, 1000):
            for x, label in return_data(flags):
                x = Variable(cuda(x, True))
                label = Variable(cuda(label, True))

                pre_y = model(x)
                classifier_loss = loss(pre_y.squeeze(), label.float())
                classifier_loss.backward()
                optimizer.step()

            wind.line([classifier_loss.cpu().detach().numpy()], [i], env='train_loss', win='train_loss', update='append')

            if i % 200 == 0 and i > 0:
                classifier_states = {'iter': i,
                                     'model_states': model.state_dict(),
                                     'optim_states': optimizer.state_dict()}
                with open('compare', mode='wb+') as f:
                    torch.save(classifier_states, f)
                print(
                    "=> saved checkpoint '{}' (iter {})".format('compare', i))
    # 测试结果
    test_model = ResNet18Enc().cuda()
    checkpoint = torch.load('compare')
    test_model.load_state_dict(checkpoint['model_states'])
    c = 0
    correct = 0
    for x, label in return_data_test(flags):
        x = Variable(cuda(x, True))
        label = Variable(cuda(label, True))
        test_y = test_model(x)
        yl = test_y.squeeze().ge(0.5).float()  # 分类
        correct = correct + (yl == label.squeeze()).sum()  # 计算其中的分类正确的个数
        c += 1
    acc = correct.item() / (c * flags.batch_size)  # 计算准确率
    print('test_acc=', acc)