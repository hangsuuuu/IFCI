import argparse
import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from pathlib import Path
from au_ import make_environment, MyMLP, mean_nll, mean_accuracy, pretty_print
from vae.beta_vae import BetaVAE_B, BetaVAE_H, Simplified_VAE, AutoEncoder
from vae.solver import Solver, reconstruction_loss, kl_divergence
from vae.utils import cuda
from torchvision import datasets
from vae.utils import str2bool
from torch import optim
from vae.dataset import return_data, return_data_test
from tqdm import tqdm
import visdom
import time
import cv2
import os


def train_ae_classify(args):
    wind = visdom.Visdom()
    wind.line([0.],  # Y的第一个点的坐标
              [0.],  # X的第一个点的坐标
              env='ae_classifier',
              win='train_ae_classify',
              opts=dict(title='train_ae_classify'))

    ae_model = AutoEncoder().to('cuda')
    # 加载AE checkpoints
    ae_checkpoints = torch.load('ae_checkpoints1')
    ae_model.load_state_dict(ae_checkpoints['model_states'])
    # 初始化分类模型
    classify_model = MyMLP(args).to('cuda')
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(classify_model.parameters(), lr=args.lr)
    optimizer.zero_grad()

    t = tqdm(total=args.classify_max_iter)
    t.update(args.classify_global_iter)
    # changed
    if os.path.exists('ae_classify_checkpoints_after'):
        print('ae_classifier had been trained')
    #   训练
    else:
        for i in range(0, int(args.classify_max_iter)):
            args.classify_global_iter += 1
            t.update(1)
            for x, label in return_data(args):
                x = Variable(cuda(x, True))
                label = Variable(cuda(label, True))
                # changed
                re_x, _ = ae_model(x)
                classifier = classify_model(re_x)
                classifier_loss = loss(classifier.squeeze(), label.float())
                classifier_loss.backward()
                optimizer.step()

            wind.line([classifier_loss.cpu().detach().numpy()], [i], env='ae_classifier', win='train_ae_classify',
                      update='append')
        classifier_states = {'iter': args.classify_max_iter,
                             'model_states': classify_model.state_dict(),
                             'optim_states': optimizer.state_dict()}
        with open('ae_classify_checkpoints_after', mode='wb+') as f:
            torch.save(classifier_states, f)
        print("=> saved checkpoint '{}' (iter {})".format('ae_classify_checkpoints_after', args.classify_max_iter))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Colored MNIST')
    parser.add_argument('--hidden_dim', type=int, default=390)
    parser.add_argument('--steps', type=int, default=30001)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--path', default='results/', help='The path results to be saved.')
    parser.add_argument('--method', default='Rex', help='ERM, IRM, CoCO')
    parser.add_argument('--grayscale_model', default=False)
    parser.add_argument('--train', default=True, type=str2bool, help='train or traverse')
    parser.add_argument('--model_type', default='VAE', help='VAE, AE')

    #beta_vae
    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=500, type=float, help='maximum training iteration')
    parser.add_argument('--classify_max_iter', default=500, type=float, help='maximum training iteration for classify')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    parser.add_argument('--env_num', default=2, type=int, help='the number of envs')

    parser.add_argument('--z_dim', default=10, type=int, help='dimension of the representation z')
    parser.add_argument('--beta', default=2, type=float, help='beta parameter for KL-term in original beta-VAE')
    parser.add_argument('--objective', default='B', type=str,
                        help='beta-vae objective proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--model', default='H', type=str, help='model proposed in Higgins et al. or Burgess et al. H/B')
    parser.add_argument('--gamma', default=1000, type=float,
                        help='gamma parameter for KL-term in understanding beta-VAE')
    parser.add_argument('--C_max', default=25, type=float, help='capacity parameter(C) of bottleneck channel')
    parser.add_argument('--C_stop_iter', default=1e5, type=float, help='when to stop increasing the capacity')
    parser.add_argument('--beta1', default=0.9, type=float, help='Adam optimizer beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='Adam optimizer beta2')

    parser.add_argument('--dset_dir', default='G:/data/mnist/mnist_dataset_color/baseset/new_img',
                        type=str, help='dataset directory')
    parser.add_argument('--dset_dir1', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/env1', type=str, help='dataset directory')
    parser.add_argument('--dset_dir2', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/env2', type=str, help='dataset directory')
    parser.add_argument('--dset_dir_test', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/test', type=str, help='dataset directory')


    parser.add_argument('--dataset', default='colored_mnist', type=str, help='dataset name')
    parser.add_argument('--image_size', default=28, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')
    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=100, type=int,
                        help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=200, type=int,
                        help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=500, type=int,
                        help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--classifier_ckpt_dir', default='classifier_checkpoints', type=str, help='checkpoint directory')


    args = parser.parse_args()
    args.classify_global_iter = 0

    train_ae_classify(args)
    # 测试
    test_model = MyMLP(args).to('cuda')

    checkpoint = torch.load('ae_classify_checkpoints_after')
    test_model.load_state_dict(checkpoint['model_states'])

    c = 0
    correct = 0
    for x, label in return_data_test(args):
        x = Variable(cuda(x, True))
        label = Variable(cuda(label, True))
        test_y = test_model(x)
        yl = test_y.squeeze().ge(0.5).float()  # 分类
        correct = correct + (yl == label).sum()  # 计算其中的分类正确的个数
        c += 1
    acc = correct.item() / (c * args.batch_size)  # 计算准确率
    print('test_acc=', acc)