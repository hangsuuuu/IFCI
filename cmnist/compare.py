import argparse
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from pathlib import Path
from au_ import make_environment, make_myenvironment, MyMLP, mean_nll, mean_accuracy, pretty_print, MLP
from vae.beta_vae import BetaVAE_B, BetaVAE_H, Simplified_VAE
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
    parser.add_argument('--compare_path', default='compare_checkpoints', help='The path compare results to be saved.')

    parser.add_argument('--cuda', default=True, type=str2bool, help='enable cuda')
    parser.add_argument('--max_iter', default=1e5, type=float, help='maximum training iteration')
    parser.add_argument('--classify_max_iter', default=500, type=float, help='maximum training iteration for classify')
    parser.add_argument('--batch_size', default=5000, type=int, help='batch size')
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

    # parser.add_argument('--dset_dir', default='/media/ruc/SH/data/mnist/mnist_dataset_color/baseset/new_img',
    #                     type=str, help='dataset directory')
    # parser.add_argument('--dset_dir1', default='/media/ruc/SH/data/mnist/mnist_dataset_color/baseset/colored_img/env1',
    #                     type=str, help='dataset directory')
    # parser.add_argument('--dset_dir2', default='/media/ruc/SH/data/mnist/mnist_dataset_color/baseset/colored_img/env2',
    #                     type=str, help='dataset directory')
    # parser.add_argument('--dset_dir_test',
    #                     default='/media/ruc/SH/data/mnist/mnist_dataset_color/baseset/colored_img/test', type=str,
    #                     help='dataset directory')
    parser.add_argument('--dset_dir', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img/test',
                        type=str, help='dataset directory')
    parser.add_argument('--dset_dir1', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img_64/env1', type=str,
                        help='dataset directory')
    parser.add_argument('--dset_dir2', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img_64/env2', type=str,
                        help='dataset directory')
    parser.add_argument('--dset_dir_test', default='G:/data/mnist/mnist_dataset_color/baseset/colored_img_64/test',
                        type=str, help='dataset directory')

    parser.add_argument('--data_type', default='coco',
                        type=str,
                        help='coco, my')
    parser.add_argument('--dataset', default='colored_mnist', type=str, help='dataset name')
    parser.add_argument('--image_size', default=28, type=int, help='image size. now only (64,64) is supported')
    parser.add_argument('--num_workers', default=4, type=int, help='dataloader num_workers')
    parser.add_argument('--viz_on', default=True, type=str2bool, help='enable visdom visualization')
    parser.add_argument('--viz_name', default='main', type=str, help='visdom env name')
    parser.add_argument('--viz_port', default=8097, type=str, help='visdom port number')
    parser.add_argument('--save_output', default=True, type=str2bool, help='save traverse images and gif')
    parser.add_argument('--output_dir', default='outputs', type=str, help='output directory')

    parser.add_argument('--gather_step', default=1000, type=int,
                        help='numer of iterations after which data is gathered for visdom')
    parser.add_argument('--display_step', default=2000, type=int,
                        help='number of iterations after which loss data is printed and visdom is updated')
    parser.add_argument('--save_step', default=10000, type=int,
                        help='number of iterations after which a checkpoint is saved')

    parser.add_argument('--ckpt_dir', default='checkpoints', type=str, help='checkpoint directory')
    parser.add_argument('--ckpt_name', default='last', type=str,
                        help='load previous checkpoint. insert checkpoint filename')
    parser.add_argument('--classifier_ckpt_dir', default='classifier_checkpoints', type=str,
                        help='checkpoint directory')
    flags = parser.parse_args()

    mnist = datasets.MNIST('~/datasets/mnist', train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])

    rng_state = np.random.get_state()
    np.random.shuffle(mnist_train[0].numpy())
    np.random.set_state(rng_state)
    np.random.shuffle(mnist_train[1].numpy())

    envs = [
        make_environment(mnist_train[0][::2], mnist_train[1][::2], 0.1),
        make_environment(mnist_train[0][1::2], mnist_train[1][1::2], 0.2),
        make_environment(mnist_val[0], mnist_val[1], 0.9)
    ]
    # for k in range(len(envs[0]['images'])):
    #     img = envs[0]['images'][k]
    #     label = envs[0]['labels'].squeeze()[k]
    #     img = img.permute(1, 2, 0)
    #     img = img.cpu().detach().numpy()
    #     print('label:', label.item())
    #     cv2.imshow('img', img)
    #     cv2.waitKey(0)
    #     print('max=', np.max(img))

    model = MyMLP(flags).to('cuda')
    # checkpoint = torch.load('checkpoints_classifier_new2/400')
    # model.load_state_dict(checkpoint['model_states'])
    coco_model = MLP(flags).cuda()
    loss = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(coco_model.parameters(), lr=flags.lr)
    optimizer.zero_grad()
    os.makedirs('checkpoints_classifier_new2', exist_ok=True)

    s = 0
    t = tqdm(total=30000)
    t.update(s)

    wind = visdom.Visdom()
    # wind.line([0.],  # Y的第一个点的坐标
    #           [0.],  # X的第一个点的坐标
    #           env='compare_loss',
    #           win='compare_loss',
    #           opts=dict(title='compare_loss'))

    if os.path.exists('checkpoints_classifier_new1') and os.path.exists('compare_checkpoints_coco'):
        print('classifier had been trained')
    else:
        for i in range(0, 30000):
            t.update(1)
            if flags.data_type == 'my':
                for x, label in return_data(flags):
                    # x0 = x[0].permute(1,2,0)
                    # x0 = x0.cpu().detach().numpy()
                    # cv2.imshow('img', x0)
                    # cv2.waitKey(0)
                    env1 = make_myenvironment(x[::2], label[::2], 0.2)
                    env2 = make_myenvironment(x[1::2], label[1::2], 0.1)
                    # 测试代码
                    # for k in range(len(env1['images'])):
                    #     img = env1['images'][k]
                    #     label = env1['labels'].squeeze()[k]
                    #     img = img.permute(1, 2, 0)
                    #     img = img.cpu().detach().numpy()
                    #     print('label:', label.item())
                    #     cv2.imshow('img', img)
                    #     cv2.waitKey(0)
                    #     print('max=', np.max(img))
                    x1 = env1['images']
                    labels1 = env1['labels_nonoise']
                    x2 = env2['images']
                    labels2 = env2['labels_nonoise']
                    # x = Variable(cuda(x, True))
                    # label = Variable(cuda(label, True))
                    # pre_y = model(x)
                    pre_y1 = model(x1)
                    pre_y2 = model(x2)
                    # classifier_loss = loss(pre_y.squeeze(), label.float())
                    classifier_loss1 = loss(pre_y1.squeeze(), labels1.squeeze().float())
                    classifier_loss2 = loss(pre_y2.squeeze(), labels2.squeeze().float())

                    # train_acc2 = mean_accuracy(pre_y2, labels2)
                    classifier_loss = torch.stack([classifier_loss1, classifier_loss2]).mean()
                    # train_acc = torch.stack([train_acc1, train_acc2]).mean()
                    optimizer.zero_grad()
                    classifier_loss.backward()
                    optimizer.step()
                    if i % 50 == 0:
                        train_acc1 = mean_accuracy(pre_y1.squeeze(), labels1.squeeze())
                        print('train_acc=', train_acc1)
            elif flags.data_type == 'coco':
                for env in envs:
                    pre_y = coco_model(env['images'])
                    risk_e = mean_nll(pre_y, env['labels'])
                    env['nll'] = risk_e
                    env['acc'] = mean_accuracy(pre_y, env['labels'])
                    env['acc_true'] = mean_accuracy(pre_y, env['labels_nonoise'])
                train_nll = torch.stack([envs[0]['nll'], envs[1]['nll']]).mean()
                classifier_loss = train_nll.clone()
                optimizer.zero_grad()
                classifier_loss.backward()
                optimizer.step()

                train_acc = torch.stack([envs[0]['acc'], envs[1]['acc']]).mean()
                test_nll = envs[2]['nll']
                test_acc = envs[2]['acc']
                test_acc_true = envs[2]['acc_true']
                if i % 500 == 0:
                    print('train_acc=', train_acc.item())
                    print('test_acc=', test_acc.item())
                    print('test_acc_true=', test_acc_true.item())

            wind.line(Y=[[classifier_loss.cpu().detach().numpy(), train_acc.item(), test_acc_true.item()]], X=[i],env='compare_loss', win='compare_loss', update='append', opts=dict(legend=['train_loss', 'train_acc', 'test_acc']))
            if flags.data_type == 'my':
                if i % 100 == 0:
                    file_path = 'checkpoints_classifier_new2/' + str(i)
                    classifier_states = {'iter': i,
                                         'model_states': model.state_dict(),
                                         'optim_states': optimizer.state_dict()}
                    with open(file_path, mode='wb+') as f:
                        torch.save(classifier_states, f)
                    print("=> saved checkpoint '{}' (iter {})".format('compare_checkpoints_newenv', i))
            elif flags.data_type == 'coco':
                if i % 1000 == 0:
                    classifier_states = {'iter': 30000,
                                         'model_states': coco_model.state_dict(),
                                         'optim_states': optimizer.state_dict()}
                    with open('compare_checkpoints_coco2', mode='wb+') as f:
                        torch.save(classifier_states, f)
                    print("=> saved checkpoint '{}' (iter {})".format('compare_checkpoints_coco1', i))

    if flags.data_type == 'my':
        test_model = MyMLP(flags).to('cuda')
        checkpoint = torch.load('checkpoints_classifier_new2/1400')
        test_model.load_state_dict(checkpoint['model_states'])
        c = 0
        correct = 0
        for x, label in return_data_test(flags):
            env_test = make_myenvironment(x, label, 1)
            x_test = env_test['images']
            labels_test = env_test['labels_nonoise']
            # x = Variable(cuda(x, True))
            # label = Variable(cuda(label, True))
            test_y = test_model(x_test)
            yl = test_y.squeeze().ge(0.5).float()  # 分类
            correct = correct + (yl == labels_test.squeeze()).sum()  # 计算其中的分类正确的个数
            c += 1
        acc = correct.item() / (c * flags.batch_size)  # 计算准确率
        print('test_acc=', acc)


