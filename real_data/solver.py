"""solver.py"""

import warnings

import numpy as np

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

from vae.utils import cuda, grid2gif
from vae.beta_vae import BetaVAE_H, BetaVAE_B, Simplified_VAE, AutoEncoder, Simplified_AutoEncoder, Linear_vae, Linear_ae
from vae.dataset import return_data, return_data1, return_data2, return_data_test

from au_ import LR, LR_realdata
from gmm.casual_infer import casual_inference
from resnet_vae import VAE, loss_func, my_VAE
from compare import ResNet18Enc

def reconstruction_loss(x, x_recon, distribution):
    batch_size = x.size(0)
    assert batch_size != 0

    if distribution == 'bernoulli':
        recon_loss = F.binary_cross_entropy_with_logits(x_recon, x, size_average=False).div(batch_size)
    elif distribution == 'gaussian':
        x_recon = F.sigmoid(x_recon)
        recon_loss = F.mse_loss(x_recon, x, size_average=False).div(batch_size)
    else:
        recon_loss = None

    return recon_loss


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5*(1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld


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


class Solver(object):
    def __init__(self, args):
        self.use_cuda = args.cuda and torch.cuda.is_available()
        self.max_iter = int(args.max_iter)
        self.classify_max_iter = args.classify_max_iter
        self.global_iter = 0
        self.classify_global_iter = 0
        self.image_size = args.image_size

        self.z_dim = args.z_dim
        self.beta = args.beta
        self.gamma = args.gamma
        self.C_max = args.C_max
        self.C_stop_iter = args.C_stop_iter
        self.objective = args.objective
        self.model = args.model
        self.lr = args.lr
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.masks = 0
        self.masks_var = 0
        self.n_env = args.n_env

        if args.dataset.lower() == 'colored_mnist':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'dsprites':
            self.nc = 1
            self.decoder_dist = 'bernoulli'
        elif args.dataset.lower() == '3dchairs':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        elif args.dataset.lower() == 'celeba':
            self.nc = 3
            self.decoder_dist = 'gaussian'
        else:
            raise NotImplementedError

        if args.model == 'H':
            net = BetaVAE_H
        elif args.model == 'B':
            net = BetaVAE_B
        elif args.model == 'Linear_vae':
            net = Linear_vae
        elif args.model == 'Linear_ae':
            net = Linear_ae
        elif args.model == 'resnet_vae':
            net = VAE
        else:
            raise NotImplementedError('only support model H or B')

        s_net = Simplified_VAE
        s_ae = Simplified_AutoEncoder

        self.net = cuda(net(self.z_dim, self.nc), self.use_cuda)
        self.s_net = cuda(s_net(self.z_dim, self.nc), False)
        self.s_ae = cuda(s_ae(self.z_dim, self.nc), self.use_cuda)
        self.optim = optim.Adam(self.net.parameters(), lr=self.lr,
                                    betas=(self.beta1, self.beta2))

        classify_net = LR()
        self.classify_model = classify_net.to('cuda')

        self.viz_name = args.viz_name
        self.viz_port = args.viz_port
        self.viz_on = args.viz_on
        self.win_recon = None
        self.win_kld = None
        self.win_mu = None
        self.win_var = None
        if self.viz_on:
            self.viz = visdom.Visdom(port=self.viz_port)

        self.ckpt_dir = os.path.join(args.ckpt_dir, args.viz_name)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)
        self.ckpt_name = args.ckpt_name
        if self.ckpt_name is not None:
            print('load checkpoints')
            self.load_checkpoint(self.ckpt_name)

        self.classifier_ckpt_dir = args.classifier_ckpt_dir
        self.classifier_ckpt_dir_realdata = args.classifier_ckpt_dir_realdata

        self.save_output = args.save_output
        self.output_dir = os.path.join(args.output_dir, args.viz_name)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        self.gather_step = args.gather_step
        self.display_step = args.display_step
        self.save_step = args.save_step

        self.dset_dir = args.dset_dir
        self.dset_dir1 = args.dset_dir1
        self.dset_dir2 = args.dset_dir2
        self.dset_dir_test = args.dset_dir_test

        self.dataset = args.dataset
        self.batch_size = args.batch_size
        self.data_loader = return_data(args)
        self.data_loader1 = return_data1(args)
        self.data_loader2 = return_data2(args)
        self.data_loader_test = return_data_test(args)

        self.gather = DataGather()

    def train(self):
        self.net_mode(train=True)
        self.C_max = Variable(cuda(torch.FloatTensor([self.C_max]), self.use_cuda))
        out = False

        pbar = tqdm(total=self.max_iter)
        pbar.update(self.global_iter)
        while not out:
            # self.global_iter += 1
            # pbar.update(1)

            for x, label in self.data_loader:
                self.global_iter += 1
                pbar.update(1)
                x = Variable(cuda(x, self.use_cuda))

                x_recon, mu, logvar = self.net(x)

                recon_loss = reconstruction_loss(x, x_recon, self.decoder_dist)
                total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

                if self.objective == 'H':
                    beta_vae_loss = recon_loss + self.beta*total_kld
                elif self.objective == 'B':
                    C = torch.clamp(self.C_max/self.C_stop_iter*self.global_iter, 0, self.C_max.data[0])
                    beta_vae_loss = recon_loss + self.gamma*(total_kld-C).abs()
                elif self.objective == 'resnet':
                    beta_vae_loss = loss_func(x_recon, x, mu, logvar)

                self.optim.zero_grad()
                beta_vae_loss.backward()
                self.optim.step()

                if self.viz_on and self.global_iter%self.gather_step == 0:
                    self.gather.insert(iter=self.global_iter,
                                       mu=mu.mean(0).data, var=logvar.exp().mean(0).data,
                                       recon_loss=recon_loss.data, total_kld=total_kld.data,
                                       dim_wise_kld=dim_wise_kld.data, mean_kld=mean_kld.data)

                if self.global_iter%self.display_step == 0:
                    pbar.write('[{}] recon_loss:{:.3f} total_kld:{:.3f} mean_kld:{:.3f}'.format(
                        self.global_iter, recon_loss.item(), total_kld.data[0], mean_kld.data[0]))

                    var = logvar.exp().mean(0).data
                    var_str = ''
                    for j, var_j in enumerate(var):
                        var_str += 'var{}:{:.4f} '.format(j+1, var_j)
                    pbar.write(var_str)

                    if self.objective == 'B':
                        pbar.write('C:{:.3f}'.format(C.data[0]))

                    if self.viz_on:
                        self.gather.insert(images=x.data)
                        self.gather.insert(images=F.sigmoid(x_recon).data)
                        self.viz_reconstruction()
                        self.viz_lines()
                        self.gather.flush()

                    # if self.viz_on or self.save_output:
                    #     self.viz_traverse()

                if self.global_iter%self.save_step == 0:
                    self.save_checkpoint('last')
                    pbar.write('Saved checkpoint(iter:{})'.format(self.global_iter))

                if self.global_iter%50000 == 0:
                    self.save_checkpoint(str(self.global_iter))

                if self.global_iter >= self.max_iter:
                    out = True
                    break

        pbar.write("[Training Finished]")
        pbar.close()

    def casual_inference(self):
        # 数据预处理
        env = []
        envs1 = np.empty((0, self.z_dim*2))
        labels1 = np.empty((0)).astype(int)
        envs2 = np.empty((0, self.z_dim*2))
        labels2 = np.empty((0)).astype(int)
        for x, label in self.data_loader1:
            x = Variable(cuda(x, self.use_cuda))
            _, mu, logvar = self.net(x)
            mu = mu.cpu().detach().numpy()
            logvar = logvar.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            data1 = np.concatenate((mu, logvar), axis=1)
            envs1 = np.concatenate((envs1, data1), axis=0)
            labels1 = np.concatenate((labels1, label))
        for x, label in self.data_loader2:
            x = Variable(cuda(x, self.use_cuda))
            _, mu, logvar = self.net(x)
            mu = mu.cpu().detach().numpy()
            logvar = logvar.cpu().detach().numpy()
            label = label.cpu().detach().numpy()
            data2 = np.concatenate((mu, logvar), axis=1)
            envs2 = np.concatenate((envs2, data2), axis=0)
            labels2 = np.concatenate((labels2, label))
        env.append([envs1, labels1])
        env.append([envs2, labels2])
        # 因果推断
        casual_var = casual_inference(self, env)
        np.savetxt('masks.txt', np.array(casual_var))
        return casual_var

    def train_classifier_realdata(self):
        masks = self.casual_inference()
        # masks = np.loadtxt('masks.txt')

        self.masks = masks
        features_dim = len(self.masks)
        print('features_dim=', features_dim)

        # loss函数可视化
        wind = visdom.Visdom()
        wind.line([0.],  # Y的第一个点的坐标
                  [0.],  # X的第一个点的坐标
                  env='train_loss',
                  win='train_loss',
                  opts=dict(title='train_loss'))

        # 初始化训练参数
        model = self.net
        checkpoint = torch.load('checkpoints_resnet_vae_realdata_size224/main/last')
        model.load_state_dict(checkpoint['model_states']['net'])

        self.classify_model_realdata = LR_realdata(features_dim)
        classify_model = self.classify_model_realdata.to('cuda')
        # classify_model = ResNet18Enc().cuda()
        loss = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(classify_model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
        optimizer.zero_grad()

        # 训练分类模型
        t = tqdm(total=self.classify_max_iter)
        t.update(self.classify_global_iter)

        if os.path.exists(self.classifier_ckpt_dir_realdata):
            print('classifier had been trained')
        else:
            for i in range(0, int(self.classify_max_iter)):
                self.classify_global_iter += 1
                t.update(1)
                for x, label in self.data_loader:
                    x = Variable(cuda(x, self.use_cuda))
                    label = Variable(cuda(label, self.use_cuda))

                    # changed
                    recon_x, mu, logvar = model(x)
                    mu = mu.cpu()
                    logvar = logvar.cpu()
                    masked_z = torch.cat([mu, logvar], 1)
                    masked_z = masked_z[:, masks]
                    masked_z = Variable(cuda(masked_z, self.use_cuda))

                    classifier = classify_model(masked_z)
                    classifier_loss = loss(classifier.squeeze(), label.float())
                    classifier_loss.backward()
                    optimizer.step()
                    scheduler.step()

                wind.line([classifier_loss.cpu().detach().numpy()], [i], env='train_loss', win='train_loss', update='append')

    #         保存分类器的checkpoints
            classifier_states = {'iter': self.classify_max_iter,
                                 'model_states': classify_model.state_dict(),
                                 'optim_states': optimizer.state_dict()}
            with open(self.classifier_ckpt_dir_realdata, mode='wb+') as f:
                torch.save(classifier_states, f)
            print("=> saved checkpoint '{}' (iter {})".format(self.classifier_ckpt_dir_realdata, self.classify_max_iter))

    def test_classify(self):
        masks = np.loadtxt('masks.txt')
        self.masks = masks

        correct = 0
        c = 0
        model = self.net
        checkpoint = torch.load('checkpoints_resnet_vae_realdata_size224/main/last')
        model.load_state_dict(checkpoint['model_states']['net'])
        self.classify_model_realdata = LR_realdata(40)
        # self.classify_model_realdata = ResNet18Enc().cuda()
        for x, label in self.data_loader_test:
            x = Variable(cuda(x, self.use_cuda))
            label = Variable(cuda(label, self.use_cuda))
            # changed
            _, mu, logvar = model(x)
            mu = mu.cpu()
            logvar = logvar.cpu()
            z = torch.cat([mu,logvar], 1)
            z = Variable(cuda(z, self.use_cuda))
            masked_z = z[:, masks]
            masked_z = Variable(cuda(masked_z, self.use_cuda))

            checkpoint = torch.load(self.classifier_ckpt_dir_realdata)
            self.classify_model_realdata.load_state_dict(checkpoint['model_states'])

            test_model = self.classify_model_realdata.to('cuda')
            y = test_model(z)
            yl = y.squeeze().ge(0.5).float()  # 分类
            correct = correct + (yl == label).sum()  # 计算其中的分类正确的个数
            c += 1
        acc = correct.item() / (c * self.batch_size)  # 计算准确率
        print('test_acc=', acc)

    def recon_img(self):
        masks = self.casual_inference()
        self.masks = masks

        for x, _ in self.data_loader:
            print(1)
            x = Variable(cuda(x, True))
            recon_model = my_VAE(20, 3).cuda()
            checkpoints = torch.load('checkpoints_resnet_vae_realdata_size224/main/last')
            recon_model.load_state_dict(checkpoints['model_states']['net'])

            recon_x, _, _ = recon_model(x, self.masks)

            self.gather.insert(images=x.data)
            self.gather.insert(images=F.sigmoid(recon_x).data)
            self.viz_reconstruction()

    def viz_reconstruction(self):
        self.net_mode(train=False)
        x = self.gather.data['images'][0][:4]
        x = make_grid(x, normalize=True)
        x_recon = self.gather.data['images'][1][:4]
        x_recon = make_grid(x_recon, normalize=True)
        images = torch.stack([x, x_recon], dim=0).cpu()
        self.viz.images(images, env=self.viz_name+'_reconstruction',
                        opts=dict(title=str(self.global_iter)), nrow=2)
        self.net_mode(train=True)

    def viz_lines(self):
        self.net_mode(train=False)
        recon_losses = torch.stack(self.gather.data['recon_loss']).cpu()

        mus = torch.stack(self.gather.data['mu']).cpu()
        vars = torch.stack(self.gather.data['var']).cpu()

        dim_wise_klds = torch.stack(self.gather.data['dim_wise_kld'])
        mean_klds = torch.stack(self.gather.data['mean_kld'])
        total_klds = torch.stack(self.gather.data['total_kld'])
        klds = torch.cat([dim_wise_klds, mean_klds, total_klds], 1).cpu()
        iters = torch.Tensor(self.gather.data['iter'])

        legend = []
        for z_j in range(self.z_dim):
            legend.append('z_{}'.format(z_j))
        legend.append('mean')
        legend.append('total')

        if self.win_recon is None:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))
        else:
            self.win_recon = self.viz.line(
                                        X=iters,
                                        Y=recon_losses,
                                        env=self.viz_name+'_lines',
                                        win=self.win_recon,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            xlabel='iteration',
                                            title='reconsturction loss',))

        if self.win_kld is None:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))
        else:
            self.win_kld = self.viz.line(
                                        X=iters,
                                        Y=klds,
                                        env=self.viz_name+'_lines',
                                        win=self.win_kld,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend,
                                            xlabel='iteration',
                                            title='kl divergence',))

        if self.win_mu is None:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=mus,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))
        else:
            self.win_mu = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_mu,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior mean',))

        if self.win_var is None:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        else:
            self.win_var = self.viz.line(
                                        X=iters,
                                        Y=vars,
                                        env=self.viz_name+'_lines',
                                        win=self.win_var,
                                        update='append',
                                        opts=dict(
                                            width=400,
                                            height=400,
                                            legend=legend[:self.z_dim],
                                            xlabel='iteration',
                                            title='posterior variance',))
        self.net_mode(train=True)

    def viz_traverse(self, limit=1, inter=2/3, loc=-1):
        self.net_mode(train=False)
        import random

        decoder = self.net.decoder
        encoder = self.net.encoder
        interpolation = torch.arange(-limit, limit+0.1, inter)

        n_dsets = len(self.data_loader.dataset)
        rand_idx = random.randint(1, n_dsets-1)

        random_img, label = self.data_loader.dataset.__getitem__(rand_idx)
        random_img = Variable(cuda(random_img, self.use_cuda), volatile=True).unsqueeze(0)
        random_img_z = encoder(random_img)[:, :self.z_dim]

        random_z = Variable(cuda(torch.rand(1, self.z_dim), self.use_cuda), volatile=True)

        if self.dataset == 'dsprites':
            fixed_idx1 = 87040 # square
            fixed_idx2 = 332800 # ellipse
            fixed_idx3 = 578560 # heart

            fixed_img1 = self.data_loader.dataset.__getitem__(fixed_idx1)
            fixed_img1 = Variable(cuda(fixed_img1, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z1 = encoder(fixed_img1)[:, :self.z_dim]

            fixed_img2 = self.data_loader.dataset.__getitem__(fixed_idx2)
            fixed_img2 = Variable(cuda(fixed_img2, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z2 = encoder(fixed_img2)[:, :self.z_dim]

            fixed_img3 = self.data_loader.dataset.__getitem__(fixed_idx3)
            fixed_img3 = Variable(cuda(fixed_img3, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z3 = encoder(fixed_img3)[:, :self.z_dim]

            Z = {'fixed_square':fixed_img_z1, 'fixed_ellipse':fixed_img_z2,
                 'fixed_heart':fixed_img_z3, 'random_img':random_img_z}
        else:
            fixed_idx = 0
            fixed_img, label = self.data_loader.dataset.__getitem__(fixed_idx)
            fixed_img = Variable(cuda(fixed_img, self.use_cuda), volatile=True).unsqueeze(0)
            fixed_img_z = encoder(fixed_img)[:, :self.z_dim]

            Z = {'fixed_img':fixed_img_z, 'random_img':random_img_z, 'random_z':random_z}

        gifs = []
        for key in Z.keys():
            z_ori = Z[key]
            samples = []
            for row in range(self.z_dim):
                if loc != -1 and row != loc:
                    continue
                z = z_ori.clone()
                for val in interpolation:
                    z[:, row] = val
                    sample = F.sigmoid(decoder(z)).data
                    samples.append(sample)
                    gifs.append(sample)
            samples = torch.cat(samples, dim=0).cpu()
            title = '{}_latent_traversal(iter:{})'.format(key, self.global_iter)

            if self.viz_on:
                self.viz.images(samples, env=self.viz_name+'_traverse',
                                opts=dict(title=title), nrow=len(interpolation))

        if self.save_output:
            output_dir = os.path.join(self.output_dir, str(self.global_iter))
            os.makedirs(output_dir, exist_ok=True)
            gifs = torch.cat(gifs)
            gifs = gifs.view(len(Z), self.z_dim, len(interpolation), self.nc, self.image_size, self.image_size).transpose(1, 2)
            for i, key in enumerate(Z.keys()):
                for j, val in enumerate(interpolation):
                    save_image(tensor=gifs[i][j].cpu(),
                               fp=os.path.join(output_dir, '{}_{}.jpg'.format(key, j)),
                               nrow=self.z_dim, pad_value=1)

                grid2gif(os.path.join(output_dir, key+'*.jpg'),
                         os.path.join(output_dir, key+'.gif'), delay=10)

        self.net_mode(train=True)

    def net_mode(self, train):
        if not isinstance(train, bool):
            raise('Only bool type is supported. True or False')

        if train:
            self.net.train()
        else:
            self.net.eval()

    def save_checkpoint(self, filename, silent=True):
        model_states = {'net':self.net.state_dict(),}
        optim_states = {'optim':self.optim.state_dict(),}
        win_states = {'recon':self.win_recon,
                      'kld':self.win_kld,
                      'mu':self.win_mu,
                      'var':self.win_var,}
        states = {'iter':self.global_iter,
                  'win_states':win_states,
                  'model_states':model_states,
                  'optim_states':optim_states}

        file_path = os.path.join(self.ckpt_dir, filename)
        with open(file_path, mode='wb+') as f:
            torch.save(states, f)
        if not silent:
            print("=> saved checkpoint '{}' (iter {})".format(file_path, self.global_iter))

    def load_checkpoint(self, filename):
        file_path = os.path.join(self.ckpt_dir, filename)
        if os.path.isfile(file_path):
            checkpoint = torch.load(file_path)
            self.global_iter = checkpoint['iter']
            self.win_recon = checkpoint['win_states']['recon']
            self.win_kld = checkpoint['win_states']['kld']
            self.win_var = checkpoint['win_states']['var']
            self.win_mu = checkpoint['win_states']['mu']
            self.net.load_state_dict(checkpoint['model_states']['net'])
            self.optim.load_state_dict(checkpoint['optim_states']['optim'])
            print("=> loaded checkpoint '{} (iter {})'".format(file_path, self.global_iter))
        else:
            print("=> no checkpoint found at '{}'".format(file_path))
