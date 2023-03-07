import torch
import torch.nn as nn
#import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable


def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std*eps


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)


class Simplified_VAE(nn.Module):

    def __init__(self, z_dim=10, nc=3):
        super(Simplified_VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            View((-1, 3 * 28 * 28)),  # B,  3*28*28
            nn.Linear(nc * 28 * 28, 256),  # B,  256
            nn.ReLU(True),
            nn.Linear(256, 64),  # B,  64
            nn.ReLU(True),
            nn.Linear(64, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, z_dim * 2),  # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 32),  # B, 32
            nn.ReLU(True),
            nn.Linear(32, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, 64),  # B,  64
            nn.ReLU(True),
            nn.Linear(64, 256),  # B,  256
            nn.ReLU(True),
            nn.Linear(256, nc * 28 * 28),  # B,  nc*28*28
            View((-1, nc, 28, 28))
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]

        return mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Simplified_AutoEncoder(nn.Module):
    def __init__(self, z_dim=50, nc=3):
        super(Simplified_AutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            View((-1, 3 * 28 * 28)),  # B,  3*28*28
            nn.Linear(nc * 28 * 28, 256),  # B,  256
            nn.ReLU(True),
            nn.Linear(256, 64),  # B,  64
            nn.ReLU(True),
            nn.Linear(64, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, z_dim),  # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 32),  # B, 32
            nn.ReLU(True),
            nn.Linear(32, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, 64),  # B,  64
            nn.ReLU(True),
            nn.Linear(64, 256),  # B,  256
            nn.ReLU(True),
            nn.Linear(256, nc * 28 * 28),  # B,  nc*28*28
            View((-1, nc, 28, 28))
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        return z

    def _encode(self, x):
        return self.encoder(x)


class Linear_vae(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(Linear_vae, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            View((-1, 3 * 28 * 28)),  # B,  3*28*28
            nn.Linear(nc*28*28, 256),  # B,  256
            nn.ReLU(True),
            nn.Linear(256, 64),  # B,  64
            nn.ReLU(True),
            nn.Linear(64, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, z_dim*2),  # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 32),  # B, 32
            nn.ReLU(True),
            nn.Linear(32, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, 64),  # B,  64
            nn.ReLU(True),
            nn.Linear(64, 256),  # B,  256
            nn.ReLU(True),
            nn.Linear(256, nc*28*28),  # B,  nc*28*28
            View((-1, nc, 28, 28))
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]
        z = reparametrize(mu, logvar)
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Linear_ae(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(Linear_ae, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            View((-1, 3 * 28 * 28)),  # B,  3*28*28
            nn.Linear(nc*28*28, 256),  # B,  256
            nn.ReLU(True),
            nn.Linear(256, 64),  # B,  64
            nn.ReLU(True),
            nn.Linear(64, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, z_dim),  # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 32),  # B, 32
            nn.ReLU(True),
            nn.Linear(32, 32),  # B,  32
            nn.ReLU(True),
            nn.Linear(32, 64),  # B,  64
            nn.ReLU(True),
            nn.Linear(64, 256),  # B,  256
            nn.ReLU(True),
            nn.Linear(256, nc*28*28),  # B,  nc*28*28
            View((-1, nc, 28, 28))
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Simplified_AutoEncoder(nn.Module):
    def __init__(self, z_dim=50, nc=3):
        super(Simplified_AutoEncoder, self).__init__()
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

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)

        return z

    def _encode(self, x):
        return self.encoder(x)


class AutoEncoder(nn.Module):
    def __init__(self, z_dim=10, nc=3):
        super(AutoEncoder, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 7, 7
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),  # B,  64, 4, 4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B,  256, 1, 1
            nn.ReLU(True),
            View((-1, 256*1*1)),  # B, 256
            nn.Linear(256*1*1, z_dim),  # B, z_dim
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4, 1),  # B,  64, 4, 4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1),  # B,  32, 7, 7
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 28, 28
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        z = self._encode(x)
        x_recon = self._decode(z)

        return x_recon, z

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class Simplified_VAE(nn.Module):

    def __init__(self, z_dim=10, nc=3):
        super(Simplified_VAE, self).__init__()
        self.z_dim = z_dim
        self.nc = nc
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
        #     nn.ReLU(True),
        #     View((-1, 256 * 1 * 1)),  # B, 256
        #     nn.Linear(256, z_dim * 2),  # B, z_dim*2
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim, 256),  # B, 256
        #     View((-1, 256, 1, 1)),  # B, 256,  1,  1
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        # )
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 7, 7
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),  # B,  64, 4, 4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B,  256, 1, 1
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256 * 1 * 1, z_dim * 2),  # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),  # B, 256
            View((-1, 256, 1, 1)),  # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4, 1),  # B,  64, 4, 4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1),  # B,  32, 7, 7
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 14, 14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B,  nc, 28, 28
        )

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

    def forward(self, x):
        distributions = self._encode(x)
        mu = distributions[:, :self.z_dim]
        logvar = distributions[:, self.z_dim:]

        return mu, logvar

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
        # 512*512
        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 3, 2, 1),  # B,  32, 256, 256
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 2, 1),  # B,  32, 128, 128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1),  # B,  64,  64, 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 3, 2, 1),  # B,  64,  32,  32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 3, 2, 1),  # B, 256,  16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 2, 1),  # B, 256,  8, 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.Conv2d(256, 512, 3, 2, 1),  # B, 512, 4,  4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 2, 1),  # B, 512, 2,  2
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            View((-1, 512*2*2)),  # B, 256
            nn.Linear(512*2*2, z_dim * 2),  # B, z_dim*2
        )
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512*2*2),  # B, 512*2*2
            View((-1, 512, 2, 2)),  # B, 512,  2,  2
            nn.ConvTranspose2d(512, 512, 3, 2, 1, output_padding=1),  # B,  512,  4,  4
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 3, 2, 1, output_padding=1),  # B,  256,  8,  8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 256, 3, 2, 1, output_padding=1),  # B,  256, 16, 16
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 3, 2, 1, output_padding=1),  # B,  64, 32, 32
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 3, 2, 1, output_padding=1),  # B, 64, 64, 64
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 3, 2, 1, output_padding=1),  # B, 32, 128, 128
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=1),  # B, 32, 256, 256
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 3, 2, 1, output_padding=1),  # B, 3, 512, 512
        )
        # 64*64
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(nc, 32, 4, 2, 1),  # B,  32, 32, 32
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
        #     nn.ReLU(True),
        #     View((-1, 256 * 1 * 1)),  # B, 256
        #     nn.Linear(256, z_dim * 2),  # B, z_dim*2
        # )
        # self.decoder = nn.Sequential(
        #     nn.Linear(z_dim, 256),  # B, 256
        #     View((-1, 256, 1, 1)),  # B, 256,  1,  1
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(256, 64, 4),  # B,  64,  4,  4
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 64, 4, 2, 1),  # B,  64,  8,  8
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(64, 32, 4, 2, 1),  # B,  32, 16, 16
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, 32, 4, 2, 1),  # B,  32, 32, 32
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(32, nc, 4, 2, 1),  # B, nc, 64, 64
        # )
        # 28*28
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
        x_recon = self._decode(z)

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


class BetaVAE_B(BetaVAE_H):
    """Model proposed in understanding beta-VAE paper(Burgess et al, arxiv:1804.03599, 2018)."""

    def __init__(self, z_dim=10, nc=3):
        super(BetaVAE_B, self).__init__()
        self.nc = nc
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(nc, 32, 4, 2, 1),          # B,  32, 14, 14
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 7, 7
            nn.ReLU(True),
            View((-1, 32*7*7)),                  # B, 1568
            nn.Linear(32*7*7, 256),              # B, 256
            nn.ReLU(True),
            nn.Linear(256, z_dim*2),             # B, z_dim*2
        )

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*7*7),              # B, 1568
            nn.ReLU(True),
            View((-1, 32, 7, 7)),                # B,  32,  7,  7
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 14, 14
            nn.ReLU(True),
            nn.ConvTranspose2d(32, nc, 4, 2, 1), # B,  nc, 28, 28
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
        x_recon = self._decode(z).view(x.size())

        return x_recon, mu, logvar

    def _encode(self, x):
        return self.encoder(x)

    def _decode(self, z):
        return self.decoder(z)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()


if __name__ == '__main__':
    pass