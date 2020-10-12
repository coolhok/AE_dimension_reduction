# coding=utf-8
import torch.autograd
import torch.nn as nn


class autoencoder(nn.Module):
    def __init__(self, dimension):
        super(autoencoder, self).__init__()
        self.z_dimension = dimension
        # 定义编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.encoder_fc = nn.Linear(32*8*8, self.z_dimension)
        self.decoder_fc = nn.Linear(self.z_dimension, 32*8*8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 5, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 1, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        code = self.encoder_fc(x)
        x = self.decoder_fc(code)
        x = x.view(x.shape[0], 32, 8, 8)
        decode = self.decoder(x)
        return code, decode
