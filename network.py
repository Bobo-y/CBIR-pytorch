import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicCBR(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, padding):
        super(BasicCBR, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)


class Encoder(nn.Module):
    def __init__(self, img_shape=(28, 28, 1), latent_dim=16):
        super(Encoder, self).__init__()
        self.img_shape = img_shape
        self.latent_dim = latent_dim
        self.conv1 = BasicCBR(in_channel=self.img_shape[2], out_channel=32, kernel=3, stride=2, padding=3//2)
        self.conv2 = BasicCBR(32, 64, 3, 2, 3//2)
        self.latent = nn.Linear(in_features=7 * 7 * 64, out_features=latent_dim)
        self.decoder_feature_size = (7, 7, 64)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        _, w, h, c = x.shape
        self.decoder_feature_size = (w, h, c)
        x = torch.flatten(x, 1)
        x = self.latent(x)
        return x


class Decoder(nn.Module):
    def __init__(self, in_feat_shape=(7, 7, 64), latent=16):
        super(Decoder, self).__init__()
        self.in_feat_shape = in_feat_shape
        self.linear = nn.Linear(in_features=latent, out_features=in_feat_shape[0] * in_feat_shape[1] * in_feat_shape[2])
        self.conv1 = BasicCBR(in_channel=in_feat_shape[2], out_channel=64, kernel=3, stride=1, padding=3//2)
        self.conv2 = BasicCBR(64, 32, kernel=3, stride=1, padding=3//2)
        self.out = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=3//2)

    def forward(self, x):
        x = self.linear(x)
        x = x.view(-1, self.in_feat_shape[2], self.in_feat_shape[0], self.in_feat_shape[1])
        x = self.conv1(x)
        x = F.interpolate(x, scale_factor=(2, 2))
        x = self.conv2(x)
        x = F.interpolate(x, scale_factor=(2, 2))
        x = torch.sigmoid(self.out(x))
        return x


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = Encoder(img_shape=(28, 28, 1))
        self.decoder = Decoder()

    def forward(self, x):
        latent = self.encoder(x)
        out = self.decoder(latent)
        return out
