"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2020-2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 09日 星期三 23:56:45 CST
# ***
# ************************************************************************************/
#
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

import pdb


class CURLNet(nn.Module):
    def __init__(self):
        super(CURLNet, self).__init__()
        self.tednet = TEDModel()
        self.curllayer = CURLLayer()

    def forward(self, img):
        # img.size() -- [1, 3, 341, 512]

        feat = self.tednet(img)
        img, gradient_regulariser = self.curllayer(feat)

        # gradient_regulariser --[1.3809e-09]
        # gradient_regulariser.size() -- [1]

        # return img.clamp(0, 1.0), gradient_regulariser
        return img.clamp(0, 1.0)


class TEDModel(nn.Module):
    def __init__(self):
        super(TEDModel, self).__init__()

        self.ted = TED()
        self.final_conv = nn.Conv2d(3, 64, 3, 1, 0, 1)
        self.refpad = nn.ReflectionPad2d(1)

    def forward(self, img):
        output_img = self.ted(img.float())

        return self.final_conv(self.refpad(output_img))


def make_layer(nIn, nOut, k, s, p, d=1):
    return nn.Sequential(nn.Conv2d(nIn, nOut, k, s, p, d), nn.LeakyReLU(inplace=True))


class MidNet2(nn.Module):
    def __init__(self, in_channels=16):
        super(MidNet2, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 2, 2)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 2, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 2, 2)

    def forward(self, x_in):
        x = self.lrelu(self.conv1((x_in)))
        x = self.lrelu(self.conv2((x)))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)

        return x


class MidNet4(nn.Module):
    def __init__(self, in_channels=16):
        super(MidNet4, self).__init__()
        self.lrelu = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 4, 4)
        self.conv2 = nn.Conv2d(64, 64, 3, 1, 4, 4)
        self.conv3 = nn.Conv2d(64, 64, 3, 1, 4, 4)
        self.conv4 = nn.Conv2d(64, 64, 3, 1, 4, 4)

    def forward(self, x_in):
        x = self.lrelu(self.conv1((x_in)))
        x = self.lrelu(self.conv2((x)))
        x = self.lrelu(self.conv3(x))
        x = self.conv4(x)

        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)


class TED(nn.Module):
    def __init__(self):
        super(TED, self).__init__()

        self.conv1 = nn.Conv2d(16, 64, 1)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)

        self.mid_net2_1 = MidNet2(in_channels=16)
        self.mid_net4_1 = MidNet4(in_channels=16)
        self.local_net = LocalNet(16)

        self.dconv_down1 = LocalNet(3, 16)
        self.dconv_down2 = LocalNet(16, 32)
        self.dconv_down3 = LocalNet(32, 64)
        self.dconv_down4 = LocalNet(64, 128)
        self.dconv_down5 = LocalNet(128, 128)

        self.maxpool = nn.MaxPool2d(2, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2)
        self.up_conv1x1_1 = nn.Conv2d(128, 128, 1)
        self.up_conv1x1_2 = nn.Conv2d(64, 64, 1)
        self.up_conv1x1_3 = nn.Conv2d(32, 32, 1)
        self.up_conv1x1_4 = nn.Conv2d(16, 16, 1)

        self.dconv_up4 = LocalNet(128, 64)
        self.dconv_up3 = LocalNet(64, 32)
        self.dconv_up2 = LocalNet(32, 16)
        self.dconv_up1 = LocalNet(32, 3)

        self.conv_fuse1 = nn.Conv2d(208, 16, 1)

        self.glob_net1 = nn.Sequential(
            make_layer(16, 64, 3, 2, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            make_layer(64, 64, 3, 2, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            make_layer(64, 64, 3, 2, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            make_layer(64, 64, 3, 2, 1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            make_layer(64, 64, 3, 2, 1),
            nn.AdaptiveAvgPool2d(1),
            Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
        )
        # pdb.set_trace() ==> Here !!!

    def forward(self, x):
        # x.size() -- [1, 3, 341, 512]

        x_in_tile = x.clone()
        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        conv4 = self.dconv_down4(x)
        x = self.maxpool(conv4)
        x = self.dconv_down5(x)
        x = self.up_conv1x1_1(self.upsample(x))

        if x.shape[3] != conv4.shape[3] and x.shape[2] != conv4.shape[2]:
            x = F.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv4.shape[2]:
            x = F.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv4.shape[3]:
            x = F.pad(x, (1, 0, 0, 0))

        del conv4

        x = self.dconv_up4(x)
        x = self.up_conv1x1_2(self.upsample(x))

        if x.shape[3] != conv3.shape[3] and x.shape[2] != conv3.shape[2]:
            x = F.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv3.shape[2]:
            x = F.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv3.shape[3]:
            x = F.pad(x, (1, 0, 0, 0))

        x = self.dconv_up3(x)
        x = self.up_conv1x1_3(self.upsample(x))

        del conv3

        if x.shape[3] != conv2.shape[3] and x.shape[2] != conv2.shape[2]:
            x = F.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv2.shape[2]:
            x = F.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv2.shape[3]:
            x = F.pad(x, (1, 0, 0, 0))

        x = self.dconv_up2(x)
        x = self.up_conv1x1_4(self.upsample(x))

        del conv2

        mid_features1 = self.mid_net2_1(conv1)
        mid_features2 = self.mid_net4_1(conv1)
        glob_features = self.glob_net1(conv1)
        glob_features = glob_features.unsqueeze(2)
        glob_features = glob_features.unsqueeze(3)
        glob_features = glob_features.repeat(1, 1, mid_features1.shape[2], mid_features1.shape[3])
        fuse = torch.cat((conv1, mid_features1, mid_features2, glob_features), 1)
        conv1_fuse = self.conv_fuse1(fuse)

        if x.shape[3] != conv1.shape[3] and x.shape[2] != conv1.shape[2]:
            x = F.pad(x, (1, 0, 0, 1))
        elif x.shape[2] != conv1.shape[2]:
            x = F.pad(x, (0, 0, 0, 1))
        elif x.shape[3] != conv1.shape[3]:
            x = F.pad(x, (1, 0, 0, 0))

        x = torch.cat([x, conv1_fuse], dim=1)
        del conv1

        x = self.dconv_up1(x)

        out = x + x_in_tile
        # out.size() -- [1, 3, 341, 512]

        return out


def rgb_to_lab(img):
    """PyTorch implementation of RGB to LAB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
    :param img: image to be adjusted
    :returns: adjusted image
    """
    # img.size() -- [3, 341, 512]

    img = img.permute(2, 1, 0)
    shape = img.shape
    img = img.contiguous()
    img = img.view(-1, 3)

    img = (img / 12.92) * img.le(0.04045).float() + (((torch.clamp(img, min=0.0001) + 0.055) / 1.055) ** 2.4) * img.gt(
        0.04045
    ).float()

    rgb_to_xyz = torch.tensor(
        [  # X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169, 0.950227],  # B
        ]
    ).to(img.device)

    img = torch.matmul(img, rgb_to_xyz)
    img = torch.mul(img, torch.tensor([1 / 0.950456, 1.0, 1 / 1.088754]).to(img.device))

    epsilon = 6.0 / 29.0

    img = ((img / (3.0 * epsilon ** 2) + 4.0 / 29.0) * img.le(epsilon ** 3).float()) + (
        torch.clamp(img, min=0.0001) ** (1.0 / 3.0) * img.gt(epsilon ** 3).float()
    )

    fxfyfz_to_lab = torch.tensor(
        [
            [0.0, 500.0, 0.0],  # fx
            [116.0, -500.0, 200.0],  # fy
            [0.0, 0.0, -200.0],  # fz
        ]
    ).to(img.device)

    img = torch.matmul(img, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).to(img.device)

    img = img.view(shape)
    img = img.permute(2, 1, 0)

    """
    L_chan: black and white with input range [0, 100]
    a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
    [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
    """
    img[0, :, :] = img[0, :, :] / 100
    img[1, :, :] = (img[1, :, :] / 110 + 1) / 2
    img[2, :, :] = (img[2, :, :] / 110 + 1) / 2

    img[(img != img).detach()] = 0

    img = img.contiguous()
    # img.size() -- [3, 341, 512]

    return img


def lab_to_rgb(img):
    """PyTorch implementation of LAB to RGB conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py
    :param img: image to be adjusted
    :returns: adjusted image
    """
    # img.size() -- [3, 341, 512]
    # img.min(), img.max() -- 0., 0.6350

    img = img.permute(2, 1, 0)
    shape = img.shape
    img = img.contiguous()
    img = img.view(-1, 3)
    img_copy = img.clone()

    img_copy[:, 0] = img[:, 0] * 100
    img_copy[:, 1] = ((img[:, 1] * 2) - 1) * 110
    img_copy[:, 2] = ((img[:, 2] * 2) - 1) * 110

    img = img_copy.clone()
    del img_copy

    lab_to_fxfyfz = torch.tensor(
        [
            # X Y Z
            [1.0 / 116.0, 1.0 / 116.0, 1 / 116.0],  # R
            [1.0 / 500.0, 0.0, 0.0],  # G
            [0.0, 0.0, -1.0 / 200.0],  # B
        ]
    ).to(img.device)

    img = torch.matmul(img + torch.tensor([16.0, 0.0, 0.0]).to(img.device), lab_to_fxfyfz)

    epsilon = 6.0 / 29.0

    img = ((3.0 * epsilon ** 2 * (img - 4.0 / 29.0)) * img.le(epsilon).float()) + (
        (torch.clamp(img, min=0.0001) ** 3.0) * img.gt(epsilon).float()
    )

    # denormalize for D65 white point
    img = torch.mul(img, torch.tensor([0.950456, 1.0, 1.088754]).to(img.device))

    xyz_to_rgb = torch.tensor(
        [  # X Y Z
            [3.2404542, -0.9692660, 0.0556434],  # R
            [-1.5371385, 1.8760108, -0.2040259],  # G
            [-0.4985314, 0.0415560, 1.0572252],  # B
        ]
    ).to(img.device)

    img = torch.matmul(img, xyz_to_rgb)

    img = (img * 12.92 * img.le(0.0031308).float()) + (
        (torch.clamp(img, min=0.0001) ** (1 / 2.4) * 1.055) - 0.055
    ) * img.gt(0.0031308).float()

    img = img.view(shape)
    img = img.permute(2, 1, 0)

    img = img.contiguous()
    img[(img != img).detach()] = 0

    return img


def hsv_to_rgb(img):
    """Converts a HSV image to RGB
    PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

    :param img: HSV image
    :returns: RGB image
    """
    # img.size() -- [3, 341, 512]

    img = torch.clamp(img, 0, 1)
    img = img.permute(2, 1, 0)

    m1 = 0
    m2 = (img[:, :, 2] * (1 - img[:, :, 1]) - img[:, :, 2]) / 60
    m3 = 0
    m4 = -1 * m2
    m5 = 0

    r = (
        img[:, :, 2]
        + torch.clamp(img[:, :, 0] * 360.0 - 0.0, 0.0, 60.0) * m1
        + torch.clamp(img[:, :, 0] * 360.0 - 60.0, 0.0, 60.0) * m2
        + torch.clamp(img[:, :, 0] * 360.0 - 120.0, 0.0, 120.0) * m3
        + torch.clamp(img[:, :, 0] * 360.0 - 240.0, 0.0, 60.0) * m4
        + torch.clamp(img[:, :, 0] * 360.0 - 300.0, 0.0, 60.0) * m5
    )

    m1 = (img[:, :, 2] - img[:, :, 2] * (1.0 - img[:, :, 1])) / 60
    m2 = 0.0
    m3 = -1.0 * m1
    m4 = 0.0

    g = (
        img[:, :, 2] * (1 - img[:, :, 1])
        + torch.clamp(img[:, :, 0] * 360.0 - 0.0, 0.0, 60.0) * m1
        + torch.clamp(img[:, :, 0] * 360.0 - 60.0, 0.0, 120.0) * m2
        + torch.clamp(img[:, :, 0] * 360.0 - 180.0, 0.0, 60.0) * m3
        + torch.clamp(img[:, :, 0] * 360.0 - 240.0, 0.0, 120.0) * m4
    )

    m1 = 0.0
    m2 = (img[:, :, 2] - img[:, :, 2] * (1.0 - img[:, :, 1])) / 60.0
    m3 = 0.0
    m4 = -1.0 * m2

    b = (
        img[:, :, 2] * (1.0 - img[:, :, 1])
        + torch.clamp(img[:, :, 0] * 360.0 - 0.0, 0.0, 120.0) * m1
        + torch.clamp(img[:, :, 0] * 360.0 - 120.0, 0.0, 60.0) * m2
        + torch.clamp(img[:, :, 0] * 360.0 - 180.0, 0.0, 120.0) * m3
        + torch.clamp(img[:, :, 0] * 360.0 - 300.0, 0.0, 60.0) * m4
    )

    img = torch.stack((r, g, b), 2)
    img[(img != img).detach()] = 0

    img = img.permute(2, 1, 0)
    img = img.contiguous()
    img = torch.clamp(img, 0, 1)

    return img


def rgb_to_hsv(img):
    """Converts an RGB image to HSV
    PyTorch implementation of RGB to HSV conversion: https://docs.opencv.org/3.3.0/de/d25/imgproc_color_conversions.html
    Based roughly on a similar implementation here: http://code.activestate.com/recipes/576919-python-rgb-and-hsv-conversion/

    :param img: RGB image
    :returns: HSV image
    """
    # img.size() -- [3, 341, 512]

    img = torch.clamp(img, 1e-9, 1.0)

    img = img.permute(2, 1, 0)
    shape = img.shape

    img = img.contiguous()
    img = img.view(-1, 3)

    mx = torch.max(img, 1)[0]
    mn = torch.min(img, 1)[0]

    ones = torch.ones(img.shape[0]).to(img.device)
    zero = torch.zeros(shape[0:2]).to(img.device)

    img = img.view(shape)

    ones1 = ones[0 : math.floor((ones.shape[0] / 2))]
    ones2 = ones[math.floor(ones.shape[0] / 2) : (ones.shape[0])]

    mx1 = mx[0 : math.floor((ones.shape[0] / 2))]
    mx2 = mx[math.floor(ones.shape[0] / 2) : (ones.shape[0])]
    mn1 = mn[0 : math.floor((ones.shape[0] / 2))]
    mn2 = mn[math.floor(ones.shape[0] / 2) : (ones.shape[0])]

    df1 = torch.add(mx1, torch.mul(ones1 * -1, mn1))
    df2 = torch.add(mx2, torch.mul(ones2 * -1, mn2))

    df = torch.cat((df1, df2), dim=0)
    del df1, df2
    df = df.view(shape[0:2]) + 1e-10
    mx = mx.view(shape[0:2])

    img = img
    df = df.to(img.device)
    mx = mx.to(img.device)

    g = img[:, :, 1].clone()
    b = img[:, :, 2].clone()
    r = img[:, :, 0].clone()

    img_copy = img.clone()

    img_copy[:, :, 0] = (
        ((g - b) / df) * r.eq(mx).float()
        + (2.0 + (b - r) / df) * g.eq(mx).float()
        + (4.0 + (r - g) / df) * b.eq(mx).float()
    )
    img_copy[:, :, 0] = img_copy[:, :, 0] * 60.0

    zero = zero.to(img.device)
    img_copy2 = img_copy.clone()

    img_copy2[:, :, 0] = img_copy[:, :, 0].lt(zero).float() * (img_copy[:, :, 0] + 360) + img_copy[:, :, 0].ge(
        zero
    ).float() * (img_copy[:, :, 0])

    img_copy2[:, :, 0] = img_copy2[:, :, 0] / 360.0

    del img, r, g, b

    img_copy2[:, :, 1] = mx.ne(zero).float() * (df / mx) + mx.eq(zero).float() * (zero)
    img_copy2[:, :, 2] = mx

    img_copy2[(img_copy2 != img_copy2).detach()] = 0

    img = img_copy2.clone()

    img = img.permute(2, 1, 0)
    img = torch.clamp(img, 1e-9, 1)

    return img


def apply_curve(img, C, slope_sqr_diff, channel_in: int, channel_out: int) -> List[torch.Tensor]:
    """Applies a peicewise linear curve defined by a set of knot points to
    an image channel

    :param img: image to be adjusted
    :param C: predicted knot points of curve
    :returns: adjusted image
    """
    # C -- tensor([1.0102, 1.0090, 1.0077, 1.0064, 1.0052, 1.0039, 1.0026, 1.0014, 1.0001,
    #     0.9988, 0.9976, 0.9963, 0.9950, 0.9938, 0.9925, 0.9912],
    #    device='cuda:0')

    slope = torch.zeros((C.shape[0] - 1)).to(img.device)
    curve_steps = C.shape[0] - 1
    #  C.shape -- torch.Size([16])

    """
    Compute the slope of the line segments
    """
    for i in range(0, C.shape[0] - 1):
        slope[i] = C[i + 1] - C[i]

    """
    Compute the squared difference between slopes
    """
    for i in range(0, slope.shape[0] - 1):
        slope_sqr_diff += (slope[i + 1] - slope[i]) * (slope[i + 1] - slope[i])

    """
    Use predicted line segments to compute scaling factors for the channel
    """
    scale = torch.zeros_like(img[:, :, channel_in]) + float(C[0])
    for i in range(0, slope.shape[0] - 1):
        scale += float(slope[i]) * (img[:, :, channel_in] * curve_steps - i)

    img_copy = img.clone()

    img_copy[:, :, channel_out] = img[:, :, channel_out] * scale

    img_copy = torch.clamp(img_copy, 0.0, 1.0)

    return img_copy, slope_sqr_diff


def adjust_hsv(img, S) -> List[torch.Tensor]:
    """Adjust the HSV channels of a HSV image using learnt curves

    :param img: image to be adjusted
    :param S: predicted parameters of piecewise linear curves
    :returns: adjust image, regularisation term
    """
    img = img.squeeze(0).permute(2, 1, 0)
    shape = img.shape
    img = img.contiguous()

    S1 = torch.exp(S[0 : int(S.shape[0] / 4)])
    S2 = torch.exp(S[(int(S.shape[0] / 4)) : (int(S.shape[0] / 4) * 2)])
    S3 = torch.exp(S[(int(S.shape[0] / 4) * 2) : (int(S.shape[0] / 4) * 3)])
    S4 = torch.exp(S[(int(S.shape[0] / 4) * 3) : (int(S.shape[0] / 4) * 4)])

    slope_sqr_diff = torch.zeros(1).to(img.device)

    """
    Adjust Hue channel based on Hue using the predicted curve
    """
    img_copy, slope_sqr_diff = apply_curve(img, S1, slope_sqr_diff, channel_in=0, channel_out=0)

    """
    Adjust Saturation channel based on Hue using the predicted curve
    """
    img_copy, slope_sqr_diff = apply_curve(img_copy, S2, slope_sqr_diff, channel_in=0, channel_out=1)

    """
    Adjust Saturation channel based on Saturation using the predicted curve
    """
    img_copy, slope_sqr_diff = apply_curve(img_copy, S3, slope_sqr_diff, channel_in=1, channel_out=1)

    """
    Adjust Value channel based on Value using the predicted curve
    """
    img_copy, slope_sqr_diff = apply_curve(img_copy, S4, slope_sqr_diff, channel_in=2, channel_out=2)

    img = img_copy.clone()
    del img_copy

    img[(img != img).detach()] = 0

    img = img.permute(2, 1, 0)
    img = img.contiguous()

    return img, slope_sqr_diff


def adjust_rgb(img, R) -> List[torch.Tensor]:
    """Adjust the RGB channels of a RGB image using learnt curves

    :param img: image to be adjusted
    :param S: predicted parameters of piecewise linear curves
    :returns: adjust image, regularisation term
    """
    img = img.squeeze(0).permute(2, 1, 0)
    shape = img.shape
    img = img.contiguous()

    """
    Extract the parameters of the three curves
    """
    R1 = torch.exp(R[0 : int(R.shape[0] / 3)])
    R2 = torch.exp(R[(int(R.shape[0] / 3)) : (int(R.shape[0] / 3) * 2)])
    R3 = torch.exp(R[(int(R.shape[0] / 3) * 2) : (int(R.shape[0] / 3) * 3)])

    """
    Apply the curve to the R channel 
    """
    slope_sqr_diff = torch.zeros(1).to(img.device)

    img_copy, slope_sqr_diff = apply_curve(img, R1, slope_sqr_diff, channel_in=0, channel_out=0)

    """
    Apply the curve to the G channel 
    """
    img_copy, slope_sqr_diff = apply_curve(img_copy, R2, slope_sqr_diff, channel_in=1, channel_out=1)

    """
    Apply the curve to the B channel 
    """
    img_copy, slope_sqr_diff = apply_curve(img_copy, R3, slope_sqr_diff, channel_in=2, channel_out=2)

    img = img_copy.clone()
    del img_copy

    img[(img != img).detach()] = 0

    img = img.permute(2, 1, 0)
    img = img.contiguous()

    return img, slope_sqr_diff


def adjust_lab(img, L) -> List[torch.Tensor]:
    """Adjusts the image in LAB space using the predicted curves

    :param img: Image tensor
    :param L: Predicited curve parameters for LAB channels
    :returns: adjust image, and regularisation parameter
    """
    img = img.permute(2, 1, 0)

    shape = img.shape
    img = img.contiguous()

    """
    Extract predicted parameters for each L,a,b curve
    """
    L1 = torch.exp(L[0 : int(L.shape[0] / 3)])
    L2 = torch.exp(L[(int(L.shape[0] / 3)) : (int(L.shape[0] / 3) * 2)])
    L3 = torch.exp(L[(int(L.shape[0] / 3) * 2) : (int(L.shape[0] / 3) * 3)])

    slope_sqr_diff = torch.zeros(1).to(img.device)

    """
    Apply the curve to the L channel 
    """
    img_copy, slope_sqr_diff = apply_curve(img, L1, slope_sqr_diff, channel_in=0, channel_out=0)

    """
    Now do the same for the a channel
    """
    img_copy, slope_sqr_diff = apply_curve(img_copy, L2, slope_sqr_diff, channel_in=1, channel_out=1)

    """
    Now do the same for the b channel
    """
    img_copy, slope_sqr_diff = apply_curve(img_copy, L3, slope_sqr_diff, channel_in=2, channel_out=2)

    img = img_copy.clone()
    del img_copy

    img[(img != img).detach()] = 0

    img = img.permute(2, 1, 0)
    img = img.contiguous()

    return img, slope_sqr_diff


class LocalNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=64):
        super(LocalNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 0, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 0, 1)
        self.lrelu = nn.LeakyReLU()
        self.refpad = nn.ReflectionPad2d(1)

    def forward(self, x_in):
        x = self.lrelu(self.conv1(self.refpad(x_in)))
        x = self.lrelu(self.conv2(self.refpad(x)))
        return x

class ConvBlock(nn.Module):
    def __init__(self, num_in_channels, num_out_channels, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(num_in_channels, num_out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        img_out = self.lrelu(self.conv(x))
        return img_out


class MaxPoolBlock(nn.Module):
    def __init__(self):
        super(MaxPoolBlock, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        img_out = self.max_pool(x)
        return img_out


class GlobalPoolingBlock(nn.Module):
    def __init__(self, receptive_field):
        super(GlobalPoolingBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        out = self.avg_pool(x)
        return out


class CURLLayer(nn.Module):
    def __init__(self, num_in_channels=64, num_out_channels=64):
        super(CURLLayer, self).__init__()

        self.num_in_channels = num_in_channels
        self.num_out_channels = num_out_channels
        self.make_init_network()

    def make_init_network(self):
        self.lab_layer1 = ConvBlock(64, 64)
        self.lab_layer2 = MaxPoolBlock()
        self.lab_layer3 = ConvBlock(64, 64)
        self.lab_layer4 = MaxPoolBlock()
        self.lab_layer5 = ConvBlock(64, 64)
        self.lab_layer6 = MaxPoolBlock()
        self.lab_layer7 = ConvBlock(64, 64)
        self.lab_layer8 = GlobalPoolingBlock(2)

        self.fc_lab = nn.Linear(64, 48)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.dropout3 = nn.Dropout(0.5)

        self.rgb_layer1 = ConvBlock(64, 64)
        self.rgb_layer2 = MaxPoolBlock()
        self.rgb_layer3 = ConvBlock(64, 64)
        self.rgb_layer4 = MaxPoolBlock()
        self.rgb_layer5 = ConvBlock(64, 64)
        self.rgb_layer6 = MaxPoolBlock()
        self.rgb_layer7 = ConvBlock(64, 64)
        self.rgb_layer8 = GlobalPoolingBlock(2)

        self.fc_rgb = nn.Linear(64, 48)

        self.hsv_layer1 = ConvBlock(64, 64)
        self.hsv_layer2 = MaxPoolBlock()
        self.hsv_layer3 = ConvBlock(64, 64)
        self.hsv_layer4 = MaxPoolBlock()
        self.hsv_layer5 = ConvBlock(64, 64)
        self.hsv_layer6 = MaxPoolBlock()
        self.hsv_layer7 = ConvBlock(64, 64)
        self.hsv_layer8 = GlobalPoolingBlock(2)

        self.fc_hsv = nn.Linear(64, 64)

    def forward(self, x) -> List[torch.Tensor]:
        x.contiguous()  # remove memory holes

        img = x[:, 0:3, :, :]
        feat = x[:, 3:64, :, :]

        shape = x.shape

        img_clamped = torch.clamp(img, 0.0, 1.0)
        img_lab = torch.clamp(rgb_to_lab(img_clamped.squeeze(0)), 0.0, 1.0)

        feat_lab = torch.cat((feat, img_lab.unsqueeze(0)), dim=1)

        x = self.lab_layer1(feat_lab)
        del feat_lab
        x = self.lab_layer2(x)
        x = self.lab_layer3(x)
        x = self.lab_layer4(x)
        x = self.lab_layer5(x)
        x = self.lab_layer6(x)
        x = self.lab_layer7(x)
        x = self.lab_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout1(x)
        L = self.fc_lab(x)

        img_lab, gradient_regulariser_lab = adjust_lab(img_lab.squeeze(0), L[0, 0:48])
        img_rgb = lab_to_rgb(img_lab.squeeze(0))
        img_rgb = torch.clamp(img_rgb, 0.0, 1.0)

        feat_rgb = torch.cat((feat, img_rgb.unsqueeze(0)), dim=1)

        x = self.rgb_layer1(feat_rgb)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        x = self.rgb_layer5(x)
        x = self.rgb_layer6(x)
        x = self.rgb_layer7(x)
        x = self.rgb_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout2(x)
        R = self.fc_rgb(x)

        img_rgb, gradient_regulariser_rgb = adjust_rgb(img_rgb.squeeze(0), R[0, 0:48])
        img_rgb = torch.clamp(img_rgb, 0.0, 1.0)

        img_hsv = rgb_to_hsv(img_rgb.squeeze(0))
        img_hsv = torch.clamp(img_hsv, 0.0, 1.0)
        feat_hsv = torch.cat((feat, img_hsv.unsqueeze(0)), 1)

        x = self.hsv_layer1(feat_hsv)
        del feat_hsv
        x = self.hsv_layer2(x)
        x = self.hsv_layer3(x)
        x = self.hsv_layer4(x)
        x = self.hsv_layer5(x)
        x = self.hsv_layer6(x)
        x = self.hsv_layer7(x)
        x = self.hsv_layer8(x)
        x = x.view(x.size()[0], -1)
        x = self.dropout3(x)
        H = self.fc_hsv(x)

        img_hsv, gradient_regulariser_hsv = adjust_hsv(img_hsv, H[0, 0:64])
        img_hsv = torch.clamp(img_hsv, 0.0, 1.0)

        img_residual = torch.clamp(hsv_to_rgb(img_hsv.squeeze(0)), 0.0, 1.0)

        img = torch.clamp(img + img_residual.unsqueeze(0), 0.0, 1.0)

        gradient_regulariser = gradient_regulariser_rgb + gradient_regulariser_lab + gradient_regulariser_hsv

        return img, gradient_regulariser
