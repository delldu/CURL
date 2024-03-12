"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2020-2024 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 09月 09日 星期三 23:56:45 CST
# ***
# ************************************************************************************/
#
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb


class CURLNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Define max GPU/CPU memory -- 4G
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 16
        # GPU 4G, 150ms

        self.tednet = TEDModel()
        self.curllayer = CURLLayer()
        self.load_weights()

    def load_weights(self, model_path="models/image_curve.pth"):
        cdir = os.path.dirname(__file__)
        checkpoint = model_path if cdir == "" else cdir + "/" + model_path
        self.load_state_dict(torch.load(checkpoint))


    def forward(self, img):
        B, C, H, W = img.size()
        pad_h = self.MAX_TIMES - (H % self.MAX_TIMES)
        pad_w = self.MAX_TIMES - (W % self.MAX_TIMES)
        img = F.pad(img, (0, pad_w, 0, pad_h), 'reflect')

        # img.size() -- [1, 3, 352, 512]
        feat = self.tednet(img)
        # feat.size() -- [1, 64, 352, 512]

        img = self.curllayer(feat)

        return img[:, :, 0:H, 0:W].clamp(0, 1.0)


class TEDModel(nn.Module):
    """
    TED -- Transformed Encoder Decoder
    """
    def __init__(self):
        super().__init__()
        self.ted = TED()
        self.final_conv = nn.Conv2d(3, 64, 3, 1, 0, 1)
        self.refpad = nn.ReflectionPad2d(1)

    def forward(self, img):
        output_img = self.ted(img)
        return self.final_conv(self.refpad(output_img)) # size() -- [1, 64, 352, 512]


def make_layer(nIn, nOut, k, s, p, d=1):
    return nn.Sequential(nn.Conv2d(nIn, nOut, k, s, p, d), nn.LeakyReLU())


class MidNet2(nn.Module):
    def __init__(self, in_channels=16):
        super().__init__()
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
        super().__init__()
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


class TED(nn.Module):
    """TED -- Transformed Encoder Decoder"""

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(16, 64, 1)
        self.conv2 = nn.Conv2d(32, 64, 1)
        self.conv3 = nn.Conv2d(64, 64, 1)

        self.mid_net2_1 = MidNet2()
        self.mid_net4_1 = MidNet4()
        self.local_net = LocalNet(16) # useless

        self.dconv_down1 = LocalNet(3, 16)
        self.dconv_down2 = LocalNet(16, 32)
        self.dconv_down3 = LocalNet(32, 64)
        self.dconv_down4 = LocalNet(64, 128)
        self.dconv_down5 = LocalNet(128, 128)

        self.maxpool = nn.MaxPool2d(kernel_size=2, padding=0)

        self.upsample = nn.UpsamplingNearest2d(scale_factor=2.0)
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
            nn.Flatten(start_dim=1),
            nn.Dropout(0.5),
            nn.Linear(64, 64),
        )

    def forward(self, x):
        # x.size() -- [1, 3, 341, 512]
        B, C, H, W = x.size()
        x_in_tile = x #.clone()

        conv1 = self.dconv_down1(x)
        x = self.maxpool(conv1)
        x = self.maxpool(self.dconv_down2(x))
        x = self.maxpool(self.dconv_down3(x))
        x = self.maxpool(self.dconv_down4(x))
        x = self.dconv_down5(x)
        x = self.up_conv1x1_1(self.upsample(x))

        x = self.dconv_up4(x)
        x = self.up_conv1x1_2(self.upsample(x))

        x = self.dconv_up3(x)
        x = self.up_conv1x1_3(self.upsample(x))

        x = self.dconv_up2(x)
        x = self.up_conv1x1_4(self.upsample(x))

        mid_features1 = self.mid_net2_1(conv1)
        mid_features2 = self.mid_net4_1(conv1)
        glob_features = self.glob_net1(conv1)
        glob_features = glob_features.unsqueeze(2).unsqueeze(3).repeat(1, 1, H, W)

        fuse = torch.cat((conv1, mid_features1, mid_features2, glob_features), dim=1)
        conv_fuse = self.conv_fuse1(fuse)

        x = torch.cat([x, conv_fuse], dim=1)
        x = self.dconv_up1(x)

        out = x + x_in_tile 

        return out # size() -- [1, 3, 341, 512]


def rgb_to_lab(img):
    # img.size() -- [3, 341, 512]
    C, H, W = img.size()

    img = img.permute(2, 1, 0).contiguous().view(W * H, 3)

    temp1 = img.le(0.04045).float()
    temp2 = 1.0 - temp1
    img = (img / 12.92) * temp1 + (((torch.clamp(img, min=0.0001) + 0.055) / 1.055) ** 2.4) * temp2

    rgb_to_xyz = torch.tensor(
        [  # X        Y          Z
            [0.412453, 0.212671, 0.019334],  # R
            [0.357580, 0.715160, 0.119193],  # G
            [0.180423, 0.072169, 0.950227],  # B
        ]
    ).to(img.device)

    img = torch.matmul(img, rgb_to_xyz)
    img = torch.mul(img, torch.tensor([1.0/0.950456, 1.0, 1.0/1.088754]).to(img.device))

    epsilon = 6.0 / 29.0
    epsilon2 = epsilon**2
    epsilon3 = epsilon**3

    temp1 = img.le(epsilon3).float()
    temp2 = 1.0 - temp1
    img = ((img / (3.0 * epsilon2) + 4.0/29.0) * temp1 + 
           (torch.clamp(img, min=0.0001) ** (1.0/3.0) * temp2))

    fxfyfz_to_lab = torch.tensor(
        [
            [0.0, 500.0, 0.0],  # fx
            [116.0, -500.0, 200.0],  # fy
            [0.0, 0.0, -200.0],  # fz
        ]
    ).to(img.device)

    img = torch.matmul(img, fxfyfz_to_lab) + torch.tensor([-16.0, 0.0, 0.0]).to(img.device)

    img = img.view(W, H, C).permute(2, 1, 0)

    """
    L_chan: black and white with input range [0, 100]
    a_chan/b_chan: color channels with input range ~[-110, 110], not exact 
    [0, 100] => [0, 1],  ~[-110, 110] => [0, 1]
    """
    img[0, :, :] = img[0, :, :] / 100.0
    img[1, :, :] = (img[1, :, :] / 110.0 + 1.0) / 2.0
    img[2, :, :] = (img[2, :, :] / 110.0 + 1.0) / 2.0

    return img.clamp(0.0, 1.0)


def lab_to_rgb(img):
    # img.size() -- [3, 341, 512]
    # img.min(), img.max() -- 0., 0.6350

    C, H, W = img.size()
    img = img.permute(2, 1, 0).contiguous().reshape(W * H, 3)

    img[:, 0] = img[:, 0] * 100.0
    img[:, 1] = ((img[:, 1] * 2.0) - 1.0) * 110.0
    img[:, 2] = ((img[:, 2] * 2.0) - 1.0) * 110.0

    lab_to_fxfyfz = torch.tensor(
        [  # X Y Z
            [1.0 / 116.0, 1.0 / 116.0, 1 / 116.0],  # R
            [1.0 / 500.0, 0.0, 0.0],  # G
            [0.0, 0.0, -1.0 / 200.0],  # B
        ]
    ).to(img.device)
    lab_base_offset = torch.tensor([16.0, 0.0, 0.0]).to(img.device)
    img = torch.matmul(img + lab_base_offset, lab_to_fxfyfz)

    epsilon = 6.0 / 29.0
    epsilon2 = epsilon**2

    temp1 = img.le(epsilon).float()
    temp2 = 1.0 - temp1
    img = ((3.0 * epsilon2 * (img - 4.0 / 29.0)) * temp1 + 
           (torch.clamp(img, min=0.0001) ** 3.0) * temp2)

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

    temp1 = img.le(0.0031308).float()
    temp2 = 1.0 - temp1
    img = (img * 12.92 * temp1) + ((torch.clamp(img, min=0.0001) ** (1.0 / 2.4) * 1.055) - 0.055) * temp2
    img = img.view(W, H, C).permute(2, 1, 0).contiguous()

    return img.clamp(0.0, 1.0)


def hsv_to_rgb(img):
    # img.size() -- [3, 341, 512]

    # img = img.clamp(0.0, 1.0)
    c0 = img[0, :, :]
    c1 = img[1, :, :]
    c2 = img[2, :, :]

    m = (c2 * (1.0 - c1) - c2) / 60.0
    r = (c2
        + torch.clamp(c0 * 360.0 - 60.0, 0.0, 60.0) * m
        - torch.clamp(c0 * 360.0 - 240.0, 0.0, 60.0) * m
    )

    g = (c2 * (1.0 - c1)
        - torch.clamp(c0 * 360.0 - 0.0, 0.0, 60.0) * m
        + torch.clamp(c0 * 360.0 - 180.0, 0.0, 60.0) * m
    )

    b = (c2 * (1.0 - c1)
        - torch.clamp(c0 * 360.0 - 120.0, 0.0, 60.0) * m
        + torch.clamp(c0 * 360.0 - 300.0, 0.0, 60.0) * m
    )

    img = torch.stack((r, g, b), dim=0)

    return img.clamp(0.0, 1.0)


def rgb_to_hsv(img):
    # img.size() -- [3, 341, 512]
    # img = img.clamp(0.0, 1.0).permute(2, 1, 0)
    img = img.permute(2, 1, 0)

    W, H, C = img.size() # [512, 352, 3]

    img = img.contiguous().view(W * H, 3) # size() -- [512 * 352, 3]

    mx = torch.max(img, dim=1)[0]
    mn = torch.min(img, dim=1)[0]

    H_2 = (W*H)//2  # half
    ones = torch.ones(W * H).to(img.device) # size() -- [180224]
    ones1 = ones[0 : H_2] # size() -- [90112]
    ones2 = ones[H_2 : W*H] # size() -- [90112]

    mx1 = mx[0 : H_2] # size() -- [90112]
    mx2 = mx[H_2 : W*H] # size() -- [90112]
    mn1 = mn[0 : H_2] # size() -- [90112]
    mn2 = mn[H_2 : W*H] # size() -- [90112]

    df1 = torch.add(mx1, torch.mul(ones1 * -1, mn1))
    df2 = torch.add(mx2, torch.mul(ones2 * -1, mn2))

    df = torch.cat((df1, df2), dim=0).to(img.device) # size() -- [180224]
    df = df.reshape(W, H) + 1e-10
    mx = mx.reshape(W, H) # size() -- [512, 352]

    img = img.reshape(W, H, C)

    r = img[:, :, 0]
    g = img[:, :, 1]
    b = img[:, :, 2]

    img[:, :, 0] = (
        ((g - b) / df) * r.eq(mx).float()
        + (2.0 + (b - r) / df) * g.eq(mx).float()
        + (4.0 + (r - g) / df) * b.eq(mx).float()
    )
    img[:, :, 0] = img[:, :, 0] * 60.0

    r = img[:, :, 0]
    zero = torch.zeros(W, H).to(img.device) # size() -- [512, 352]
    img[:, :, 0] = r.lt(zero).float() * (r + 360.0) + r.ge(zero).float() * r
    img[:, :, 0] = img[:, :, 0] / 360.0
    img[:, :, 1] = mx.ne(zero).float() * (df / mx) + mx.eq(zero).float() * (zero)
    img[:, :, 2] = mx
    img = img.permute(2, 1, 0)

    return img.clamp(0.0, 1.0)


def apply_curve(img, C, channel_in: int, channel_out: int):
    # img.size() -- [3, 352, 528]
    # C.size() ==== [16]

    curve_steps = 15 # C.shape[0] - 1 # ==== 15
    scale = torch.zeros_like(img[channel_in, :, :]) + C[0]
    for i in range(0, curve_steps - 1): # ==== 14
        # (C[i + 1] - C[i]) -- slope of the line segments
        scale += (C[i + 1] - C[i]) * (img[channel_in, :, :] * curve_steps - i)

    img[channel_out, :, :] = img[channel_out, :, :] * scale
    return img.clamp(0.0, 1.0)


def adjust_hsv(img, S):
    """
        Adjust the HSV channels of a HSV image using learnt curves
    """
    # S -- H[0, 0:64]
    img = img.squeeze(0)

    S_4 = S.shape[0] // 4
    S1 = torch.exp(S[0 : S_4])
    S2 = torch.exp(S[S_4 : S_4 * 2])
    S3 = torch.exp(S[S_4 * 2 : S_4 * 3])
    S4 = torch.exp(S[S_4 * 3 : S_4 * 4])

    # Apply the curve to the HSV channel 
    apply_curve(img, S1, channel_in=0, channel_out=0)
    apply_curve(img, S2, channel_in=0, channel_out=1)
    apply_curve(img, S3, channel_in=1, channel_out=1)
    apply_curve(img, S4, channel_in=2, channel_out=2)

    return img


def adjust_rgb(img, R):
    """
        Adjust the RGB channels of a RGB image using learnt curves
    """
    # R -- R[0, 0:48]
    img = img.squeeze(0)

    # Extract the parameters of the three curves
    R_3 = R.shape[0] // 3
    R1 = torch.exp(R[0 : R_3])
    R2 = torch.exp(R[R_3 : R_3 * 2])
    R3 = torch.exp(R[R_3 * 2 : R_3 * 3])

    # Apply the curve to the R/G/B channel 
    apply_curve(img, R1, channel_in=0, channel_out=0)
    apply_curve(img, R2, channel_in=1, channel_out=1)
    apply_curve(img, R3, channel_in=2, channel_out=2)

    return img


def adjust_lab(img, L):
    """
        Adjusts the image in LAB space using the predicted curves
    """
    # img.size() -- [3, 352, 528]
    # L -- L[0, 0:48]
    # Extract predicted parameters for each L,a,b curve
    L_3 = L.shape[0] // 3
    L1 = torch.exp(L[0 : L_3])
    L2 = torch.exp(L[L_3 : L_3 * 2])
    L3 = torch.exp(L[L_3 * 2 : L_3 * 3])

    # Apply the curve to the Lab channel 
    apply_curve(img, L1, channel_in=0, channel_out=0)
    apply_curve(img, L2, channel_in=1, channel_out=1)
    apply_curve(img, L3, channel_in=2, channel_out=2)

    return img


class LocalNet(nn.Module):
    def __init__(self, in_channels=16, out_channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 0, 1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 0, 1)
        self.lrelu = nn.LeakyReLU()
        self.refpad = nn.ReflectionPad2d(1)

    def forward(self, x_in):
        x = self.lrelu(self.conv1(self.refpad(x_in)))
        return self.lrelu(self.conv2(self.refpad(x)))


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        return self.lrelu(self.conv(x))


class MaxPoolBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.max_pool(x)


class GlobalPoolingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):
        return self.avg_pool(x)


class CURLLayer(nn.Module):
    def __init__(self, in_channels=64, out_channels=64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.make_init_network()

    def make_init_network(self):
        self.lab_layer1 = ConvBlock(64, 64)
        self.lab_layer2 = MaxPoolBlock()
        self.lab_layer3 = ConvBlock(64, 64)
        self.lab_layer4 = MaxPoolBlock()
        self.lab_layer5 = ConvBlock(64, 64)
        self.lab_layer6 = MaxPoolBlock()
        self.lab_layer7 = ConvBlock(64, 64)
        self.lab_layer8 = GlobalPoolingBlock()
        self.fc_lab = nn.Linear(64, 48)

        self.rgb_layer1 = ConvBlock(64, 64)
        self.rgb_layer2 = MaxPoolBlock()
        self.rgb_layer3 = ConvBlock(64, 64)
        self.rgb_layer4 = MaxPoolBlock()
        self.rgb_layer5 = ConvBlock(64, 64)
        self.rgb_layer6 = MaxPoolBlock()
        self.rgb_layer7 = ConvBlock(64, 64)
        self.rgb_layer8 = GlobalPoolingBlock()
        self.fc_rgb = nn.Linear(64, 48)

        self.hsv_layer1 = ConvBlock(64, 64)
        self.hsv_layer2 = MaxPoolBlock()
        self.hsv_layer3 = ConvBlock(64, 64)
        self.hsv_layer4 = MaxPoolBlock()
        self.hsv_layer5 = ConvBlock(64, 64)
        self.hsv_layer6 = MaxPoolBlock()
        self.hsv_layer7 = ConvBlock(64, 64)
        self.hsv_layer8 = GlobalPoolingBlock()
        self.fc_hsv = nn.Linear(64, 64)

    def forward(self, x):
        # x.size() -- [1, 64, 352, 512]
        img = x[:, 0:3, :, :]
        feat = x[:, 3:64, :, :]

        #######################################################
        # Step 1
        img_lab = rgb_to_lab(img.squeeze(0))

        feat_lab = torch.cat((feat, img_lab.unsqueeze(0)), dim=1)
        x = self.lab_layer1(feat_lab)
        x = self.lab_layer2(x)
        x = self.lab_layer3(x)
        x = self.lab_layer4(x)
        x = self.lab_layer5(x)
        x = self.lab_layer6(x)
        x = self.lab_layer7(x)
        x = self.lab_layer8(x)

        # x.size() -- [1, 64, 1, 1]
        B, C, H, W = x.size()
        x = x.view(B, C * H * W)

        L = self.fc_lab(x) # size() -- [1, 48]
        img_lab = adjust_lab(img_lab.squeeze(0), L[0, 0:48])

        #######################################################
        # Step 2
        img_rgb = lab_to_rgb(img_lab.squeeze(0))
        feat_rgb = torch.cat((feat, img_rgb.unsqueeze(0)), dim=1)

        x = self.rgb_layer1(feat_rgb)
        x = self.rgb_layer2(x)
        x = self.rgb_layer3(x)
        x = self.rgb_layer4(x)
        x = self.rgb_layer5(x)
        x = self.rgb_layer6(x)
        x = self.rgb_layer7(x)
        x = self.rgb_layer8(x)
        # x.size() -- [1, 64, 1, 1]
        B, C, H, W = x.size()
        x = x.view(B, C * H * W)
        R = self.fc_rgb(x)

        img_rgb = adjust_rgb(img_rgb.squeeze(0), R[0, 0:48])


        #######################################################
        # Step 3
        img_hsv = rgb_to_hsv(img_rgb.squeeze(0))
        feat_hsv = torch.cat((feat, img_hsv.unsqueeze(0)), dim=1)

        x = self.hsv_layer1(feat_hsv)
        x = self.hsv_layer2(x)
        x = self.hsv_layer3(x)
        x = self.hsv_layer4(x)
        x = self.hsv_layer5(x)
        x = self.hsv_layer6(x)
        x = self.hsv_layer7(x)
        x = self.hsv_layer8(x)
        B, C, H, W = x.size()
        x = x.view(B, C * H * W)
        H = self.fc_hsv(x)

        img_hsv = adjust_hsv(img_hsv, H[0, 0:64])

        #######################################################
        # Step 4
        img_residual = hsv_to_rgb(img_hsv.squeeze(0))

        img += img_residual.unsqueeze(0)

        return img.clamp(0.0, 1.0)
