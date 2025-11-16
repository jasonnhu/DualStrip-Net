import model.backbone.resnet as resnet
from model.backbone.xception import xception
from segmentation_models_pytorch import Unet
import numpy as np
from einops import rearrange
import torch
from torch import nn
import torch.nn.functional as F


class PatchNet(nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        self.cfg = cfg
        if cfg['model'] == 'deeplabv3plus':
            self.segnet = DeepLabV3Plus(cfg)
        if cfg['model'] == 'unet':
            self.segnet = Unet(encoder_name=cfg['backbone'], classes=2)

        self.aug = str(cfg.get('training_config', {}).get('use_aug', True)).lower() == 'true'

    def forward(self, x):
        # h, w = img.shape[-2:]
        if self.training:
            img_ori, img_u, img_u_s1, img_u_s2 = x
            h, w = img_ori.shape[-2:]
            img = torch.cat((img_ori, img_u))
            img_h = torch.cat((img_ori, img_u_s1))
            img_v = torch.cat((img_ori, img_u_s2))
        else:
            img = x

        if self.cfg['model'] == 'deeplabv3plus':
            fea = self.segnet(img)
            out = self.segnet.classifier(fea)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        if self.cfg['model'] == 'unet':
            fea = self.segnet.encoder(img)
            fea = self.segnet.decoder(*fea)
            out = self.segnet.segmentation_head(fea)
        
        if self.training:
            fea_h, fea_v, out_h, out_v = self._strip_forward(img_h, img_v)

            return out, fea_h, fea_v, out_h, out_v
        else:
            return out

    def _strip_forward(self, img_h, img_v):

        fea_h, out_h = self.forward_puzzles(img_h, p=0)
        fea_v, out_v = self.forward_puzzles(img_v, p=1)
        return fea_h, fea_v, out_h, out_v

    def puzzle(self, x, p, grid):
        if p == 0:
            fea = rearrange(x, 'b c (n h) (m w) -> (n m b) c h w', n=grid*2, m=2)
        if p == 1:
            fea = rearrange(x, 'b c (n h) (m w) -> (n m b) c h w', n=2, m=grid*2)
        return fea, p

    def recover_puzzle(self,x, grid, p):
        if p == 0:
            fea = rearrange(x, '(n m b) c h w -> b c (n h) (m w)', n=grid*2, m=2)
        if p == 1:
            fea = rearrange(x, '(n m b) c h w -> b c (n h) (m w)', n=2, m=grid*2)
        return fea

    def forward_puzzles(self, x, p):
        h, w = x.shape[-2:]
        if self.aug:
            C, B, H, W = x.shape
            data = [0, 32, 64]
            pad = 2*np.random.choice(data)
            if p == 0:
                padding = [0, 0, pad, pad]
            elif p == 1:
                padding = [pad, pad, 0, 0]
            x = F.pad(x, padding, 'constant', 0)

        else:
            pad = 0

        x, puzzle_types = self.puzzle(x, p, 2)
        if self.cfg['model'] == 'deeplabv3plus':
            fea = self.segnet(x)
        if self.cfg['model'] == 'unet':
            fea = self.segnet.encoder(x)
            fea = self.segnet.decoder(*fea)
        re_fea = self.recover_puzzle(fea, 2, puzzle_types)
        if self.cfg['model'] == 'deeplabv3plus':
            pad = int(pad / 4)
        if pad > 0:
            if p == 0:
                re_fea = re_fea[:, :, pad:-pad,:]
            elif p == 1:
                re_fea = re_fea[:, :, :, pad:-pad]

        if self.cfg['model'] == 'deeplabv3plus':
            out = self.segnet.classifier(re_fea)
            out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)
        if self.cfg['model'] == 'unet':
            out = self.segnet.segmentation_head(re_fea)

        return re_fea, out

class DeepLabV3Plus(nn.Module):
    def __init__(self, cfg):
        super(DeepLabV3Plus, self).__init__()

        if 'resnet' in cfg['backbone']:
            self.backbone = resnet.__dict__[cfg['backbone']](pretrained=True,
                                                             replace_stride_with_dilation=cfg['replace_stride_with_dilation'])
        else:
            assert cfg['backbone'] == 'xception'
            self.backbone = xception(pretrained=True)

        low_channels = 256
        high_channels = 2048

        self.head = ASPPModule(high_channels, cfg['dilations'])

        self.reduce = nn.Sequential(nn.Conv2d(low_channels, 48, 1, bias=False),
                                    nn.BatchNorm2d(48),
                                    nn.ReLU(True))

        self.fuse = nn.Sequential(nn.Conv2d(high_channels // 8 + 48, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True),
                                  nn.Conv2d(256, 256, 3, padding=1, bias=False),
                                  nn.BatchNorm2d(256),
                                  nn.ReLU(True))

        self.classifier = nn.Conv2d(256, cfg['nclass'], 1, bias=True)

    def forward(self, x, need_fp=False):

        h, w = x.shape[-2:]

        feats = self.backbone.base_forward(x)
        c1, c4 = feats[0], feats[-1]

        if need_fp:
            outs = self._decode(torch.cat((c1, nn.Dropout2d(0.5)(c1))),
                                torch.cat((c4, nn.Dropout2d(0.5)(c4))))
            outs = F.interpolate(outs, size=(h, w), mode="bilinear", align_corners=True)
            out, out_fp = outs.chunk(2)

            return out, out_fp

        feature = self._decode(c1, c4)
        # out = F.interpolate(out, size=(h, w), mode="bilinear", align_corners=True)

        return feature

    def _decode(self, c1, c4):
        c4 = self.head(c4)
        c4 = F.interpolate(c4, size=c1.shape[-2:], mode="bilinear", align_corners=True)

        c1 = self.reduce(c1)

        feature = torch.cat([c1, c4], dim=1)
        feature = self.fuse(feature)

        # out = self.classifier(feature)

        return feature


def ASPPConv(in_channels, out_channels, atrous_rate):
    block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                                    dilation=atrous_rate, bias=False),
                          nn.BatchNorm2d(out_channels),
                          nn.ReLU(True))
    return block


class ASPPPooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__()
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 nn.BatchNorm2d(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        h, w = x.shape[-2:]
        pool = self.gap(x)
        return F.interpolate(pool, (h, w), mode="bilinear", align_corners=True)


class ASPPModule(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPPModule, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = atrous_rates

        self.b0 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                nn.BatchNorm2d(out_channels),
                                nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1)
        self.b2 = ASPPConv(in_channels, out_channels, rate2)
        self.b3 = ASPPConv(in_channels, out_channels, rate3)
        self.b4 = ASPPPooling(in_channels, out_channels)

        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
                                     nn.BatchNorm2d(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        return self.project(y)