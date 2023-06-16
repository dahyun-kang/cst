""" Correlation Transformer """
import math
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from model.module.corrtrblock import CorrTransformBlock


class CorrelationTransformer(nn.Module):
    """
    Correlation transformer
    - two transformer layers
    - attentively reduce support img token length by pooling
    - classification head with 1x1 convs
    - segmentation head with 1x1 convs
    """
    def __init__(self, inch, way):
        super(CorrelationTransformer, self).__init__()
        self.way = way

        def make_building_attentive_block(in_channel, out_channels, kernel_sizes):
            building_block_layers = []
            for idx, (outch, ksz) in enumerate(zip(out_channels, kernel_sizes)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                padding = 0
                stride = ksz  # stride set to the same with pooling sizes
                building_block_layers.append(CorrTransformBlock(inch, outch, ksz, stride, padding))

            return nn.Sequential(*building_block_layers)

        self.corrtransformer = make_building_attentive_block(inch[0], [32, 128], [4, 3])

        modules = []
        for _ in range(2):
            modules.append(nn.Conv2d(128, 128, (1, 1), padding=(0, 0), bias=True))
            modules.append(nn.GroupNorm(4, 128))
            modules.append(nn.ReLU(inplace=True))

        self.linear = nn.Sequential(*modules)

        # classification and segmentation task heads
        self.decoder1_cls = nn.Sequential(nn.Conv2d(128, 128, (1, 1), padding=(0, 0), bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 64, (1, 1), padding=(0, 0), bias=True),
                                          nn.ReLU(inplace=True))

        self.decoder1_seg = nn.Sequential(nn.Conv2d(128, 128, (3, 3), padding=(1, 1), bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(128, 64, (3, 3), padding=(1, 1), bias=True),
                                           nn.ReLU(inplace=True))

        self.decoder2_cls = nn.Sequential(nn.Conv2d(64, 64, (1, 1), padding=(0, 0), bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 2, (1, 1), padding=(0, 0), bias=True))

        self.decoder2_seg = nn.Sequential(nn.Conv2d(64, 64, (3, 3), padding=(1, 1), bias=True),
                                          nn.ReLU(inplace=True),
                                          nn.Conv2d(64, 2, (3, 3), padding=(1, 1), bias=True))

    def forward(self, headwise_corr, spt_mask):

        interm_feat = self.corrtransformer((headwise_corr, spt_mask))[0]
        interm_feat = self.linear(interm_feat)
        H = W = int(math.sqrt(interm_feat.shape[2]))

        # reshaping to 2D
        in_cls, in_seg = interm_feat[:, :, :, 0], interm_feat[:, :, :, 1]
        in_cls = in_cls.view(*in_cls.shape[:2], H, W)
        in_seg = in_seg.view(*in_seg.shape[:2], H, W)

        # classification
        out_cls = self.decoder1_cls(in_cls)
        out_cls = F.avg_pool2d(out_cls, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))  # half-size
        out_cls = self.decoder2_cls(out_cls)
        out_cls = F.adaptive_avg_pool2d(out_cls, (1, 1)).squeeze(-1).squeeze(-1)

        # segmentation
        out_seg = self.decoder1_seg(in_seg)
        upsample_size = (out_seg.size(-1) * 2,) * 2
        out_seg = F.interpolate(out_seg, upsample_size, mode='bilinear', align_corners=True)
        out_masks = self.decoder2_seg(out_seg)

        # out_cls: [BN, 2] where N is numclass
        # output_masks: [BN, 2, H, W]
        return out_cls, out_masks
