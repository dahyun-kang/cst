import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CorrTransformBlock(nn.Module):
    """
    Correlation transform block consisting of
    - self-attention
    - feed-forward
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True, heads=8, groups=4):
        super(CorrTransformBlock, self).__init__()
        self.attn = Attention(in_channels, out_channels, kernel_size, stride, padding, bias, heads, groups)
        self.ff = FeedForward(out_channels, groups)

    def forward(self, input):
        x, support_mask = input
        out = self.attn((x, support_mask))
        out = self.ff(out)
        return out, support_mask


class Attention(nn.Module):
    """
    - performs self-attention on correlation tokens of a query token index
    - takes [cls|img] correlation tokens
    - reduces img correlation tokens via avg pooling (RearrangeAndAvgpool)
    - returns [clf|seg] correlation tokens with reduced seg token length
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, bias=True, heads=8, groups=4):
        super(Attention, self).__init__()
        self.heads = heads
        '''
        Size of conv output = floor((input  + 2 * pad - kernel) / stride) + 1
        The second condition of `retain_dim` checks the spatial size consistency by setting input=output=0;
        Use this term with caution to check the size consistency for generic cases!
        '''
        retain_dim = in_channels == out_channels and math.floor((2 * padding - kernel_size) / stride) == -1
        hidden_channels = out_channels // 2
        assert hidden_channels % self.heads == 0, "out_channels should be divided by heads. (example: out_channels: 40, heads: 4)"

        ksz_q = (1, kernel_size, kernel_size)
        str_q = (1, stride, stride)
        pad_q = (0, padding, padding)

        self.avgpool = RearrangeAndAvgpool(ksz_q, str_q, pad_q)
        self.short_cut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            RearrangeAndAvgpool(ksz_q, str_q, pad_q),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        ) if not retain_dim else nn.Identity()

        self.qkv = nn.Conv2d(in_channels, hidden_channels * 3, kernel_size=1, stride=1, padding=0, bias=bias)

        self.agg = nn.Sequential(
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.ReLU(inplace=True)
        )
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, input):
        # x: [batch, dim, nqrytokens, nspttokens]
        x, support_mask = input

        x_ = self.short_cut(x)

        B, C, Q, S = x.shape
        H = W = int(math.sqrt(S - 1))

        qkv = self.qkv(x)
        # qkv: [qkv=3, batch, nhead, dim, nqrytokens, nspttokens]
        qkv = rearrange(qkv, 'b (x g c) t s -> x b g c t s', x=3, g=self.heads)
        # q_out: [batch, nhead, dim, nqrytokens, nspttokens]
        q_out, k_out, v_out = qkv[0], qkv[1], qkv[2]

        q_out = rearrange(q_out, 'b g c t s -> b (g c) t s')
        q_out = self.avgpool(q_out)
        q_out = rearrange(q_out, 'b (g c) t s -> b g c t s', g=self.heads)

        # out: [batch, nhead, nqrytokens, nspttokens from q_out, nspttokens from k_out]
        out = torch.einsum('b g c t l, b g c t m -> b g t l m', q_out, k_out)
        if support_mask is not None:
            out = self.attn_mask(out, support_mask, spatial_size=(H, W))
        out = F.softmax(out, dim=-1)
        out = torch.einsum('b g t l m, b g c t m -> b g c t l', out, v_out)
        out = rearrange(out, 'b g c t s -> b (g c) t s')
        out = self.agg(out)

        return self.out_norm(out + x_)

    def attn_mask(self, x, mask, spatial_size):
        assert mask is not None
        mask = F.interpolate(mask.float().unsqueeze(1), spatial_size, mode='bilinear', align_corners=True)
        mask = rearrange(mask, 'b 1 h w -> b 1 1 1 (h w)')
        # attention masking does not affect cls token
        cls_corr = x[:, :, :, :, :1]
        # attention masking on img tokens
        out = x[:, :, :, :, 1:].masked_fill_(mask == 0, -1e9)
        # [cls|img] tokens recombined
        return torch.cat([cls_corr, out], dim=-1)


class RearrangeAndAvgpool(nn.Module):
    """
    - reduces num of spt img tokens by 2D avg pooling
    - first rearranges the spt img tokens to 2D, applies avg pool, and flatten
    """
    def __init__(self, ksz_q, str_q, pad_q):
        super(RearrangeAndAvgpool, self).__init__()
        self.pool = nn.AvgPool3d(kernel_size=ksz_q, stride=str_q, padding=pad_q)

    def forward(self, x):
        h = w = int(math.sqrt(x.shape[-1] - 1))
        # cls token is not pooled
        cls_corr = x[:, :, :, 0].unsqueeze(-1)
        out = rearrange(x[:, :, :, 1:], 'b c t (h w) -> b c t h w', h=h, w=w)
        # img tokens are pooled
        out = self.pool(out)
        out = rearrange(out, 'b c t h w -> b c t (h w)')
        # [cls|img] tokens recombined
        return torch.cat([cls_corr, out], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, out_channels, groups=4, size=2):
        super(FeedForward, self).__init__()
        hidden_channels = out_channels // size
        self.ff = nn.Sequential(
            nn.Conv2d(out_channels, hidden_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.GroupNorm(groups, hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        )
        self.out_norm = nn.GroupNorm(groups, out_channels)

    def forward(self, x):
        x_ = x
        out = self.ff(x)
        return self.out_norm(out + x_)
