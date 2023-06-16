from functools import reduce
from operator import add

import torch

from einops import rearrange

from model.pl_module import FSCSModule
from model.module.cst import CorrelationTransformer
import model.backbone.dino.vision_transformer as vits
import torch.nn.functional as F
import torchvision.transforms.functional as tvF


class ClfSegTransformer(FSCSModule):
    def __init__(self, args):
        super(ClfSegTransformer, self).__init__(args)

        self.backbone = vits.__dict__['vit_small'](patch_size=8, num_classes=0)
        url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
        self.backbone.load_state_dict(state_dict, strict=True)
        self.nlayer = 12
        self.nhead = 6
        self.imgsize = args.imgsize
        self.sptsize = int(int(args.imgsize // 8) // 4)

        self.sup = args.sup
        self.backbone.eval()

        for k, v in self.backbone.named_parameters():
            v.requires_grad = False
        self.learner = CorrelationTransformer([self.nhead * self.nlayer], args.way)

    def forward(self, batch):
        '''
        query_img.shape : [bsz, 3, H, W]
        support_imgs.shape : [bsz, way, 3, H, W]
        support_masks.shape : [bsz, way, H, W]
        '''

        spt_img = rearrange(batch['support_imgs'].squeeze(2), 'b n c h w -> (b n) c h w')
        spt_mask = None if self.sup == 'pseudo' else rearrange(batch['support_masks'].squeeze(2), 'b n h w -> (b n) h w')
        qry_img = batch['query_img']

        with torch.no_grad():
            qry_feats = self.extract_dino_feats(qry_img, return_qkv=self.sup == 'pseudo')
            spt_feats = self.extract_dino_feats(spt_img, return_qkv=self.sup == 'pseudo')

            if self.sup == 'pseudo':
                qry_qkv, qry_feats = qry_feats
                spt_qkv, spt_feats = spt_feats
                qry_qkv = qry_qkv.repeat_interleave(self.args.way, dim=1)

                resize = (self.imgsize, self.imgsize) if self.training else (batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item())
                spt_mask, qry_mask = self.generate_pseudo_mask(qry_qkv, spt_qkv, batch['query_class_presence'].flatten(), thr=0.4, resize=resize)

                # save the outputs
                # pmask : pseudo-GT binary mask
                batch['query_pmask'] = qry_mask  # used only for avg-head-pseudo-mask training
                batch['support_pmasks'] = spt_mask  # only used for vis

            qry_feats = torch.stack(qry_feats, dim=1)
            spt_feats = torch.stack(spt_feats, dim=1)
            qry_feats = qry_feats.repeat_interleave(self.args.way, dim=0)

            # [batch, nlayer, (1+HW), dim]
            B, L, T, C = spt_feats.shape

            h = w = int(self.imgsize // 8)
            ch = int(C // self.nhead)

            qry_feat = qry_feats.reshape(B * L, T, C)[:, 1:, :] # 1-HW token: img tokens
            spt_feat = spt_feats.reshape(B * L, T, C)[:, 1:, :] # 1-HW token: img tokens
            spt_cls = spt_feats.reshape(B * L, T, C)[:, 0, :]   # 0-th token: cls token

            qry_feat = rearrange(qry_feat, 'b p (n c) -> b n p c', n=self.nhead, c=ch)

            # resize support features 50x50 -> 12x12 to reduce computation
            spt_feat = rearrange(spt_feat, 'b (h w) d -> b d h w', h=h, w=w)
            spt_feat = F.interpolate(spt_feat, (self.sptsize, self.sptsize), mode='bilinear', align_corners=True)
            spt_feat = rearrange(spt_feat, 'b (n c) h w -> b n (h w) c', n=self.nhead, c=ch)

            spt_cls = rearrange(spt_cls, 'b (n c) -> b n 1 c', n=self.nhead, c=ch)
            spt_feat = torch.cat([spt_cls, spt_feat], dim=2)

            qry_feat = F.normalize(qry_feat, p=2, dim=-1)
            spt_feat = F.normalize(spt_feat, p=2, dim=-1)

            headwise_corr = torch.einsum('b n q c, b n s c -> b n q s', qry_feat, spt_feat)
            headwise_corr = rearrange(headwise_corr, '(b l) n q s -> b (n l) q s', b=B, l=L)

        output_cls, output_masks = self.learner(headwise_corr, spt_mask)

        # BN, 2, H, W
        output_cls = output_cls.view(-1, self.way, 2)
        output_masks = self.upsample_logit_mask(output_masks, batch)
        output_masks = output_masks.view(-1, self.way, *output_masks.shape[1:])

        return output_cls, output_masks

    def extract_dino_feats(self, img, return_qkv=False):
        feat = self.backbone.get_intermediate_layers(img, n=self.nlayer, return_qkv=return_qkv)
        return feat

    def generate_pseudo_mask(self, qry_qkv, spt_qkv, class_gt, resize=(400, 400), thr=0.4,
                             qry_img=None, spt_img=None):
        # 0-th token: cls token
        # 1-HW token: img token
        # qry_qkv [qkv, batch, head, (1+HW), dim]
        _, B, N, L, C = qry_qkv.shape
        spt_cls = spt_qkv[0, :, :, 0, :]
        spt_key = spt_qkv[1, :, :, 1:, :]
        qry_key = qry_qkv[1, :, :, 1:, :]

        h = w = int(self.imgsize // 8)
        ch = int(C // self.nhead)

        qry_key = rearrange(qry_key, 'b n (h w) c -> b n h w c', h=h, w=w)
        spt_key = rearrange(spt_key, 'b n (h w) c -> b n h w c', h=h, w=w)

        qry_key = F.normalize(qry_key, p=2, dim=-1)
        spt_key = F.normalize(spt_key, p=2, dim=-1)
        spt_cls = F.normalize(spt_cls, p=2, dim=-1)

        cros_corr = torch.einsum('b n h w c, b n c -> b n h w', qry_key, spt_cls)
        self_corr = torch.einsum('b n h w c, b n c -> b n h w', spt_key, spt_cls)

        self_corr = self_corr.mean(dim=1, keepdim=True)
        self_corr = F.interpolate(self_corr, (self.imgsize, self.imgsize), mode='bilinear', align_corners=True).squeeze(1)
        self_corr_ret = (self_corr + 1.) * .5  # [-1, 1] -> [0, 1]

        cros_corr = cros_corr.mean(dim=1, keepdim=True)
        cros_corr = F.interpolate(cros_corr, resize, mode='bilinear', align_corners=True).squeeze(1)
        cros_corr_ret = (cros_corr + 1.) * .5  # [-1, 1] -> [0, 1]

        self_corr_ret = (self_corr_ret > thr).float()
        cros_corr_ret = (cros_corr_ret > thr).float()

        cros_corr_ret[class_gt.squeeze(-1) == False] = 0.

        return self_corr_ret, cros_corr_ret

    def upsample_logit_mask(self, logit_mask, batch):
        if self.training:
            spatial_size = batch['query_img'].shape[-2:]
        else:
            spatial_size = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
        return F.interpolate(logit_mask, spatial_size, mode='bilinear', align_corners=True)

    def compute_objective(self, output_cls, output_masks, gt_presence, gt_mask):
        ''' supports 1-way training '''
        logit_cls = torch.log_softmax(output_cls, dim=2).squeeze(1)
        logit_mask = torch.log_softmax(output_masks, dim=2).squeeze(1)
        cls_loss = F.nll_loss(logit_cls, gt_presence.long().squeeze(-1))
        seg_loss = F.nll_loss(logit_mask, gt_mask.long())
        return cls_loss * 0.1 + seg_loss

    def predict_cls(self, output_cls):
        with torch.no_grad():
            logit_cls = torch.softmax(output_cls, dim=2)
            pred_cls = logit_cls[:, :, 1] > 0.5
        return pred_cls

    def predict_mask(self, output_masks):
        with torch.no_grad():
            logit_seg = torch.softmax(output_masks, dim=2)
            max_fg_val, max_fg_idx = logit_seg[:, :, 1].max(dim=1)
            max_fg_idx = max_fg_idx + 1  # smallest idx should be 1
            max_fg_idx[max_fg_val < 0.5] = 0  # set it as bg
            pred_seg = max_fg_idx
        return pred_seg

    def predict_cls_seg(self, batch, nshot):
        logit_mask_agg = 0
        cls_score_agg = 0
        support_imgs = batch['support_imgs'].clone()
        support_masks = batch['support_masks'].clone()

        for s_idx in range(nshot):
            batch['support_imgs'] = support_imgs[:, :, s_idx]
            batch['support_masks'] = support_masks[:, :, s_idx]
            output_cls, output_masks = self.forward(batch)
            cls_score_agg += torch.softmax(output_cls, dim=2).clone()
            logit_mask_agg += torch.softmax(output_masks, dim=2).clone()

        pred_cls = self.predict_cls(cls_score_agg / float(nshot))
        pred_seg = self.predict_mask(logit_mask_agg / float(nshot))

        return pred_cls, pred_seg

    def train_mode(self):
        self.train()
        self.backbone.eval()  # to prevent BN from learning data statistics with exponential averaging

    def configure_optimizers(self):
        return torch.optim.Adam([{"params": self.parameters(), "lr": self.args.lr}])
