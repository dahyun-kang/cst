import abc
import math
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F

from common import utils
from common.evaluation import AverageMeter


class FSCSModule(pl.LightningModule, metaclass=abc.ABCMeta):
    """
    Few-Shot Classification and Segmentation PL wrapper module
    """
    def __init__(self, args):
        super(FSCSModule, self).__init__()

        self.args = args
        self.way = args.way
        self.sup = args.sup
        self.range = torch.arange(args.way + 1, requires_grad=False).view(1, args.way + 1, 1, 1)
        self.learner = None
        self.avg_meter = {'trn' : None, 'val' : None}

    @abc.abstractmethod
    def forward(self, batch):
        pass

    @abc.abstractmethod
    def train_mode(self):
        pass

    @abc.abstractmethod
    def configure_optimizers(self):
        pass

    @abc.abstractmethod
    def predict_cls_seg(self, batch, nshot):
        pass

    def on_train_epoch_start(self):
        utils.fix_randseed(None)
        # PyTorch 1.12 ver issue; should assign capturable = True for rerun
        # https://github.com/pytorch/pytorch/issues/80831
        if self.trainer.rerun:
            self.trainer.optimizers[0].param_groups[0]['capturable'] = True
        self.avg_meter['trn'] = AverageMeter(self.trainer.train_dataloader.dataset.datasets, self.args.way)
        self.train_mode()

    def on_validation_epoch_start(self):
        self._shared_eval_epoch_start(self.trainer.val_dataloaders[0].dataset)

    def on_test_epoch_start(self):
        self._shared_eval_epoch_start(self.trainer.test_dataloaders[0].dataset)

    def _shared_eval_epoch_start(self, dataset):
        utils.fix_randseed(0)
        self.avg_meter['val'] = AverageMeter(dataset, self.args.way)
        self.eval()

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, 'trn')

    def _shared_step(self, batch, batch_idx, split):
        """
        batch.keys()
        > dict_keys(['query_img', 'query_mask', 'query_name', 'query_ignore_idx', 'org_query_imsize', 'support_imgs', 'support_masks', 'support_names', 'support_ignore_idxs', 'class_id'])

        batch['query_img'].shape : [bsz, 3, H, W]
        batch['query_mask'].shape : [bsz, H, W]
        batch['query_name'].len : [bsz]
        batch['query_ignore_idx'].shape : [bsz, H, W]
        batch['query_ignore_idx'].shape : [bsz, H, W]
        batch['org_query_imsize'].len : [bsz]
        batch['support_imgs'].shape : [bsz, way, shot, 3, H, W]
        batch['support_masks'].shape : [bsz, way, shot, H, W]
        # FYI: this support_names' shape is transposed so keep in mind for vis
        batch['support_names'].shape : [bsz, shot, way]
        batch['support_ignore_idxs'].shape: [bsz, way, shot, H, W]
        batch['class_id'].shape : [bsz]
        batch['support_classes'].shape : [bsz, way] (torch.int64)
        batch['query_class_presence'].shape : [bsz, way] (torch.bool)
        # FYI: K-shot is always fixed to 1 for training
        """

        output_cls, output_masks = self.forward(batch)

        pred_cls = self.predict_cls(output_cls)
        pred_seg = self.predict_mask(output_masks)

        if self.sup == 'pseudo':
            loss = self.compute_objective(output_cls, output_masks, batch['query_class_presence'], batch['query_pmask'])
        elif self.sup == 'mask':
            loss = self.compute_objective(output_cls, output_masks, batch['query_class_presence'], batch['query_mask'])

        with torch.no_grad():
            self.avg_meter[split].update_cls(pred_cls, batch['query_class_presence'])
            self.avg_meter[split].update_seg(pred_seg, batch, loss.item())

            self.log(f'{split}/loss', loss, on_step=True, on_epoch=False, prog_bar=False, logger=False)
        return loss

    def training_epoch_end(self, training_step_outputs):
        self._shared_epoch_end(training_step_outputs, 'trn')

    def validation_step(self, batch, batch_idx):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self._shared_step(batch, batch_idx, 'val')

    def validation_epoch_end(self, validation_step_outputs):
        # model.eval() and torch.no_grad() are called automatically for validation
        # in pytorch_lightning
        self._shared_epoch_end(validation_step_outputs, 'val')

    def _shared_epoch_end(self, steps_outputs, split):
        miou = self.avg_meter[split].compute_iou()
        er = self.avg_meter[split].compute_cls_er()
        loss = self.avg_meter[split].avg_seg_loss()

        dict = {f'{split}/loss': loss,
                f'{split}/miou': miou,
                f'{split}/er': er}

        for k in dict:
            self.log(k, dict[k], on_epoch=True, logger=True)

        # Moved to common/callback.py due to the pl ver. update
        # self.print(f'{space}[{split}] ep: {self.current_epoch:>3}| {split}/loss: {loss:.3f} | {split}/miou: {miou:.3f} | {split}/er: {er:.3f}')

    def test_step(self, batch, batch_idx):

        pred_cls, pred_seg = self.predict_cls_seg(batch, self.args.shot)
        er_b = self.avg_meter['val'].update_cls(pred_cls, batch['query_class_presence'], loss=None)
        iou_b = self.avg_meter['val'].update_seg(pred_seg, batch, loss=None)

        if self.args.vis:
            print(batch_idx, 'qry:', batch['query_name'])
            print(batch_idx, 'spt:', batch['support_names'])
            if self.args.shot > 1: raise NotImplementedError
            if self.sup == 'mask':
                support_masks = batch['support_masks']
            elif self.sup == 'pseudo':
                support_masks = batch['support_pmasks'].unsqueeze(0)
                # rename 0 or 1 binary label to 1-N label to color them
                for c in range(self.way):
                    spt_mask_c = support_masks[:, c]
                    spt_mask_c[spt_mask_c > 0] = c + 1
                    support_masks[:, c] = spt_mask_c
            from common.vis import Visualizer
            Visualizer.initialize(True, self.way, path='./vis_results/')
            Visualizer.visualize_prediction_batch(batch['support_imgs'].squeeze(2),
                                                  support_masks.squeeze(2),
                                                  batch['query_img'],
                                                  batch['query_mask'],
                                                  batch['org_query_imsize'],
                                                  pred_seg,
                                                  batch_idx,
                                                  iou_b=iou_b,
                                                  er_b=er_b,
                                                  to_cpu=True)

    def test_epoch_end(self, test_step_outputs):
        miou = self.avg_meter['val'].compute_iou()
        er = self.avg_meter['val'].compute_cls_er()

        dict = {'test/miou': miou.item(),
                'test/er': er.item()}

        for k in dict:
            self.log(k, dict[k], on_epoch=True)
