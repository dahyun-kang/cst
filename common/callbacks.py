import os

import torch
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pprint import PrettyPrinter
from common import utils


class CustomProgressBar(TQDMProgressBar):
    """
    Progress bar & printing info every epoch
    """
    def __init__(self, args):
        super(CustomProgressBar, self).__init__()
        self.args = args

    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)

        if not self.args.nowandb and trainer.global_rank == 0:
            self.trainer._loggers[0].experiment.config.update(self.args)

        PrettyPrinter().pprint(vars(self.args))
        print(pl_module.learner)
        utils.print_param_count(pl_module)

        if not self.args.nowandb and not self.args.eval:
            trainer.logger.experiment.watch(pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        """
        This function is called when the both training and validation epochs end
        PL 1.6.5 assumes one train epoch = training dataset epoch + validation dataset epoch if any
        """
        super().on_train_epoch_end(trainer, pl_module)
        print('')

        for split in ['trn', 'val']:
            loss = trainer.callback_metrics[f'{split}/loss']
            miou = trainer.callback_metrics[f'{split}/miou']
            er   = trainer.callback_metrics[f'{split}/er']

            print(f'[{split}] ep: {trainer.current_epoch:>3}| {split}/loss: {loss:.3f} | {split}/miou: {miou:.3f} | {split}/er: {er:.3f}')

    def on_test_start(self, trainer, pl_module):
        super().on_test_start(trainer, pl_module)
        PrettyPrinter().pprint(vars(self.args))
        utils.print_param_count(pl_module)


class CustomCheckpoint(ModelCheckpoint):
    """
    Checkpoint load & save
    """
    def __init__(self, args):
        self.dirpath = os.path.join('logs', args.benchmark, f'fold{args.fold}', args.backbone, args.logpath, args.sup)
        '''
        if not args.eval and not args.resume:
            assert not os.path.exists(self.dirpath), f'{self.dirpath} already exists'
        '''
        self.filename = 'best_model'
        self.way = args.way
        self.monitor = 'val/miou'

        super(CustomCheckpoint, self).__init__(dirpath=self.dirpath,
                                               monitor=self.monitor,
                                               filename=self.filename,
                                               mode='max',
                                               verbose=True,
                                               save_last=True)
        # For evaluation, load best_model-v(k).cpkt where k is the max index
        if args.eval:
            self.modelpath = self.return_best_model_path(self.dirpath, self.filename)
            print('evaluating', self.modelpath)
        # For training, set the filename as best_model.ckpt
        # For resuming training, pytorch_lightning will automatically set the filename as best_model-v(k).ckpt
        else:
            self.modelpath = os.path.join(self.dirpath, self.filename + '.ckpt')
        self.lastmodelpath = os.path.join(self.dirpath, 'last.ckpt')

    def return_best_model_path(self, dirpath, filename):
        ckpt_files = os.listdir(dirpath)  # list of strings
        vers = [ckpt_file for ckpt_file in ckpt_files if filename in ckpt_file]
        vers.sort()
        # vers = ['best_model.ckpt'] or
        # vers = ['best_model-v1.ckpt', 'best_model-v2.ckpt', 'best_model.ckpt']
        best_model = vers[-1] if len(vers) == 1 else vers[-2]
        return os.path.join(self.dirpath, best_model)


class OnlineLogger(WandbLogger):
    """
    A wandb logger class that is customed with the experiment log path
    """
    def __init__(self, args):
        super(OnlineLogger, self).__init__(
            name=args.logpath,
            project=f'fscs-{args.benchmark}-{args.backbone}-{args.sup}',
            group=f'fold{args.fold}',
            log_model=False,
        )
