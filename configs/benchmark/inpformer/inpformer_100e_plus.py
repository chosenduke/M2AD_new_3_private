from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

from configs.__base__ import *


class cfg(cfg_common, cfg_dataset_default, cfg_model_dinomaly):
    def __init__(self):
        cfg_common.__init__(self)
        cfg_dataset_default.__init__(self)
        cfg_model_dinomaly.__init__(self)

        self.vis = True
        self.fvcore_b = 1
        self.fvcore_c = 3
        self.seed = 42

        self.epoch_full = 100
        self.warmup_epochs = 0
        self.test_start_epoch = self.epoch_full
        self.test_per_epoch = max(1, self.epoch_full // 2)
        self.batch_train = 12
        self.batch_test_per = 12

        self.lr = 0.004 * self.batch_train / 16
        self.weight_decay = 0.05

        ###### Train DATA
        # 默认单类，建议按需修改
        self.train_data.cls_names = ['Ring']
        self.test_data.cls_names = ['Ring']

        if self.size == 256:
            self.size = 224
        elif self.size == 512:
            self.size = 448

        self.model.name = 'inpformer'
        # self.model.kwargs = dict(
        #     pretrained=False,
        #     checkpoint_path='',
        #     strict=False,
        #     encoder_arch='dinov2reg_vit_base_14',
        #     INP_num=6,
        #     # 新增可调参数
        #     aggregation_depth=2,
        #     decoder_depth=8,
        #     drop=0.05,
        #     drop_path=0.05,
        #     learnable_fuse=True,
        #     gather_weight=1.0,
        #     diversity_weight=0.05,
        # )
        #v5
        self.model.kwargs = dict(
            pretrained=False,
            checkpoint_path='',
            strict=False,
            encoder_arch='dinov2reg_vit_base_14',
            INP_num=6,
            # 新增可调参数
            aggregation_depth=2,
            decoder_depth=8,
            drop=0.05,
            drop_path=0.05,
            learnable_fuse=True,
            gather_weight=1,
            diversity_weight=0.06,
        )



        self.optim.lr = self.lr
        self.optim.kwargs = dict(name='adam', betas=(0.5, 0.999))

        # ==> trainer
        self.trainer.name = 'INPFormerTrainer'
        self.trainer.logdir_sub = 'plus'
        self.trainer.resume_dir = ''
        self.trainer.epoch_full = self.epoch_full
        self.trainer.scheduler_kwargs = dict(
            name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
            warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs,
            cooldown_epochs=0, use_iters=True,
            patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8),
            cycle_decay=0.1, decay_rate=0.1)

        self.trainer.test_start_epoch = self.test_start_epoch
        self.trainer.test_per_epoch = self.test_per_epoch

        self.trainer.data.batch_size = self.batch_train
        self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

        self.train_data.train_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.train_data.test_transforms = self.train_data.train_transforms
        self.train_data.target_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
        ]

        self.test_data.train_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
            dict(type='Normalize', mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, inplace=True),
        ]
        self.test_data.test_transforms = self.test_data.train_transforms
        self.test_data.target_transforms = [
            dict(type='Resize', size=(self.size, self.size), interpolation=F.InterpolationMode.BILINEAR),
            dict(type='CenterCrop', size=(self.size, self.size)),
            dict(type='ToTensor'),
        ]

        # ==> loss
        self.loss.loss_terms = [
            dict(type='INPFormerLoss', name='inpformer_loss', y=3),
        ]

        # ==> logging
        self.logging.log_terms_train = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='data_t', fmt=':>5.3f'),
            dict(name='optim_t', fmt=':>5.3f'),
            dict(name='lr', fmt=':>7.6f'),
            dict(name='inpformer_loss', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]
        self.logging.log_terms_test = [
            dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
            dict(name='inpformer_loss', suffixes=[''], fmt=':>5.3f', add_name='avg'),
        ]


