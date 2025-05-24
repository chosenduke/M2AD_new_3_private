from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F
from configs.__base__ import *
from configs.__base__.cfg_dataset_default import cfg_dataset_default
from data.dataset_info import DATA_ROOT

class cfg(cfg_common, cfg_dataset_default, cfg_model_rdpp):

	def __init__(self):
		cfg_common.__init__(self)
		cfg_dataset_default.__init__(self)
		cfg_model_rdpp.__init__(self)

		self.vis = True
		self.fvcore_b = 1
		self.fvcore_c = 3
		self.seed = 42

		self.epoch_full = 100
		self.warmup_epochs = 0
		self.test_start_epoch = self.epoch_full
		self.test_per_epoch = self.epoch_full // 2

		self.batch_train = 8
		self.batch_test_per = 8
		self.lr = 0.0005 * self.batch_train / 16
		self.weight_decay = 0.05

		self.use_adeval = True

		###### Train DATA
		self.train_data.cls_names = ['Ring']
		self.test_data.cls_names = ['Ring']

		self.train_data.anomaly_generator.name = 'draem'
		self.train_data.anomaly_generator.kwargs = dict(anomaly_source_path=f"{DATA_ROOT}/dtd/images")
		self.train_data.anomaly_generator.enable = True ### TODO
		self.train_data.sampler.name = 'naive'
		self.train_data.sampler.kwargs = dict()

		# ==> model
		self.model_t = Namespace()
		self.model_t.name = 'timm_wide_resnet50_2'
		self.model_t.kwargs = dict(pretrained=True, checkpoint_path='',
							  strict=False, features_only=True, out_indices=[1, 2, 3])
		self.model_s = Namespace()
		self.model_s.name = 'de_wide_resnet50_2'
		self.model_s.kwargs = dict(pretrained=False, checkpoint_path='', strict=False)
		self.model.name = 'rdpp'
		self.model.kwargs = dict(pretrained=False, checkpoint_path='', strict=False, model_t=self.model_t, model_s=self.model_s)


		# ==> evaluator
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100, use_adeval=self.use_adeval)

		# ==> optimizer
		self.optim.proj_opt = Namespace()
		self.optim.distill_opt = Namespace()
		self.optim.lr = self.lr
		self.optim.proj_opt.kwargs = dict(name='adam', betas=(0.5, 0.999))
		self.optim.distill_opt.kwargs = dict(name='adam', betas=(0.5, 0.999))

		# ==> trainer
		self.trainer.name = 'RDPPTrainer'
		self.trainer.logdir_sub = 'exp'
		self.trainer.resume_dir = ''
		self.trainer.epoch_full = self.epoch_full
		self.trainer.scheduler_kwargs = dict(
			name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=self.lr / 1e2,
			warmup_lr=self.lr / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=self.warmup_epochs, cooldown_epochs=0, use_iters=True,
			patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8), cycle_decay=0.1, decay_rate=0.1)

		self.trainer.test_start_epoch = self.test_start_epoch
		self.trainer.test_per_epoch = self.test_per_epoch

		self.trainer.data.batch_size = self.batch_train
		self.trainer.data.batch_size_per_gpu_test = self.batch_test_per

		# ==> loss
		self.loss.loss_terms = [
			dict(type='CosLoss', name='cos', avg=False, lam=1.0),
		]

		# ==> logging
		self.logging.log_terms_train = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='data_t', fmt=':>5.3f'),
			dict(name='optim_t', fmt=':>5.3f'),
			dict(name='lr', fmt=':>7.6f'),
			dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
		self.logging.log_terms_test = [
			dict(name='batch_t', fmt=':>5.3f', add_name='avg'),
			dict(name='cos', suffixes=[''], fmt=':>5.3f', add_name='avg'),
		]
