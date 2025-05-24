from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_common(Namespace):

	def __init__(self):
		Namespace.__init__(self)

		self.fvcore_is = True
		self.fvcore_b = 1
		self.fvcore_c = 3
		self.epoch_full = 300

		# ==> evaluator
		self.m2ad_synergy_metrics = [
			'AUROC_obj_sp', 'AUROC_view_sp', 'AUROC_view_px',
			'mF1_obj_sp', 'mF1_view_sp', 'mF1_view_px',
			'AP_obj_sp', 'AP_view_sp', 'AP_view_px',
			'AUPRO_view'
		]

		self.m2ad_invariant_metrics = [
			'AUROC_img_sp', 'AUROC_img_px', 'mF1_img_sp', 'mF1_img_px',
			'AP_img_sp', 'AP_img_px', 'AUPRO_img'
		]

		self.metrics = self.m2ad_synergy_metrics
		self.use_adeval = True
		self.evaluator = Namespace()
		self.evaluator.kwargs = dict(metrics=self.metrics, pooling_ks=None, max_step_aupro=100, use_adeval=self.use_adeval)

		self.vis = False
		self.vis_dir = None

		# ==> optim
		self.optim = Namespace()
		self.optim.lr = 0.005
		self.optim.kwargs = dict(name='adam', betas=(0.5, 0.999))

		# ==> trainer
		self.trainer = Namespace()
		self.trainer.name = 'ViTADTrainer'
		# self.trainer.checkpoint = 'runs'
		self.trainer.checkpoint = 'runs'

		self.trainer.logdir_sub = 'exp1'
		self.trainer.resume_dir = ''
		self.trainer.cuda_deterministic = False
		self.trainer.epoch_full = self.epoch_full
		self.trainer.scheduler_kwargs = dict(
			name='step', lr_noise=None, noise_pct=0.67, noise_std=1.0, noise_seed=42, lr_min=0.005 / 1e2,
			warmup_lr=0.005 / 1e3, warmup_iters=-1, cooldown_iters=0, warmup_epochs=0, cooldown_epochs=0,
			use_iters=True, patience_iters=0, patience_epochs=0, decay_iters=0, decay_epochs=int(self.epoch_full * 0.8),
			cycle_decay=0.1, decay_rate=0.1)
		# self.trainer.mixup_kwargs = dict(mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_minmax=None, prob=0.0, switch_prob=0.5,
		# 							mode='batch', correct_lam=True, label_smoothing=0.1)
		self.trainer.test_start_epoch = self.trainer.epoch_full
		self.trainer.test_per_epoch = 30
		self.trainer.find_unused_parameters = False
		self.trainer.sync_BN = 'apex'  # [none, native, apex, timm]
		self.trainer.dist_BN = ''  # [ , reduce, broadcast], valid when sync_BN is 'none'
		self.trainer.scaler = 'none'  # [none, native, apex]

		self.trainer.data = Namespace()
		self.trainer.data.batch_size = 16
		self.trainer.data.batch_size_per_gpu = None
		self.trainer.data.batch_size_test = None
		self.trainer.data.batch_size_per_gpu_test = 16
		self.trainer.data.num_workers_per_gpu = 8
		self.trainer.data.drop_last = False
		self.trainer.data.pin_memory = True
		self.trainer.data.persistent_workers = False



		# ==> loss
		self.loss = Namespace()
		self.loss.loss_terms = [
			dict(type='CosLoss', name='cos', avg=False, lam=1.0),
		]
		self.loss.clip_grad = 5.0
		self.loss.create_graph = False
		self.loss.retain_graph = False
		self.adv = False

		# ==> logging
		self.logging = Namespace()
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
		self.logging.train_reset_log_per = 50
		self.logging.train_log_per = 50
		self.logging.test_log_per = 50