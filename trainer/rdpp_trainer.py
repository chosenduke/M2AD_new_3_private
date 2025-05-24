from util.compute_am import compute_discrepancy_map, maximum_as_anomaly_score
from ._base_trainer import BaseTrainer
from . import TRAINER
from optim import get_optim

@TRAINER.register_module
class RDPPTrainer(BaseTrainer):
	def __init__(self, cfg):
		super(RDPPTrainer, self).__init__(cfg)
		self.optim.proj_opt = get_optim(cfg.optim.proj_opt.kwargs, self.net.proj_layer,	lr=cfg.optim.lr)
		proj_layer = self.net.proj_layer
		self.net.proj_layer = None
		self.optim.distill_opt = get_optim(cfg.optim.distill_opt.kwargs, self.net, lr=cfg.optim.lr * 5)
		self.net.proj_layer = proj_layer

	def pre_train(self):
		pass

	def pre_test(self):
		pass

	def set_input(self, inputs, train=True):
		self.imgs = inputs['img'].cuda()
		self.imgs_mask = inputs['img_mask'].cuda()

		if train:
			self.augmented_image = inputs['augmented_image'].cuda()
			self.augmented_mask = inputs['augmented_mask'].cuda()

		self.bs = self.imgs.shape[0]

		### Note: necessray for evaluations and visualizations
		self.view = inputs['view']
		self.illumination = inputs['illumination']
		self.object_anomaly = inputs['object_anomaly']
		self.image_anomaly = inputs['image_anomaly']
		self.object_name = inputs['object_name']
		self.cls_name = inputs['cls_name']
		self.img_path = inputs['img_path']

	def forward(self, train=True):
		if train:
			self.feats_t, self.feats_s, self.L_proj = self.net(self.imgs, self.augmented_image)
		else:
			self.feats_t, self.feats_s, self.L_proj = self.net(self.imgs)

	def compute_loss(self, train=True):
		loss_cos = self.loss_terms['cos'](self.feats_t, self.feats_s) + 0.2 * self.L_proj
		loss_log = {'cos': loss_cos}
		return loss_cos, loss_log

	def compute_anomaly_scores(self):
		anomaly_map, anomaly_map_list = compute_discrepancy_map(self.feats_t, self.feats_s,
														[self.imgs.shape[2], self.imgs.shape[3]],
														uni_am=False, amap_mode='add', gaussian_sigma=4)
		anomaly_score = maximum_as_anomaly_score(anomaly_map, 0.01)
		return anomaly_map, anomaly_score

