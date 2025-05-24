from util.compute_am import compute_discrepancy_map, maximum_as_anomaly_score
try:
	from apex import amp
	from apex.parallel import DistributedDataParallel as ApexDDP
	from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
	from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN

from ._base_trainer import BaseTrainer
from . import TRAINER
import torch
from torch.nn import functional as F


@TRAINER.register_module
class DinomalyTrainer(BaseTrainer):
	def __init__(self, cfg):
		super(DinomalyTrainer, self).__init__(cfg)


	def pre_train(self):
		# self.complexity_analysis((1,3,256,256))
		pass

	def pre_test(self):
		pass

	def set_input(self, inputs, **kwargs):
		self.imgs = inputs['img'].cuda()
		self.imgs_mask = inputs['img_mask'].cuda()

		self.bs = self.imgs.shape[0]

		### Note
		self.view = inputs['view']
		self.illumination = inputs['illumination']
		self.object_anomaly = inputs['object_anomaly']
		self.image_anomaly = inputs['image_anomaly']
		self.object_name = inputs['object_name']
		self.cls_name = inputs['cls_name']
		self.img_path = inputs['img_path']

	def forward(self, **kwargs):
		self.en, self.de = self.net(self.imgs)

	def compute_loss(self, **kwargs):
		dinomaly_loss = self.loss_terms['dinomaly_loss'](self.en, self.de)
		loss_log = {'dinomaly_loss': dinomaly_loss}
		return dinomaly_loss, loss_log

	def compute_anomaly_scores(self):
		anomaly_map, anomaly_map_list = compute_discrepancy_map(self.en, self.de,
														[self.imgs.shape[2], self.imgs.shape[3]], use_cos=True,
														uni_am=False, amap_mode='add', gaussian_sigma=4)
		anomaly_score = maximum_as_anomaly_score(anomaly_map, 0)
		return anomaly_map, anomaly_score

