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
class MSFlowTrainer(BaseTrainer):
	def __init__(self, cfg):
		super(MSFlowTrainer, self).__init__(cfg)


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
		self.z_list, self.jac = self.net(self.imgs)

	def compute_loss(self, **kwargs):
		loss = 0.
		for z in self.z_list:
			loss += 0.5 * torch.sum(z ** 2, (1, 2, 3))
		loss = loss - self.jac
		loss = loss.mean()

		loss_log = {'msflowloss': loss}

		return loss, loss_log

	def compute_anomaly_scores(self):
		anomaly_score, anomaly_map_list = self.net.postprocess(self.z_list, self.jac)
		anomaly_map_list = [a for a in anomaly_map_list]
		return anomaly_map_list, anomaly_score

