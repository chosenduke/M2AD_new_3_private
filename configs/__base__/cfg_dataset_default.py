from argparse import Namespace
from timm.data.constants import IMAGENET_DEFAULT_MEAN
from timm.data.constants import IMAGENET_DEFAULT_STD
import torchvision.transforms.functional as F

class cfg_dataset_default(Namespace):

	def __init__(self):
		Namespace.__init__(self)

		# self.size = 512
		self.size = 256
		self.setup = 'M2AD-Synergy' # 'M2AD-Invariant'

		###### Train DATA
		self.train_data = Namespace()
		self.train_data.anomaly_generator = Namespace()
		self.train_data.sampler = Namespace()
		self.train_data.sampler.name = 'naive'
		# self.train_data.sampler.name = 'balanced'
		self.train_data.sampler.kwargs = dict()
		self.train_data.loader_type = 'pil'
		self.train_data.loader_type_target = 'pil_L'
		self.train_data.type = 'M2AD'
		self.train_data.name = 'm2ad' # mvtec, visa
		self.train_data.mode = 'unsupervised'
		self.train_data.cls_names = ['Ring'] ###### Set to [] to utilize all classes
		self.train_data.anomaly_generator.name = 'white_noise'
		self.train_data.anomaly_generator.enable = False ### TODO
		self.train_data.anomaly_generator.kwargs = dict()

		###### Test DATA
		self.test_data = Namespace()
		self.test_data.anomaly_generator = Namespace()
		self.test_data.sampler = Namespace()
		self.test_data.sampler.name = 'naive'
		self.test_data.sampler.kwargs = dict()
		self.test_data.loader_type = 'pil'
		self.test_data.loader_type_target = 'pil_L'
		self.test_data.type = 'M2AD'
		self.test_data.name = 'm2ad' # mvtec, visa
		self.test_data.mode = 'unsupervised'
		self.test_data.cls_names = ['Ring'] ###### Set to [] to utilize all classes -- multi-class anomaly detection
		self.test_data.anomaly_generator.name = 'white_noise'
		self.test_data.anomaly_generator.enable = False ### TODO
		self.test_data.anomaly_generator.kwargs = dict()


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

