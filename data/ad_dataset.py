import warnings
import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
from data.dataset_info import *
from util.data import get_img_loader
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)
from . import DATA

@DATA.register_module
class M2AD(data.Dataset):
	def __init__(self, data_cfg, train=True, transform=None, target_transform=None, setup='M2AD-Synergy'):

		assert setup in ['M2AD-Synergy', 'M2AD-Invariant'], f"Only support mode in ['M2AD-Synergy', 'M2AD-Invariant']"
		self.name = data_cfg.name
		assert self.name in DATA_SUBDIR.keys(), f"Only Support {DATA_SUBDIR.keys()}"

		self.root = os.path.join(DATA_ROOT, DATA_SUBDIR[self.name])
		self.setup = setup
		self.train = train
		self.transform = transform
		self.target_transform = target_transform

		self.loader = get_img_loader(data_cfg.loader_type)
		self.loader_target = get_img_loader(data_cfg.loader_type_target)

		self.all_cls_names = CLASS_NAMES[self.name]
		self.data_all = []

		meta_path = EXPERIMENTAL_SETUP[data_cfg.mode]
		meta_info = json.load(open(f'{self.root}/jsons/{meta_path}', 'r'))
		meta_info = meta_info['train' if self.train else 'test']
		self.cls_names = data_cfg.cls_names
		if not isinstance(self.cls_names, list):
			self.cls_names = [self.cls_names]
		self.cls_names = list(meta_info.keys()) if len(self.cls_names) == 0 else self.cls_names

		for cls_name in self.cls_names:
			self.data_all.extend(meta_info[cls_name])

		# adjust the testing images according to the utilized mode
		if not self.train:
			if self.setup == 'M2AD-Synergy':
				pass # load all images, need to do nothing
			elif self.setup == 'M2AD-Invariant':
				# load only detectable abnormal images and normal images
				# i.e, detectable == '' or True -> != False
				self.data_all = [f for f in self.data_all if f['detectable'] != False]

		random.shuffle(self.data_all) if self.train else None
		self.length = len(self.data_all)

		self.enable_anomaly_generation = data_cfg.anomaly_generator.enable
		if hasattr(data_cfg, 'anomaly_generator'):
			if data_cfg.anomaly_generator.name:
				self.anomaly_generator = ANOMALY_GENERATOR.get(data_cfg.anomaly_generator.name, None)
				self.anomaly_generator = self.anomaly_generator(**data_cfg.anomaly_generator.kwargs)

		self.normal_idx = []
		self.outlier_idx = []
		for indx, item in enumerate(self.data_all):
			if item['image_anomaly'] == 0:
				self.normal_idx.append(indx)
			else:
				self.outlier_idx.append(indx)

		#### verbose
		self.verbose_info = []
		cls_stats = {cls: {'normal': 0, 'anomalous': 0} for cls in self.cls_names}

		for v in self.data_all:
			cls = v['cls_name']
			if cls in cls_stats:
				if v['image_anomaly'] == 1:
					cls_stats[cls]['anomalous'] += 1
				elif v['image_anomaly'] == 0:
					cls_stats[cls]['normal'] += 1

		for cls in self.cls_names:
			normal_samples = cls_stats[cls]['normal']
			anomalous_samples = cls_stats[cls]['anomalous']
			self.verbose_info.append(
				f'Training: {train}, Class: {cls}, #Normal: {normal_samples}, #Abnormal: {anomalous_samples}, '
				f'#Total: {normal_samples + anomalous_samples}')

	def __len__(self):
		return self.length

	def __getitem__(self, index):
		data = self.data_all[index]

		img_path, mask_path  = data['img_path'], data['mask_path']
		view, illumination = data['view'], data['illumination']
		object_anomaly, image_anomaly = data['object_anomaly'], data['image_anomaly']
		object_name, cls_name = data['object_name'], data['cls_name']

		img_path = f'{self.root}/{img_path}'
		img = self.loader(img_path)
		if image_anomaly == 0:
			img_mask = Image.fromarray(np.zeros((img.size[0], img.size[1])), mode='L')
		else:
			img_mask = np.array(self.loader_target(f'{self.root}/{mask_path}')) > 0
			img_mask = Image.fromarray(img_mask.astype(np.uint8) * 255, mode='L')

		return_dict = {'view': view, 'illumination': illumination,
					   'object_anomaly': object_anomaly, 'image_anomaly': image_anomaly,
					   'object_name': object_name, 'cls_name': cls_name, 'img_path': img_path,
					   'mask_path': mask_path}

		if hasattr(self, 'anomaly_generator'):
			if self.train: # only need to generate anomalies during the training stage
				if self.enable_anomaly_generation and image_anomaly != 1:	# we don't generate anomalies for real anomalies...
					augmented_image, augmented_mask, augmented_anomaly = self.anomaly_generator(img)
				else: # just use the original data
					augmented_image = img
					augmented_mask = img_mask
					augmented_anomaly = image_anomaly

				augmented_image = self.transform(augmented_image) if self.transform is not None else augmented_image
				augmented_mask = self.target_transform(
					augmented_mask) if self.target_transform is not None and augmented_mask is not None else augmented_mask
				augmented_mask = [] if augmented_mask is None else augmented_mask

				return_dict.update({'augmented_image': augmented_image,
									'augmented_mask': augmented_mask,
									'augmented_anomaly': augmented_anomaly,
									})

		img = self.transform(img) if self.transform is not None else img
		img_mask = self.target_transform(img_mask) if self.target_transform is not None and img_mask is not None else img_mask
		img_mask = [] if img_mask is None else img_mask

		return_dict.update({'img': img, 'img_mask': img_mask})

		return return_dict


class ToTensor(object):
	def __call__(self, image):
		try:
			image = torch.from_numpy(image.transpose(2, 0, 1))
		except:
			print('Invalid_transpose, please make sure images have shape (H, W, C) before transposing')
		if not isinstance(image, torch.FloatTensor):
			image = image.float()
		return image


class Normalize(object):
	def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
		self.mean = np.array(mean)
		self.std = np.array(std)

	def __call__(self, image):
		image = (image - self.mean) / self.std
		return image

def get_data_transforms(size, isize):
	data_transforms = transforms.Compose([Normalize(),ToTensor()])
	gt_transforms = transforms.Compose([
		transforms.Resize((size, size)),
		transforms.ToTensor()])
	return data_transforms, gt_transforms


