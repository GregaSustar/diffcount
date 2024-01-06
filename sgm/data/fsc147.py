import os
import cv2
import json
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Literal
import pytorch_lightning as pl
import warnings

class FSC147DataDictWrapper(Dataset):
	def __init__(self, dset):
		super().__init__()
		self.dset = dset

	def __getitem__(self, i):
		img, bbs, dm = self.dset[i]
		return {"jpg": img, "bboxes": bbs, "dm": dm}

	def __len__(self):
		return len(self.dset)
	

# TODO change this from hardcoded variables to config file (n_exemplars, resize_shape, root, dm_dirname, [maybe also transforms????])
class FSC147Loader(pl.LightningDataModule):
	def __init__(
			self,
			batch_size, 
			num_workers=0, 
			shuffle=True,):
		super().__init__()
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.shuffle = shuffle
		n_exemplars = 3
		root = os.environ.get("DATA_ROOT", None)
		if root is None:
			root = './'
			warnings.warn("'DATA_ROOT' environment variable not set, using current directory as data root")
		dm_dirname = 'gt_density_maps_ksize=3x3_sig=0.25'
		
		self.train_dataset = FSC147DataDictWrapper(
			FSC147Dataset(
				root=root,
				dm_dirname=dm_dirname,
				split='train',
				n_examplars=n_exemplars
			)
		)

		self.val_dataset = FSC147DataDictWrapper(
			FSC147Dataset(
				root=root,
				dm_dirname=dm_dirname,
				split='val',
				n_examplars=n_exemplars
			)
		)

		self.test_dataset = FSC147DataDictWrapper(
			FSC147Dataset(
				root=root,
				dm_dirname=dm_dirname,
				split='test',
				n_examplars=n_exemplars
			)
		)

	def prepare_data(self):
		pass

	def train_dataloader(self):
		return DataLoader(
			self.train_dataset, 
			batch_size=self.batch_size, 
			num_workers=self.num_workers, 
			shuffle=self.shuffle
		)
	
	def val_dataloader(self):
		return DataLoader(
			self.val_dataset, 
			batch_size=self.batch_size, 
			num_workers=self.num_workers, 
			shuffle=False
		)
	
	def test_dataloader(self):
		return DataLoader(
			self.test_dataset, 
			batch_size=self.batch_size, 
			num_workers=self.num_workers, 
			shuffle=False
		)


class FSC147Dataset(Dataset):
	"""
	:param root: Root directory of dataset.
	:param dm_dirname: Name of the directory containing the GT density maps.
	:param split: 'train', 'val' or 'test'.
	:param transform: Optional transform to be applied to the image.
	:param target_transform: Optional transform to be applied to the density map.
	:param n_examplars: Number of examplars
	:param return_patches: If True, returns image patches of the examplars, otherwise returns only bounding box coordinates.

	Make sure the root directory has the following structure:

   root
	├── images_384_VarV2
	│       ├─ 2.jpg
	│       ├─ 3.jpg
	│       ├─ ...
	│       └─ 7714.jpg
	├── annotation_FSC147_384.json
	├── Train_Test_Val_FSC_147.json                         
	├── dm_dirname
	│       ├─ 2.npy
	│       ├─ 3.npy
	│       ├─ ...
	│       └─ 7714.npy
	└── ImageClasses_FSC147.json	(optional)

	"""

	def __init__(
			self,
			root: str,
			dm_dirname: str,
			split: Literal['train', 'val', 'test'] = 'train',
			n_examplars: int = 3,
	):
		self.root = root
		self.dm_dirname = dm_dirname
		self.split = split
		self.n_examplars = n_examplars
		self.re_size = (256, 256)
		self.hflip_p = 0.5
		self.cj_p = 0.8

		self.img_names = None
		with open(os.path.join(self.root, 'Train_Test_Val_FSC_147.json'), 'rb') as f:
			self.img_names = json.load(f)[self.split]

		self.annotations = None
		with open(os.path.join(self.root, 'annotation_FSC147_384.json'), 'rb') as f:
			self.annotations = {k: v for k, v in json.load(f).items() if k in self.img_names}


	def transform(self, img, bboxes, dm, split='train'):
		# ToTensor
		img = F.to_tensor(img) # (C, H, W)
		dm = F.to_tensor(dm)

		# Resize
		old_h, old_w = img.shape[-2:]
		img = F.resize(img, self.re_size, antialias=True)
		dm = F.resize(dm, self.re_size, antialias=True)
		new_h, new_w = img.shape[-2:]
		rw = new_w / old_w
		rh = new_h / old_h
		bboxes = bboxes * torch.tensor([rw, rh, rw, rh])

		if split == 'train':
			# RandomHorizontalFlip
			if torch.rand(1) < self.hflip_p:
				img = F.hflip(img)
				dm = F.hflip(dm)
				bboxes[:, [0, 2]] = self.re_size[0] - bboxes[:, [2, 0]]

			# RandomColorJitter
			if torch.rand(1) < self.cj_p:
				img = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)(img)

		# Normalize to [-1, 1]
		img = img * 2.0 - 1.0
		return img, bboxes, dm


	def __len__(self):
		return len(self.img_names)


	def __getitem__(self, index):
		if torch.is_tensor(index):
			index = index.tolist()

		img = Image.open(
			os.path.join(
				self.root,
				'images_384_VarV2',
				self.img_names[index]
			)
		).convert('RGB')

		density_map = np.load(
			os.path.join(
				self.root,
				self.dm_dirname,
				os.path.splitext(self.img_names[index])[0] + '.npy'
			)
		)

		bboxes = torch.as_tensor(self.annotations[self.img_names[index]]['box_examples_coordinates'])
		assert len(bboxes) >= self.n_examplars, f'Not enough examplars for image {self.img_names[index]}'
		bboxes = bboxes[:, [0, 2], :].reshape(-1, 4)[:self.n_examplars, ...]											# (x_min, y_min, x_max, y_max)

		img, bboxes, density_map = self.transform(img, bboxes, density_map, split=self.split)
		return img, bboxes, density_map



def generate_density_maps(rootdir, ksize, sigma, dtype=np.float32):
	"""
	Generates GT density maps from dot annotations and saves them to rootdir.
	"""
	savedir = os.path.join(rootdir, f'gt_density_maps_ksize={ksize}x{ksize}_sig={sigma}')
	if not os.path.isdir(savedir):
		os.makedirs(savedir)
	with open(os.path.join(rootdir, 'annotation_FSC147_384.json'), 'rb') as f:
		annotations = json.load(f)
		for img_name, ann in annotations.items():
			w, h = int(ann['W'] * ann['ratio_w']), int(ann['H'] * ann['ratio_h'])
			bitmap = np.zeros((h, w), dtype=dtype)
			for point in ann['points']:
				x, y = int(point[0])-1, int(point[1])-1
				bitmap[y, x] = 1.0
			density_map = cv2.GaussianBlur(bitmap, (ksize, ksize), sigma)
			np.save(
				os.path.join(savedir, os.path.splitext(img_name)[0] + '.npy'), 
				density_map
			)
			print(f'{img_name}.npy saved')