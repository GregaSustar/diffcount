from sgm.data.fsc147 import FSC147Dataset, generate_density_maps, FSC147Loader
from sgm.data.mnist import MNISTLoader
from sgm.data.cifar10 import CIFAR10Loader
# from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# import torch
from sgm.modules.diffusionmodules.openaimodel import UNetModel
# from safetensors import safe_open


if __name__ == "__main__":
	pass
	# data = MNISTLoader(batch_size=64)
	# data = CIFAR10Loader(batch_size=1)
	# data = FSC147Loader(batch_size=1)
	# for samp in data.train_dataloader():
	# 	print(samp['bboxes'])
	# 	plt.imshow(samp['jpg'][0].permute(1, 2, 0))
	# 	plt.plot(samp['bboxes'][0][:, 0], samp['bboxes'][0][:, 1], 'ro')
	# 	plt.plot(samp['bboxes'][0][:, 2], samp['bboxes'][0][:, 3], 'go')
	# 	plt.show()


	# import json
	# import os
	# import torch
	# root = "/mnt/c/users/grega/faks/mag/FSC147_384_V2/"
	# key = '7232.jpg'
	# with open(os.path.join(root, 'annotation_FSC147_384.json'), 'rb') as f:
	# 		annotations = {k: v for k, v in json.load(f).items()}
	# 		img = Image.open(
	# 			os.path.join(
	# 				root,
	# 				'images_384_VarV2',
	# 				key
	# 			)
	# 		).convert('RGB')
	# 		bboxes = torch.as_tensor(annotations[key]['box_examples_coordinates'])
	# 		bboxes = bboxes[:, [0, 2], :].reshape(-1, 4)[:3, ...]
	# 		print(bboxes)
	# 		plt.imshow(img)
	# 		plt.plot(bboxes[:, 0], bboxes[:, 1], 'ro')
	# 		plt.plot(bboxes[:, 2], bboxes[:, 3], 'go')
	# 		plt.show()


	# generate_density_maps('/mnt/c/Users/grega/faks/mag/FSC147_384_V2', ksize=3, sigma=0.25)

	# x = np.load("/d/hpc/projects/FRI/DL/gs1121/FSC147_384_V2/gt_density_maps_ksize=3x3_sig=0.25/2.npy")
	# x = np.load("../FSC147_384_V2/gt_density_maps_ksize=3x3_sig=0.25/2.npy")
	# print(x.dtype)
	# plt.imshow(x)
	# plt.show()
	# plt.savefig("dmtest.png")

	# plt.imshow(Image.open('C:/Users/grega/faks/mag/FSC147_384_V2/images_384_VarV2/1123.jpg'))
	# plt.imshow(np.load('C:/Users/grega/faks/mag/FSC147_384_V2/gt_density_maps_ksize=3x3_sig=0.25/1123.npy'), alpha=0.5)
	# plt.show()


	"""
	use_checkpoint: True
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [1, 2, 4]
        num_res_blocks: 2
        channel_mult: [1, 2, 4, 4]
        num_head_channels: 64
        num_classes: sequential
        adm_in_channels: 1792
        num_heads: 1
        transformer_depth: 1
        context_dim: 768
        spatial_transformer_attn_type: softmax-xformers



		adm_in_channels: 2816
        num_classes: sequential
        use_checkpoint: True
        in_channels: 4
        out_channels: 4
        model_channels: 320
        attention_resolutions: [4, 2]
        num_res_blocks: 2
        channel_mult: [1, 2, 4]
        num_head_channels: 64
        use_linear_in_transformer: True
        transformer_depth: [1, 2, 10]
        context_dim: 2048
        spatial_transformer_attn_type: softmax-xformers
	"""


	# model = UNetModel(
	# 	adm_in_channels= 2816,
    #     num_classes= 'sequential',
    #     use_checkpoint= True,
    #     in_channels= 4,
    #     out_channels= 4,
    #     model_channels= 320,
    #     attention_resolutions= [4, 2],
    #     num_res_blocks= 2,
    #     channel_mult= [1, 2, 4],
    #     num_head_channels= 64,
    #     use_linear_in_transformer= True,
    #     transformer_depth= [1, 2, 10],
    #     context_dim= 2048,
    #     spatial_transformer_attn_type= 'softmax-xformers'
	# )

	# path = 'C:/Users/grega/faks/mag/alpha/models/sd_xl_base_1.0.safetensors'
	# path = 'C:/Users/grega/faks/mag/alpha/models/sd-v2-1-base_unet.safetensors'

	# tensors = {}
	# with safe_open(path, framework="pt", device="cpu") as f:
	# 	for key in f.keys():
	# 		print(key)
	# 		tensors[key] = f.get_tensor(key)


	# for k in model.state_dict().keys():
	# 	print(k)

	# m, u = model.load_state_dict(tensors, strict=False)	
	# print('Missing keys:\n')
	# for k in m:
	# 	print(k)

	# print('\nUnexpected keys:\n')
	# for k in u:
	# 	print(k)
	# pass

