import os
import numpy as np
import h5py
import torch
from glob import glob
from natsort import natsorted
from torch.utils.data import Dataset
import cv2
from torch.utils.data import DataLoader
from flow_utils import plot
from torchvision.models.optical_flow import raft_large
# from torchvision.models.optical_flow.raft.RAFT Raft_Large_Weights
# import torchvision.transforms.functional as F
from torchvision.utils import flow_to_image 
import itertools
import collections

cudaid = "cuda:"+str(1)
device = torch.device(cudaid)

torch.manual_seed(45)

def accuracy(output, labels):
    preds  = output.max(1)[1].type_as(labels)
    labels = labels.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


class CustomData(Dataset):
	def __init__(self, path_img, path_flow, rows=200, cols=200):
		self.rows = rows
		self.cols = cols
		path_img  = natsorted(glob(path_img))
		# self.path_flow = natsorted(glob(path_flow))

		self.pair_img_path = []
		self.pair_img_targ = []
		for path in path_img:
			subpath  = natsorted(glob(path+'/*'))
			# self.pair_img_path.extend([(subpath[idx], subpath[idx+1], subpath[idx+2], subpath[idx+3]) for idx in range(0, len(subpath)-3, 1)])
			tmp_lst = [(subpath[idx], subpath[idx+1], subpath[idx+2], subpath[idx+3]) for idx in range(0, len(subpath)-3, 1)]

			for item in tmp_lst:
				self.pair_img_path.extend(list(itertools.permutations(item))[:12])
				self.pair_img_targ.extend(list(range(12)))

		self.pos_feat = []
		for x in range(rows):
			for y in range(cols):
				self.pos_feat.append([x,y])

		self.pos_feat = np.array(self.pos_feat)
		self.pos_feat = self.pos_feat / np.max(self.pos_feat)
		# print(len(self.pair_img_path), len(self.path_flow))

	def __len__(self):
		return len(self.pair_img_path)

	def compute_feat(self, im):

		im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

		dx = cv2.Sobel(im_gray, 0, dx=1,dy=0)/255.0

		dy = cv2.Sobel(im_gray, 0, dx=0,dy=1)/255.0

		rgb_feat = im.reshape(-1,3)

		dx_feat = dx.reshape(-1,1)
		dy_feat = dy.reshape(-1,1)

		feat = np.concatenate((self.pos_feat, rgb_feat, dx_feat, dy_feat),axis=1)
		return feat

	def __getitem__(self, idx):
		# print(self.pair_img_path[idx])
		ima_path = self.pair_img_path[idx][0]
		imb_path = self.pair_img_path[idx][1]
		imc_path = self.pair_img_path[idx][2]
		imd_path = self.pair_img_path[idx][3]
		image_a = cv2.resize(cv2.imread(ima_path, 1), (self.rows, self.cols))
		image_b = cv2.resize(cv2.imread(imb_path, 1), (self.rows, self.cols))
		image_c = cv2.resize(cv2.imread(imc_path, 1), (self.rows, self.cols))
		image_d = cv2.resize(cv2.imread(imd_path, 1), (self.rows, self.cols))
		image_a = cv2.cvtColor(image_a, cv2.COLOR_BGR2RGB)
		image_b = cv2.cvtColor(image_b, cv2.COLOR_BGR2RGB)
		image_c = cv2.cvtColor(image_c, cv2.COLOR_BGR2RGB)
		image_d = cv2.cvtColor(image_d, cv2.COLOR_BGR2RGB)

		# flow = cv2.calcOpticalFlowFarneback(cv2.cvtColor(image_a, cv2.COLOR_RGB2GRAY),\
		# 									cv2.cvtColor(image_b, cv2.COLOR_RGB2GRAY), None, 0.5, 3, 15, 3, 5, 1.2, 0)

		image_a = image_a.astype(np.float32)/255.0
		image_b = image_b.astype(np.float32)/255.0
		image_c = image_c.astype(np.float32)/255.0
		image_d = image_d.astype(np.float32)/255.0

		imfeat_a = torch.FloatTensor(self.compute_feat(image_a)).unsqueeze(0)
		imfeat_b = torch.FloatTensor(self.compute_feat(image_b)).unsqueeze(0)
		imfeat_c = torch.FloatTensor(self.compute_feat(image_c)).unsqueeze(0)
		imfeat_d = torch.FloatTensor(self.compute_feat(image_d)).unsqueeze(0)

		feat     = torch.cat((imfeat_a, imfeat_b, imfeat_c, imfeat_d),0)

		targ     = np.zeros(shape=(12,))
		targ[self.pair_img_targ[idx]] = 1.0
		targ     = torch.FloatTensor(targ)

		# # Thanks to: https://stackoverflow.com/a/28016469
		# # with open(self.path_flow[idx], 'rb') as f:
		# # 	magic, = np.fromfile(f, np.float32, count=1)
		# # 	if 202021.25 != magic:
		# # 		print('Magic number incorrect. Invalid .flo file')
		# # 	else:
		# # 		w, h = np.fromfile(f, np.int32, count=2)
		# # 		# print(f'Reading {w} x {h} flo file')
		# # 		data = np.fromfile(f, np.float32, count=2*w*h)
		# # 		# Reshape data into 3D array (columns, rows, bands)
		# # 		flow_data = np.resize(data, (w, h, 2))
		# # 		# flow_data = torch.FloatTensor(flow_data).resize_(self.rows, self.cols, 2)
		# # 		flow_data = torch.FloatTensor(flow_data)
		# # magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
		# flow_data = torch.FloatTensor(flow).reshape(-1, 2)
		# # print(flow_data)
		
		# return image_a, image_b, feat, flow_data 
		return image_a, image_b, image_c, image_d, targ, feat


# dataloader = CustomData(path_img='/home/cvig/Documents/Clip_Ordering_2/data/MPI-Sintel-complete/training/clean/*',\
# 						path_flow='/home/cvig/Documents/Clip_Ordering_2/data/MPI-Sintel-complete/training/flow/*/*',
# 						)

# image_a, image_b, image_c, image_d, targ, feat = next(iter(dataloader))

# print(image_a.shape, image_b.shape, image_c.shape, image_d.shape, targ, feat.shape)

# train_dataloader = DataLoader(dataloader, batch_size=1, shuffle=True, num_workers=16, prefetch_factor=4)

# for data in train_dataloader:
# 	ima, imb, feat, flowdata = data
# 	# print(ima.shape, imb.shape, feat.shape, flowdata.shape)
# 	ima  = ima.squeeze(0)
# 	imb  = imb.squeeze(0)
# 	feat = feat.squeeze(0)
# 	flowdata  = flowdata.squeeze(0)
# 	ima = torch.permute(ima, [2, 0, 1])
# 	imb = torch.permute(imb, [2, 0, 1])
# 	flowdata = torch.permute(flowdata, [2, 1, 0])
# 	flow_imgs = flow_to_image(flowdata)
# 	# flow_imgs = cv2.cvtColor(flow_imgs, cv2.COLOR_BGR2RGB)
# 	# flow_imgs = torch.FloatTensor(flow_imgs)
# 	# flow_imgs = torch.permute(flow_imgs, [2, 1, 0])
# 	# print(ima.shape, imb.shape, feat.shape, flowdata.shape, flow_imgs.shape)
# 	plot([imb, flow_imgs])
# 	# break
	

# 	# If you can, run this example on a GPU, it will be a lot faster.
# 	# device = "cuda" if torch.cuda.is_available() else "cpu"

# 	# model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
# 	# model = model.eval()

# 	# list_of_flows = model(imga.to(device), imgb.to(device))
# 	# print(f"type = {type(list_of_flows)}")
# 	# print(f"length = {len(list_of_flows)} = number of iterations of the model")
