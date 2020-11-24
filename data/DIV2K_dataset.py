import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import numpy as np
import random


class ImageTrainDataset(data.Dataset):
	def __init__(self, opt):
		super(ImageTrainDataset, self).__init__()
		self.cache_data = True
		self.name = opt['name'] 
		self.GT_root, self.LQ_root = './'+self.name, './'+self.name
		self.data_type = 'img'
		self.data_info = {'path_LQ':[],  'path_GT':[],	'folder':[],  'idx':[],  'border':[]}
		if self.data_type == 'lmdb':
			raise ValueError('No need to use LMDB during validation/test.')
		else:
			self.imgs_LQ, self.imgs_GT = {}, {}
			acc = 0
			if self.name.lower() in ['div2k_easy', 'div2k_train']:
				subfolders_LQ = util.glob_file_list(self.LQ_root)
				subfolders_GT = util.glob_file_list(self.GT_root)
				for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
				  
					subfolder_name = osp.basename(subfolder_GT)
					img_paths_LQ = util.glob_file_list(subfolder_LQ)
					img_paths_GT = util.glob_file_list(subfolder_GT)
					max_idx = len(img_paths_LQ)
					assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'
				  
					self.data_info['path_LQ'].extend(img_paths_LQ)
					self.data_info['path_GT'].extend(img_paths_GT)
					self.data_info['folder'].extend([subfolder_name] * max_idx)
					for i in range(0, max_idx):
						self.data_info['idx'].append('{}/{}'.format(i, i))

				   
					if self.cache_data:
						#self.imgs_LQ[subfolder_name] = util.read_img_seq_div(img_paths_LQ)
						self.imgs_GT[subfolder_name] = util.read_img_seq_div(img_paths_GT)

			else:
				raise ValueError('Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

	def __getitem__(self, index):
		path_LQ = self.data_info['path_LQ'][index]
		path_GT = self.data_info['path_GT'][index]
		folder = self.data_info['folder'][index]
		idx, max_idx = self.data_info['idx'][index].split('/')
		idx, max_idx = int(idx), int(max_idx)
		if self.cache_data:
			img_GT = []
			GT = self.imgs_GT[folder][idx]
			H, W = GT.shape[1:]
			# Random crop
			rnd_h = random.randint(0, H-192)
			rnd_w = random.randint(0, W-192)
			GT = GT[:, rnd_h:rnd_h+192, rnd_w:rnd_w+192]
			# Random flip
			hflip = random.random() < 0.5
			vflip = random.random() < 0.5
			GT = GT[:, ::-1, :] if hflip else GT
			GT = GT[:, :, ::-1] if vflip else GT
			img_GT = torch.from_numpy(np.ascontiguousarray(GT))
			imgs_LQ = img_GT
		return {'LQ':imgs_LQ,  'GT':img_GT, 
		 'folder':folder, 
		 'idx':self.data_info['idx'][index], 
		 'LQ_path':path_LQ, 
		 'GT_path':path_GT}

	def __len__(self):
		return len(self.data_info['idx'])


def crop(img_):
	h,w = img_.shape[1:]
	if h>800:
		s = int((h-800)/2)
		img_ = img_[:, s:s+800, :]
	if w>800:
		s = int((w-800)/2)
		img_ = img_[:, :, s:s+800]
	return img_


class ImageValDataset(data.Dataset):
	def __init__(self, opt):
		super(ImageValDataset, self).__init__()
		self.cache_data = True
		self.name = opt['name'] 
		self.GT_root, self.LQ_root = './'+self.name, './'+self.name
		self.data_type = 'img'
		self.data_info = {'path_LQ':[],  'path_GT':[],	'folder':[],  'idx':[],  'border':[]}
		if self.data_type == 'lmdb':
			raise ValueError('No need to use LMDB during validation/test.')
		else:
			self.imgs_LQ, self.imgs_GT = {}, {}
			acc = 0
			if self.name.lower() in ['div2k_val']:
				subfolders_LQ = util.glob_file_list(self.LQ_root)
				subfolders_GT = util.glob_file_list(self.GT_root)
				for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):
				  
					subfolder_name = osp.basename(subfolder_GT)
					img_paths_LQ = util.glob_file_list(subfolder_LQ)
					img_paths_GT = util.glob_file_list(subfolder_GT)
					max_idx = len(img_paths_LQ)
					assert max_idx == len(img_paths_GT), 'Different number of images in LQ and GT folders'
				  
					self.data_info['path_LQ'].extend(img_paths_LQ)
					self.data_info['path_GT'].extend(img_paths_GT)
					self.data_info['folder'].extend([subfolder_name] * max_idx)
					for i in range(0, max_idx):
						self.data_info['idx'].append('{}/{}'.format(i, i))
 
					if self.cache_data:
						self.imgs_GT[subfolder_name] = [crop(img_) for img_ in util.read_img_seq_div(img_paths_GT)]
						

			else:
				raise ValueError('Not support video test dataset. Support Vid4, REDS4 and Vimeo90k-Test.')

	def __getitem__(self, index):
		path_LQ = self.data_info['path_LQ'][index]
		path_GT = self.data_info['path_GT'][index]
		folder = self.data_info['folder'][index]
		idx, max_idx = self.data_info['idx'][index].split('/')
		idx, max_idx = int(idx), int(max_idx)
		if self.cache_data:
			GT = np.asarray(self.imgs_GT[folder][idx])
			img_GT = GT
			imgs_LQ = GT
		return {'LQ':imgs_LQ,  'GT':img_GT, 
		 'folder':folder, 
		 'idx':self.data_info['idx'][index], 
		 'LQ_path':path_LQ, 
		 'GT_path':path_GT}

	def __len__(self):
		return len(self.data_info['idx'])
