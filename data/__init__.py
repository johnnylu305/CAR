'''create dataset and dataloader'''
import logging
import torch
import torch.utils.data


def create_dataloader(dataset, dataset_opt, opt=None, sampler=None):
	phase = dataset_opt['phase']
	if phase == 'train':
		num_workers = 6*4 #dataset_opt['n_workers'] * len(opt['gpu_ids'])
		batch_size = 16*4 #dataset_opt['batch_size']
		shuffle = True
		return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
										   num_workers=num_workers, sampler=sampler, drop_last=True,
										   pin_memory=False)
	else:
		return torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1,
										   pin_memory=True)


def create_dataset(dataset_opt):
	# assign dataset
	# Vimeo90K: Vimeo90K train&val
	# video_test: Vid4 test
	mode = dataset_opt['mode']
	if mode == 'LQ':
		from data.LQ_dataset import LQDataset as D
	elif mode == 'LQGT':
		from data.LQGT_dataset import LQGTDataset as D
	elif mode == 'Vimeo90K':
		from data.Vimeo90K_dataset import Vimeo90KDataset as D
	elif mode == 'video_test':
		from data.video_test_dataset import VideoTestDataset as D
	elif mode in ['DIV2K_easy', 'DIV2K_train']:
		from data.DIV2K_dataset import ImageTrainDataset as D
	elif mode in ['DIV2K_val']:
		from data.DIV2K_dataset import ImageValDataset as D
	else:
		raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))
	dataset = D(dataset_opt)

	logger = logging.getLogger('base')
	logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
														   dataset_opt['name']))
	return dataset
