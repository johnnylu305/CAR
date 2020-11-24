import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function, gradcheck

from .adaptive_gridsampler_cuda import forward, backward

class GridSamplerFunction(Function):
	@staticmethod
	def forward(ctx, img, kernels, offsets_h, offsets_v, offset_unit, padding, downscale_factor):
		assert isinstance(downscale_factor, int)
		assert isinstance(padding, int)

		ctx.padding = padding
		ctx.offset_unit = offset_unit

		b, c, h, w = img.size()
		assert h // downscale_factor == kernels.size(2)
		assert w // downscale_factor == kernels.size(3)

		img = nn.ReflectionPad2d(padding)(img)
		ctx.save_for_backward(img, kernels, offsets_h, offsets_v)
		output = img.new(b, c, h // downscale_factor, w // downscale_factor).zero_()
		forward(img, kernels, offsets_h, offsets_v, offset_unit, padding, output)
		return output
	
	
	@staticmethod
	def backward(ctx, grad_output):
		img, kernels, offsets_h, offsets_v, = ctx.saved_tensors
		K = torch.zeros(kernels.shape).cuda()
		H = torch.zeros(offsets_h.shape).cuda()
		V = torch.zeros(offsets_v.shape).cuda()
		#print(offsets_h.shape, offsets_v.shape, kernels.shape)
		backward(img, kernels, offsets_h, offsets_v, ctx.offset_unit, ctx.padding, grad_output, K, H, V)
		#raise NotImplementedError
		#return None, None, None, None, None, None, None
		# user
		return None, K, None, None, None, None, None
	

class Downsampler(nn.Module):
	def __init__(self, ds, k_size):
		super(Downsampler, self).__init__()
		self.ds = ds
		self.k_size = k_size

	def forward(self, img, kernels, offsets_h, offsets_v, offset_unit):
		assert self.k_size ** 2 == kernels.size(1)
		return GridSamplerFunction.apply(img, kernels, offsets_h, offsets_v, offset_unit, self.k_size // 2, self.ds)
