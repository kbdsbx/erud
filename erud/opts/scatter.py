from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
	import cupy as cp
import numpy as np

class scatter(payload) :
	__axises : any

	__prop_type : int = 1

	# 重载
	@property
	def prop_type (self) -> any:
		return self.__prop_type


	def __init__(self, axises : int) -> None:
		self.__axises = axises
	
	def fprop(self, x) -> any :
		_axises = self.__axises

		# 计算分块后的矩阵维度
		nshape = list(x.shape)
		nshape.pop(_axises)
		nshape = tuple(nshape)

		ls = [l.reshape(nshape) for l in np.split(x, x.shape[_axises], _axises)]

		return ls
	
	def bprop(self, dz) -> list[any] :
		_axises = self.__axises

		return [np.stack(dz, _axises)]

	

