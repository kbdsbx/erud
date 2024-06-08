from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
	import cupy as cp
import numpy as np

class gather(payload) :
	__axises : any

	def __init__(self, axises : int) -> None:
		self.__axises = axises
	
	def fprop(self, *x) -> any :
		_axises = self.__axises
		r = np.stack(x, _axises)
		return r
		
	def bprop(self, dz) -> list[any] :
		_axises = self.__axises

		# 计算分块后的矩阵维度
		nshape = list(dz.shape)
		nshape.pop(_axises)
		nshape = tuple(nshape)

		lt = [l.reshape(nshape) for l in np.split(dz, dz.shape[_axises], _axises)]
		return lt
	
	

