from erud.cg.payload import payload
from erud._utils import useGPU

if useGPU :
	import cupy as cp
import numpy as np

# lambda/(m * 2) * (sum(W1^2) + sum(W2^2) + ... + sum(Wn^2))
class L2_regularization(payload) :

	__lamb : float = 0.
	__m : int = 0
	__args : list[any] = None

	def __init__(self, lamb) :
		self.__lamb = lamb

	def fprop(self, *args) :
		self.__args = args
		_lamb = self.__lamb

		_m = 0
		_a = 0.
		for w in args :
			_m = _m + np.size(w)
			_a = _a + np.sum(w * w)
		
		self.__m = _m

		return _lamb / 2. / _m * _a
	
	def bprop(self, dz) -> list[any]:
		ret = []
		_args = self.__args
		_lamb = self.__lamb
		_m = self.__m

		for w in _args :
			ret.append(dz * _lamb / _m * w)
		
		return ret
