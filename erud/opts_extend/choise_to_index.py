from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
from erud.errors import *
import time

# 为rnn设计
# 从指定概率随机选择一个下标，并将其转换为one_hot向量
class choise_to_index (payload) :
	def fprop(self, y) -> any :
		np.random.seed(int(time.time()))


		size = np.size(y)
		p = y.reshape((size,))
		idx = np.random.choice([i for i in range(size)], p = p.ravel())

		z = np.zeros((size, 1))
		z[idx] = 1

		return z.reshape(y.shape)


	def bprop(self, dz) -> list[any] :
		raise UnsupportedError('Can not call function bprop from "accuracy".')