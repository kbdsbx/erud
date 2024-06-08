from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

class reshape(payload) :
    __old_shape : any
    __shape : any

    def __init__ (self, shape) :
        self.__shape = shape


    def fprop(self, x) -> any :
        self.__old_shape = x.shape
        _shape = self.__shape

        # 第一个维度为样本数量不变，只reshape后续其他维度
        _shape = (x.shape[0], *_shape)

        return np.reshape(x, _shape)
    
    def bprop(self, dz) -> list[any] :
        _old_shape = self.__old_shape

        dx = np.reshape(dz, _old_shape)

        return [dx]