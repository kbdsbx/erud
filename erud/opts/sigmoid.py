from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

class sigmoid(payload) :
    __a : any = None

    @property
    def data (self) :
        return self.__a

    # a = 1 / (1 + e^-x)
    def fprop(self, x) -> any :
        # 缓存中间变量用于反向传播
        self.__a = 1. / ( 1. + np.exp(-x))

        return self.__a
    
    # dz/dx = a(1 - a)
    def bprop(self, dz) -> list[any] :
        _a = self.__a

        return [_a * (1. - _a) * dz]