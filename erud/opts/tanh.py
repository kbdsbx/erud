from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

class tanh(payload) :
    # 缓存计算结果用于反向传播
    __a : any

    # a = (e^x - e^-x) / (e^x + e^-x)
    def fprop(self, x) -> any :
        _a = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

        self.__a = _a

        return _a
    
    # dz/dx = 1 - a^2
    def bprop(self, dz) -> list[any] :
        _a = self.__a
        
        dx = (1 - _a ** 2) * dz

        return [dx]