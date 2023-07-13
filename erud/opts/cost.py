from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

# 总代价函数
class cost (payload) :
    __x = any

    def fprop(self, x) -> float :
        r = x
        if isinstance(x, np.ndarray) :
            r = 1. / np.size(x) * np.sum(x)
        
        self.__x = x

        return r
    
    def bprop(self, dz) -> list[any] :
        _x = self.__x

        dx = 1
        if isinstance(_x, np.ndarray) :
            dx = np.ones_like(_x) / np.size(_x) * dz
            dx.reshape(_x.shape)

        return [dx]

