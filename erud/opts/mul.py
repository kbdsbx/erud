from erud.cg.payload import payload
from erud.opts._utils import partial_sum
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

class mul (payload) :
    # 缓存
    __x : any
    __y : any

    # 前向传播时，计算z = x * y，并缓存
    def fprop(self, x, y) -> any :
        self.__x = x
        self.__y = y

        return np.multiply(self.__x, self.__y)
    
    # 反向传播时，dz/dx = y; dz/dy = x
    def bprop(self, dz) -> list[any] :
        _x = self.__x
        _y = self.__y

        dx = np.multiply(dz, _y)
        # 偏导相加，使dx成为与x相同维度的导数张量
        dx = partial_sum(dx, _x)
        
        dy = np.multiply(dz, _x)
        dy = partial_sum(dy, _y)
            
        return [dx, dy]