from erud.cg.payload import payload
import numpy as np

# 总代价函数
class cost (payload) :
    __x = any

    def fprop(self, x) -> float :
        r = x
        if isinstance(x, np.ndarray) :
            r = 1. / np.size(x) * np.nansum(x)
        
        self.__x = x

        return r
    
    def bprop(self, dz) -> list[any] :
        _x = self.__x

        dx = 1
        if isinstance(_x, np.ndarray) :
            dx = np.ones_like(_x) / np.size(_x) * dz

        return [dx]

