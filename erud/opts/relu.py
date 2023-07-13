from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

class relu(payload) :
    __x : any

    def fprop(self, x) -> any :
        self.__x = x

        return np.maximum(self.__x, 0)
    
    def bprop(self, dz) -> list[any]:
        _x = self.__x

        # 使用右导数，当xi为0时，dz/dxi = 1
        dx = dz * (_x >= 0)
        return [dx]