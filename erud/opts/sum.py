from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

class sum (payload) :
    __x : any

    def fprop(self, x) -> any :
        self.__x = x

        return np.sum(self.__x)
    
    def bprop(self, dz) -> list[any] :
        _x = self.__x

        dx = np.ones_like(_x) * dz

        return [dx]