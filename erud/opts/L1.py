from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

class L1(payload) :
    __a : any = None

    def fprop(self, yhat, y) -> any :
        self.__a = y - yhat

        return np.sum(np.abs(self.__a))
    
    def bprop(self, dz) -> list[any] :
        _a = self.__a

        dyhat = _a < 0
        dy = _a > 0

        dyhat = dyhat * dz
        dy = dy * dz

        return [dyhat, dy]
