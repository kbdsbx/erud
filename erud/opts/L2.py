from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

class L2(payload) :
    __a : any

    def fprop(self, yhat, y) -> any :
        _a = y - yhat

        self.__a = _a

        return np.sum(_a * _a)
    
    def bprop(self, dz) -> list[any] :
        _a = self.__a

        dyhat = -2 * _a
        dy = 2 * _a

        dyhat = dyhat * dz
        dy = dy * dz

        return [dyhat, dy]

