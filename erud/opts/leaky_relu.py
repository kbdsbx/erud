from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

class leaky_relu(payload) :
    __rate : float
    __mask : None

    def __init__ (self, rate = 0.1) :
        self.__rate = rate

    def fprop(self, x) -> any :
        _rate = self.__rate
        mask = (x < 0) * _rate + (x >= 0)
        self.__mask = mask

        return x * mask
    
    def bprop(self, dz) -> list[any]:
        _mask = self.__mask

        # 使用右导数，当xi为0时，dz/dxi = 1
        dx = dz * _mask
        return [dx]