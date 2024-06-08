from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
from erud._utils import epsilon as eps

class softmax_cross_entropy(payload) :
    __axises : any
    __yhat : any = None
    __y : any = None

    def __init__(self, axises) : 
        self.__axises = axises

    def fprop(self, x, y) -> any :
        _ax = self.__axises
        # 均值归零
        _a = np.exp(x - np.max(x, axis = _ax, keepdims = True))
        _yhat = _a / np.sum(_a, axis = _ax, keepdims = True)
        self.__yhat = _yhat
        self.__y = y
        
        _lost = -1 * np.sum(y * np.log(_yhat + eps), axis = _ax, keepdims = True)

        return _lost
    
    def bprop(self, dz) -> list[any] :
        _yhat = self.__yhat
        _y = self.__y

        dx = (_yhat - _y) * dz
        dy = np.zeros_like(_y)

        return [dx, dy]


