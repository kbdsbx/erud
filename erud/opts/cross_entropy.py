from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
from erud._utils import epsilon as eps

# 交叉熵损失函数
class cross_entropy(payload) :
    __yhat : any
    __y : any

    def fprop(self, yhat, y) -> any :
        # 防止上溢，使yhat in (0, 1)，不包括0和1
        yhat -= ((yhat == np.ones_like(yhat)) * eps)
        yhat += ((yhat == np.zeros_like(yhat)) * eps)

        self.__yhat = yhat
        self.__y = y
        
        return -1 * (y * np.log(yhat) + (1. - y) * np.log(1. - yhat))
    
    def bprop(self, dz) -> list[any] :
        _yhat = self.__yhat
        _y = self.__y
        
        if np.any(_yhat <= 0) :
            print(_yhat)

        if np.any(_yhat >=1) :
            print(_yhat)

        dyhat = -1. * _y / _yhat + ((1. - _y) / (1. - _yhat))
        dy = -1. * np.log(_yhat) + np.log(1. - _yhat)

        dyhat = dyhat * dz
        dy = dy * dz

        return [dyhat, dy]