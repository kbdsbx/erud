from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
from erud.errors import *

# 精准度
class accuracy (payload) :
    def fprop(self, yhat, y) -> any :
        _yhat = yhat.reshape(y.shape)
        # print('yhat', _yhat)
        # print('y', y)
        return np.mean(_yhat == y) * 100

    def bprop(self, dz) -> list[any] :
        raise UnsupportedError('Can not call function bprop from "accuracy".')