from erud.cg.payload import payload
import numpy as np
from erud.errors import *

# 精准度
class accuracy (payload) :
    def fprop(self, yhat, y) -> any :
        return np.mean(yhat == y) * 100

    def bprop(self, dz) -> list[any] :
        raise UnsupportedError('Can not call function bprop from "accuracy".')