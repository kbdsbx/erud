from erud.cg.payload import payload
import numpy as np
from erud.errors import *

class max_index(payload) :
    __axises : any = None

    def __init__(self, axises) :
        self.__axises = axises

    def fprop(self, x) -> any :
        _ax = self.__axises
        mx = np.argmax(x, axis = _ax, keepdims = True)

        return mx
    
    def bprop(self, dz) -> list[any] :
        raise UnsupportedError('Can not call function bprop from "accuracy".')