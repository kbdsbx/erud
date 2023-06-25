from erud.cg.payload import payload
import numpy as np

class threshold (payload) :
    __x : any

    def fprop(self, x, thresholds) -> any :
        self.__x = x
        return x > thresholds
    
    def bprop(self, dz) -> list[any] :
        return [np.zeros_like(self.__x), 0]