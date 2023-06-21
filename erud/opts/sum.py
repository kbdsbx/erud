from erud.cg.payload import payload
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