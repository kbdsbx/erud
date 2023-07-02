from erud.cg.payload import payload
import numpy as np

class threshold (payload) :
    __x : any
    __threshold : any

    @property
    def threshold(self) :
        return self.__threshold

    def __init__(self, threshold = 0.5) :
        self.__threshold = threshold


    def fprop(self, x) -> any :
        self.__x = x
        return x > self.__threshold
    
    def bprop(self, dz) -> list[any] :
        return [np.zeros_like(self.__x), 0]