from erud.cg.payload import payload
import numpy as np
from erud.errors import *

class threshold (payload) :
    __threshold : any

    @property
    def threshold(self) :
        return self.__threshold

    def __init__(self, threshold = 0.5) :
        self.__threshold = threshold


    def fprop(self, x) -> any :
        return x > self.__threshold
    
    def bprop(self, dz) -> list[any] :
        raise UnsupportedError('Can not call function bprop from "threshold".')