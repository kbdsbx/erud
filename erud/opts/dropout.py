from erud.cg.payload import payload
import numpy as np

class dropout (payload) :
    # 掩层
    __mash : any
    # 概率
    __posi : any

    def __init__(self, posi = 0.5) :
        self.__posi = posi

    def fprop(self, x) -> any :
        _posi = self.__posi
        
        mash = np.random.rand(*x.shape) < _posi
        self.__mash = mash

        return x * mash / _posi
    
    def bprop(self, dz) -> list[any] :
        _mash = self.__mash
        _posi = self.__posi

        dx = dz * _mash / _posi

        return [dx, 0.]

