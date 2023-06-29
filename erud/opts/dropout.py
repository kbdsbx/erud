from erud.cg.payload import payload
import numpy as np

class dropout (payload) :
    # 掩层
    __mash : any
    # 概率
    __posi : any

    def fprop(self, x, posi) -> any :
        self.__posi = posi
        
        mash = np.random.rand(*x.shape) < posi
        self.__mash = mash

        return x * mash / posi
    
    def bprop(self, dz) -> list[any] :
        _mash = self.__mash
        _posi = self.__posi

        dx = dz * _mash / _posi

        return [dx, 0.]

