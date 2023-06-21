from erud.cg.payload import payload
import numpy as np

class sigmoid(payload) :
    __a : any = None

    # a = 1 / (1 + e^-x)
    def fprop(self, x) -> any :
        # 缓存中间变量用于反向传播
        self.__a = 1 / ( 1 + np.exp(-x))

        return self.__a
    
    # dz/dx = a(1 - a)
    def bprop(self, dz) -> list[any] :
        _a = self.__a

        return [_a * (1 - _a) * dz]