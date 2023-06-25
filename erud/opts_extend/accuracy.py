from erud.cg.payload import payload
import numpy as np

# 精准度
class accuracy (payload) :
    __yhat : any = None
    __y : any = None

    def fprop(self, yhat, y) -> any :
        return 100 - np.mean(np.abs(yhat - y) * 100)

    def bprop(self, dz) -> list[any] :
        return [np.zeros_like(self.__yhat), np.zeros_like(self.__y)]