from erud.cg.payload import payload
import numpy as np

# 交叉熵损失函数
class cross_entropy(payload) :
    __yhat : any
    __y : any

    def fprop(self, yhat, y) -> any :
        self.__yhat = yhat
        self.__y = y

        return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat))
    
    def bprop(self, dz) -> list[any] :
        _yhat = self.__yhat
        _y = self.__y

        dyhat = -_y / _yhat + (1 - _y) / (1 - _yhat)
        dy = -np.log(_yhat) + np.log(1 - _yhat)

        dyhat = dyhat * dz
        dy = dy * dz

        return [dyhat, dy]