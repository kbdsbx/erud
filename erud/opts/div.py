from erud.cg.payload import payload
from erud.opts._utils import partial_sum
import numpy as np

class div (payload) :
    # 缓存
    __x : any
    __y : any

    # 前向传播时，计算z = x / y，并缓存
    def fprop(self, x, y) -> any :
        self.__x = x
        self.__y = y

        return np.divide(x, y)

    # 反向传播时，dz/dx = 1 / y; dz/dy = -x / (y^2)
    def bprop(self, dz) -> list[any]:
        _x = self.__x
        _y = self.__y

        dx = np.divide(dz, _y)
        dx = partial_sum(dx, _x)

        dy = dz * -_x * np.divide(1, np.power(_y, 2))
        dy = partial_sum(dy, _y)

        return [dx, dy]
