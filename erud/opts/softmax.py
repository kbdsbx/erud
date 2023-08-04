from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

class softmax(payload) :
    __a : any
    __axises : any

    def __init__(self, axises) : 
        self.__axises = axises

    def fprop(self, x) -> any :
        # ex = np.exp(x)

        _ax = self.__axises
        
        # ex_sum = np.sum(ex, axis = ax, keepdims = True)

        # self.__ex = ex
        # self.__axises = ax
        # self.__ex_sum = ex_sum

        # return ex / ex_sum

        # 增加了偏置防止数值差距过大导致结果均值离谱
        _v = np.exp(x - np.max(x, axis = _ax, keepdims = True))
        _a = _v / np.sum(_v, axis = _ax, keepdims = True)

        self.__a = _a

        return _a
    
    # 虽然公式和我推导的大致相同
    # 但正确性有待考证
    def bprop(self, dz) -> list[any] :

        # https://zhuanlan.zhihu.com/p/315749528
        # _ex = self.__ex
        # _axiese = self.__axises
        # _ex_sum = self.__ex_sum

        # dx = (dz / _ex_sum - np.sum(_ex * dz, axis = _axiese, keepdims = True) / (_ex_sum * _ex_sum)) * _ex

        # return [dx, np.zeros(_ex.shape)]

        # https://zhuanlan.zhihu.com/p/67759205
        _a = self.__a
        _axises = self.__axises
        # dx = _a * (dz - np.einsum('ij,ij->i', dz, _a, optimize = True) )
        dx = _a * (dz - np.sum(dz * _a, axis = _axises) )

        daxies = 0
        if isinstance(_axises, list) :
            daxies = np.zeros(_axises.shape)
                   
        return [dx, daxies]


