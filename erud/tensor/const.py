from erud.cg.payload import payload
import numpy as np

class const (payload) :
    __data : any = None

    def __init__ (self, d) :
        self.__data = d
    
    # 前向传播时，常量提供值
    # 常量值会向后分发（沿着出度）给所有使用此常量的表达式
    def fprop(self) -> any:
        return self.__data

    # 反向传播时，对当前常量求导值为单位张量
    def bprop(self, values) -> any:
        return np.sum(values)

