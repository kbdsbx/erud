from erud.cg.payload import payload
import numpy as np

# 结果量，存放计算图的结果
class ret (payload) :
    __data : any = None

    @property
    def data(self) -> any :
        return self.__data
    
    # 前向传播时，存储计算得到的最终值
    # 一般来说结果节点是计算图或计算子图的汇点，但如果不是汇点，也可以当成中间变量向后传递值
    def fprop(self, d) -> any :
        self.__data = d
        return self.__data

    # 反向传播时，dz/dz = 1
    def bprop(self, dz = None) -> list[any] :
        if isinstance(self.__data, np.ndarray) :
            return [np.ones_like(self.__data)]
        else :
            return [1]