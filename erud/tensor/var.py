from erud.cg.payload import payload
import numpy as np

class var (payload) :
    __data : any = None
        
    @property
    def data (self) -> any :
        return self.__data
    
    # 变量赋值
    @data.setter
    def data(self, d) :
        self.__data = d

    def __init__ (self, d) :
        self.__data = d
    
    # 前向传播时，变量提供值
    # 变量值会向后分发（沿着出度）给所有使用此变量的表达式
    def fprop(self) -> any:
        return self.__data

    # 反向传播时，dz/dc = 0
    def bprop(self, dz = None) -> any:
        if ( isinstance(self.__data, np.ndarray) ) :
            return np.zeros_like(self.__data)
        else :
            return 0
    

