from erud.cg.payload import payload
import numpy as np

class var (payload) :
    __data : any = None
    __update_fun = None
        
    @property
    def data (self) -> any :
        return self.__data
    
    # 变量赋值
    @data.setter
    def data(self, d) :
        self.__data = d

    # d: 负载
    # update_fun(z, dz) : 参数的更新方法
    def __init__ (self, d, update_fun = None) :
        self.__data = d
        self.__update_fun = update_fun
    
    # 前向传播时，变量提供值
    # 变量值会向后分发（沿着出度）给所有使用此变量的表达式
    def fprop(self) -> any:
        return self.__data

    # 反向传播时，dz/dc = 0
    def bprop(self, dz = None) -> list[any]:
        # 反向传播更新参数
        if dz is not None and self.__update_fun is not None :
            self.__data = self.__update_fun( self.__data, dz )

        if isinstance(self.__data, np.ndarray) :
            return [np.zeros_like(self.__data)]
        else :
            return [0]
    
    def __str__ (self) :
        if self.__data is not None :
            return str(self.__data)
        else :
            return ""
    

