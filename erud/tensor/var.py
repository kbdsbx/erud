from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

class var (payload) :
    __data : any = None
    __update_func = None
        
    @property
    def data (self) -> any :
        return self.__data
    
    # 变量赋值
    @data.setter
    def data(self, d) :
        self.__data = d
    
    # 更新函数
    @property
    def update_func(self) -> any:
        return self.__update_func
    
    @update_func.setter
    def update_func(self, d) :
        self.__update_func = d

    # d: 负载
    # update_fun(z, dz) : 参数的更新方法
    def __init__ (self, d, update_func = None) :
        self.__data = d
        self.__update_func = update_func
    
    # 前向传播时，变量提供值
    # 变量值会向后分发（沿着出度）给所有使用此变量的表达式
    def fprop(self) -> any:
        return self.__data

    # 反向传播时，dz/dc = 0
    def bprop(self, dz = None) -> list[any]:
        # 反向传播更新参数
        if dz is not None and self.__update_func is not None :
            res = self.__update_func( self.__data, dz )
            if res is not None :
                self.__data = res

        if isinstance(self.__data, np.ndarray) :
            return [np.zeros_like(self.__data)]
        else :
            return [0]
    
    def __str__ (self) :
        if self.__data is not None :
            return str(self.__data)
        else :
            return ""
    
    # 导出
    def exports (self) -> object:
        name = (super(var, self).exports())["name"]

        exp = {
            'type' : 'var',
            'name' : name,
            'payload' : '',
        }

        if isinstance(self.__data, np.ndarray) :
            exp['payload'] = self.__data.tolist()
        else :
            exp['payload'] = self.__data
        
        return exp
    
    # 导入
    def imports (self, value) :
        super(var, self).imports(value)
        self.__data = value['payload']
        
    

