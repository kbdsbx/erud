from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

class const (payload) :
    __data : any = None

    @property
    def data (self) -> any :
        return self.__data

    def __init__ (self, d) :
        self.__data = d
    
    # 前向传播时，常量提供值
    # 常量值会向后分发（沿着出度）给所有使用此常量的表达式
    def fprop(self) -> any:
        return self.__data

    # 反向传播时，dz/dc = 0
    def bprop(self, dz = 0) -> list[any]:
        if isinstance(self.__data, np.ndarray) :
            return [np.zeros_like(self.__data)]
        else :
            return [0]
        
    
    # 导出
    def exports (self) -> object:
        name = (super(const, self).exports())["name"]

        exp = {
            'type' : 'const',
            'name' : name,
        }

        if isinstance(self.__data, np.ndarray) :
            exp['payload'] = self.__data.tolist()
        else :
            exp['payload'] = self.__data
        
        return exp
    
    # 导入
    def imports (self, value) :
        super(const, self).imports(value)
        v = value['payload']
        if isinstance(v, list) :
            self.__data = np.array(v)
        else :
            self.__data = v
        
    



