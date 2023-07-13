from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

# 结果量，存放计算图的结果
class rest (payload) :
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
    

    # 导出
    def exports (self) -> object:
        name = (super(rest, self).exports())["name"]

        exp = {
            'type' : 'rest',
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
        super(rest, self).imports(value)
        self.__data = value['payload']
        
    

