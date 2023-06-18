import erud.cg.edge as edge
from erud.cg.payload import payload

class ComputationNode :
    #### 数据域
    __data : payload = None

    #### 代码
    __code : str = None

    @property
    def data (self) :
        return self.__data
    
    @property
    def code (self) :
        return self.__code

    #### 前向传播路径
    fFirstEdge : "edge.ComputationEdge" = None
    
    #### 反向传播路径
    bFirstEdge : "edge.ComputationEdge" = None

    # 计算前向传播
    def fprop(self, *args) :
        return self.__data.fprop(*args)
    
    # 计算反向传播
    def bprop(self, args) :
        return self.__data.bprop(args)

    def __init__ (self, pl : payload = None, code : str = None) :
        self.__data = pl
        self.__code = code

    def __str__(self) :
        if self.__data == None :
            return ""
        else :
            return self.__data