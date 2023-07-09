import erud.cg.edge as edge
from erud.cg.payload import payload
import time

class ComputationNode :
    #### 数据域
    __data : payload = None

    #### 代码
    __code : str = ''

    #### debuger
    # 前向传播上一次计算花费的时间
    __ftimespend : int = 0
    # 前向传播计算花费的总时间
    __ftimetotal : int = 0
    # 反向传播上一次计算花费的时间
    __btimespend : int = 0
    # 反向传播计算花费的总时间
    __btimetotal : int = 0

    @property
    def data (self) :
        return self.__data
    
    @property
    def code (self) :
        return self.__code
    
    @property
    def ftimespend(self) :
        return self.__ftimespend
    
    @property
    def ftimetotal(self) :
        return self.__ftimetotal

    @property
    def btimespend(self) :
        return self.__btimespend
    
    @property
    def btimetotal(self) :
        return self.__btimetotal

    #### 前向传播路径
    fFirstEdge : "edge.ComputationEdge" = None
    
    #### 反向传播路径
    bFirstEdge : "edge.ComputationEdge" = None

    # 计算前向传播
    def fprop(self, *args) :
        tic = time.process_time()

        res = self.__data.fprop(*args)

        toc = time.process_time()
        spt = (toc - tic) * 1000
        self.__ftimespend = spt
        self.__ftimetotal += spt

        return res
    
    # 计算反向传播
    def bprop(self, args) :
        tic = time.process_time()

        res = self.__data.bprop(args)

        toc = time.process_time()
        spt = (toc - tic) * 1000
        self.__btimespend = spt
        self.__btimetotal += spt

        return res

    def __init__ (self, pl : payload = None, code : str = None) :
        self.__data = pl
        self.__code = code

    def __str__(self) :
        if self.__data == None :
            return ""
        else :
            return self.__data