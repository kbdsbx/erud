import erud.cg.edge as edge
from erud.cg.payload import payload

class ComputationNode :
    #### 数据域
    data : payload = None

    #### 前向传播路径
    fFirstEdge : "edge.ComputationEdge" = None
    
    #### 反向传播路径
    bFirstEdge : "edge.ComputationEdge" = None

    def __init__ (self, pl : payload = None) :
        self.data = pl

    def __str__(self) :
        if self.data == None :
            return ""
        else :
            return self.data