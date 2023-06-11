import erud.edge as edge

class ComputationNode :
    #### 数据域
    info : None

    #### 前向传播路径
    fFirstEdge : "edge.ComputationEdge" = None
    
    #### 反向传播路径
    bFirstEdge : "edge.ComputationEdge" = None

    def __init__ (self, paylaod = None) :
        self.info = paylaod

    def __str__(self) :
        if self.info == None :
            return ""
        else :
            return self.info