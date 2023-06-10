import erud.node as node

# 计算图的边
# 从数据结构来讲，fNode是出度，bNode是入度
# 从神经网络来讲，fNode是前向传播的指向节点，bNode是反向传播的指向节点
class ComputationEdge :
    #### 前向传播
    # 边指向的节点
    fNode : "node.ComputationNode" = None
    # 下一条边
    fNextEdge : "ComputationEdge" = None


    #### 反向传播
    # 边指向的节点
    bNode : "node.ComputationNode" = None
    # 下一条边
    bNextEdge : "ComputationEdge" = None

    # 边信息
    weight = None

    def __init__(self, nodeX : node.ComputationNode, nodeY : node.ComputationNode) :
        self.fNode = nodeY
        self.bNode = nodeX