import erud.cg.node as node
from erud.cg.node import payload

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

    # 运载
    # 正在计算的节点会从上一层边中一次性取得所有运载并进行计算
    # 已计算完成的节点会将值分发给下一层中所有边的运载
    # 还未计算的节点出边的所有carry都为None，相似的，已经计算的节点出边的所有carry都不为None
    # 可以借此判断一个节点的入度是否为零，如果它所有入边的运载都不是None，则这个节点入度为0
    carry : any = None

    def __init__(self, nodeX : node.ComputationNode, nodeY : node.ComputationNode) :
        self.fNode = nodeY
        self.bNode = nodeX