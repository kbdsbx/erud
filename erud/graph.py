from erud.edge import ComputationEdge as edge
from erud.node import ComputationNode as node

# 计算图
# 使用十字链表存储
class ComputationGraph: 
    __nodes = []

    def __init__ (self) :
        self.__nodes = []

    # 节点数量
    @property
    def hasNode(self) :
        return len(self.__nodes)
    
    # 增加节点
    def insertNode(self, node):
        self.__nodes.append(node)

    # 增加边
    def addEdge(self, nodeX : node, nodeY : node):
        try :
            self.__nodes.index(nodeX)
            self.__nodes.index(nodeY)
        except :
            raise IndexError('Can not find relative node in computation graph when adds a edge.')

        _insert_edge = edge(nodeX, nodeY)

        if nodeX.fFristEdge == None :
            nodeX.fFristEdge = _insert_edge
        else :
            _x_edge = nodeX.fFristEdge

            while _x_edge.fNextEdge is not None :
                _x_edge = _x_edge.fNextEdge
            else :
                _x_edge.fNextEdge = _insert_edge

        if nodeY.bFristEdge is None :
            nodeY.bFristEdge = _insert_edge
        else :
            _y_edge = nodeX.bFristEdge

            while _y_edge.bNextEdge is not None:
                _y_edge = _y_edge.bNextEdge
            else :
                _y_edge.bNextEdge = _insert_edge
    
    # 删除节点
    def removeNode(self, node):
        # 遍历删除node边

        # 删除节点
        self.__nodes.remove(node)

    def __str__ (self) :
        str = "\n"
        for node in self.__nodes :
            str += "| %s |\t" %(node.info)
            p = node.fFristEdge
            while p != None:
                str += "[->%s] " %(p.fNode.info)
                p = p.fNextEdge

            p = node.bFristEdge
            while p != None:
                str += "[<-%s]" %(p.bNode.info)
                p = p.bNextEdge

            str += "\n"
        return str