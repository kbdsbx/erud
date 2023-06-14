from erud.cg.edge import ComputationEdge as edge
from erud.cg.node import ComputationNode as node
from erud.cg.errors import *

# 计算图
# 使用十字链表存储
class ComputationGraph: 
    __nodes : list[node] = []

    def __init__ (self) :
        self.__nodes = []

    # 节点数量
    @property
    def hasNode(self) :
        return len(self.__nodes)
    
    # 增加节点
    def insertNode(self, node : node):
        for _n in self.__nodes :
            if _n is node :
                raise NodeRepeatError('This node is exists.')
        self.__nodes.append(node)


    # 增加边
    def addEdge(self, nodeX : node, nodeY : node):
        try :
            self.__nodes.index(nodeX)
            self.__nodes.index(nodeY)
        except :
            raise NodeNotFindError('Can not find relative node in computation graph when adds a edge.')

        _insert_edge = edge(nodeX, nodeY)

        if nodeX.fFirstEdge is None :
            nodeX.fFirstEdge = _insert_edge
        else :
            _x_edge = nodeX.fFirstEdge

            if _x_edge.fNode is nodeY :
                raise EdgeRepeatError('this edge is exists.')

            while _x_edge.fNextEdge is not None :
                if _x_edge.fNode is nodeY :
                    raise EdgeRepeatError('this edge is exists.')
                _x_edge = _x_edge.fNextEdge
            else :
                if _x_edge.fNode is nodeY :
                    raise EdgeRepeatError('this edge is exists.')
                _x_edge.fNextEdge = _insert_edge

        if nodeY.bFirstEdge is None :
            nodeY.bFirstEdge = _insert_edge
        else :
            _y_edge = nodeY.bFirstEdge

            while _y_edge.bNextEdge is not None:
                _y_edge = _y_edge.bNextEdge
            else :
                _y_edge.bNextEdge = _insert_edge
    

    # 删除节点
    def removeNode(self, node : node):
        try :
            self.__nodes.index(node)
        except :
            raise NodeNotFindError('Can not find node in computation graph.')

        # 遍历删除node边

        _f_nodes : list[node] = []
        _f = node.fFirstEdge

        # 找到所有起点为此节点的其他节点
        while _f is not None :
            _f_nodes.append(_f.fNode)
            _f = _f.fNextEdge
        
        # 遍历删除边
        for _n in _f_nodes:
            _b_edge = _n.bFirstEdge

            if _b_edge.bNode is node :
                _n.bFirstEdge = _b_edge.bNextEdge
            else :
                while _b_edge.bNextEdge is not None :
                    if _b_edge.bNextEdge.bNode is node :
                        _b_edge.bNextEdge = _b_edge.bNextEdge.bNextEdge
                        break
                    _b_edge = _b_edge.bNextEdge
        

        _b_nodes : list[node] = []
        _b = node.bFirstEdge

        # 找到所有终点为此节点的其他节点
        while _b is not None :
            _b_nodes.append(_b.bNode)
            _b = _b.bNextEdge

        # 遍历删除边
        for _n in _b_nodes :
            _f_edge = _n.fFirstEdge

            if _f_edge.fNode is node :
                _n.fFirstEdge = _f_edge.fNextEdge
            else :
                while _f_edge.fNextEdge is not None :
                    if _f_edge.fNextEdge.fNode is node :
                        _f_edge.fNextEdge = _f_edge.fNextEdge.fNextEdge
                        break
                    _f_edge = _f_edge.fNextEdge


        # 删除节点
        self.__nodes.remove(node)


    # 删除边
    def removeEdge(self, nodeX : node, nodeY : node) :
        try :
            self.__nodes.index(nodeX)
            self.__nodes.index(nodeY)
        except :
            raise NodeNotFindError('Can not find relative node in computation graph when removes a edge.')
        
        _x_edge = nodeX.fFirstEdge


        if _x_edge is None :
            raise EdgeNotFindError('Can not find edge in computation graph.')
        elif _x_edge.fNode is nodeY :
            nodeX.fFirstEdge = _x_edge.fNextEdge
        else :
            while _x_edge.fNextEdge is not None :
                if _x_edge.fNextEdge.fNode is nodeY :
                    _x_edge.fNextEdge = _x_edge.fNextEdge.fNextEdge
                    break
                _x_edge = _x_edge.fNextEdge
            else :
                raise EdgeNotFindError('Can not find edge in computation graph.')
        
        _y_edge = nodeY.bFirstEdge

        if _y_edge is None :
            raise EdgeNotFindError('Can not find edge in computation graph.')
        elif _y_edge.bNode is nodeX :
            nodeY.bFirstEdge = _y_edge.bNextEdge
        else :
            while _y_edge.bNextEdge is not None :
                if _y_edge.bNextEdge.bNode is nodeX :
                    _y_edge.bNextEdge = _y_edge.bNextEdge.bNextEdge
                    break
                _y_edge = _y_edge.bNextEdge
            else :
                raise EdgeNotFindError('Can not find edge in computation graph.')


    # 针对前向传播
    # 获得子图中入度为0的点（源点）
    # 如果点的入边数为零，或所有入边都有负载，则此点入度为0
    def _getStartingNodeInForwardPropagation(self, nodes) -> list(node):
        res = []
        for n in nodes:
            p = n.bFirstEdge
            while p is not None:
                # 如果此入边的负载存在，则边不计入入度
                if p.carry is not None :
                    p = p.bNextEdge
                # 如果此入边的负载不存在，则计入入度，此点的入度大于零
                else :
                    break
            else :
                res.append(n)

        return res
    

    # 针对前向传播
    # 获得子图中出度为0的点（汇点）
    def _getEndedNodeInForwardPropagation(self, nodes) -> list(node) :
        res = []
        for n in nodes:
            if n.fFirstEdge is None :
                res.append(n)
        
        return res



    # 前向传播
    # 此算法目前只适用于有向无环图，对于可能带环状结构的循环网络不兼容
    def fprop (self) :
        # 还未计算的节点
        apply_node = self.__nodes

        # 从还未计算的节点中筛选出入度为0的节点
        s_node = self._getStartingNodeInForwardPropagation(apply_node)
        # 将入度为0的节点从未计算节点中去掉
        apply_node = list(set(apply_node).difference(set(s_node)))

        while len(s_node) :
            for n in s_node :
                args = []
                # 从此节点的入边中取得所有运载
                p = n.bFirstEdge
                while p is not None :
                    args.append(p.carry)
                    # 取得运载后将运载清空，方便后续计算
                    p.carry = None
                    p = p.bNextEdge

                # 将运载投入节点负载进行计算
                res = n.fprop(*args)

                q = n.fFirstEdge
                # 将计算结果沿着出边向后传递，即分发赋值给所有出边的运载
                while q is not None:
                    q.carry = res
                    q = q.fNextEdge
            
            # 去掉图中所有入度为0的点后，会产生新的入度为0的点，继续此步骤直到遍历完所有点
            # 从还未计算的节点中筛选出入度为0的节点
            s_node = self._getStartingNodeInForwardPropagation(apply_node)
            # 将入度为0的节点从未计算节点中去掉
            apply_node = list(set(apply_node).difference(set(s_node)))
        
        # 计算完成后将所有结果点的值收集并返回
        res_node = self._getEndedNodeInForwardPropagation(self.__nodes)
        return [n.data for n in res_node]
            


                

    # 十字链表法输出计算图结构
    def __str__ (self) :
        str = "\n"
        for node in self.__nodes :
            str += "| %s |\t" %(node.data)
            p = node.fFirstEdge
            while p != None:
                str += "[->%s] " %(p.fNode.data)
                p = p.fNextEdge

            p = node.bFirstEdge
            while p != None:
                str += "[<-%s] " %(p.bNode.data)
                p = p.bNextEdge

            str += "\n"
        return str