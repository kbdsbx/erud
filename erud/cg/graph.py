from erud.cg.edge import ComputationEdge as edge
from erud.cg.node import ComputationNode as node
from erud.errors import *

# 计算图
# 使用十字链表存储
class ComputationGraph: 
    __nodes : list[node] = []

    def __init__ (self) :
        self.__nodes = []

    # 节点数量
    @property
    def nodeCount(self) :
        return len(self.__nodes)

    # 是否拥有节点
    def hasNode(self, node : node) :
        for _n in self.__nodes :
            if _n is node :
                return True
        return False

    @property
    def nodes(self) -> list[node] :
        return self.__nodes
    
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

            # 允许多条边了现在
            # if _x_edge.fNode is nodeY :
            #     raise EdgeRepeatError('this edge is exists.')

            while _x_edge.fNextEdge is not None :
                # 允许多条边了现在
                # if _x_edge.fNode is nodeY :
                #     raise EdgeRepeatError('this edge is exists.')
                _x_edge = _x_edge.fNextEdge
            else :
                # 允许多条边了现在
                # if _x_edge.fNode is nodeY :
                #     raise EdgeRepeatError('this edge is exists.')
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
        # 这个大概是有问题的
        # 待更改
        for _n in _f_nodes:
            _b_edge = _n.bFirstEdge

            if _b_edge.bNode is node :
                _n.bFirstEdge = _b_edge.bNextEdge
            else :
                while _b_edge.bNextEdge is not None :
                    if _b_edge.bNextEdge.bNode is node :
                        _b_edge.bNextEdge = _b_edge.bNextEdge.bNextEdge
                    else :
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
                    else :
                        _f_edge = _f_edge.fNextEdge


        # 删除节点
        self.__nodes.remove(node)


    # 删除边
    # 返回删除边的数量
    def removeEdge(self, nodeX : node, nodeY : node) -> int:
        try :
            self.__nodes.index(nodeX)
            self.__nodes.index(nodeY)
        except :
            raise NodeNotFindError('Can not find relative node in computation graph when removes a edge.')
        
        # 删除边的数量
        # 由于前向边和反向边数量相同，所以只计算一侧
        ct = 0

        # 伪头，方便计算
        _h = edge(None, None)
        _h.fNextEdge = nodeX.fFirstEdge
        _p = _h

        if _p.fNextEdge is None :
            raise EdgeNotFindError('Can not find this edge in computation graph.')
        
        while _p.fNextEdge is not None :
            if _p.fNextEdge.fNode is nodeY :
                _p.fNextEdge = _p.fNextEdge.fNextEdge
                ct += 1
            else :
                _p = _p.fNextEdge
        
        nodeX.fFirstEdge = _h.fNextEdge

        # elif _x_edge.fNode is nodeY :
        #     nodeX.fFirstEdge = _x_edge.fNextEdge
        #     ct += 1
        # else :
        #     while _x_edge.fNextEdge is not None :
        #         if _x_edge.fNextEdge.fNode is nodeY :
        #             _x_edge.fNextEdge = _x_edge.fNextEdge.fNextEdge
        #             ct += 1
        #         else :
        #             _x_edge = _x_edge.fNextEdge
    
        _h = edge(None, None)
        _h.bNextEdge = nodeY.bFirstEdge
        _p = _h

        if _p.bNextEdge is None :
            raise EdgeNotFindError('Can not find this edge in computation graph.')
        
        while _p.bNextEdge is not None :
            if _p.bNextEdge.bNode is nodeX :
                _p.bNextEdge = _p.bNextEdge.bNextEdge
            else :
                _p = _p.bNextEdge
        
        nodeY.bFirstEdge = _h.bNextEdge
        
        # _y_edge = nodeY.bFirstEdge

        # if _y_edge is None :
        #     raise EdgeNotFindError('Can not find this edge in computation graph.')
        # elif _y_edge.bNode is nodeX :
        #     nodeY.bFirstEdge = _y_edge.bNextEdge
        # else :
        #     while _y_edge.bNextEdge is not None :
        #         if _y_edge.bNextEdge.bNode is nodeX :
        #             _y_edge.bNextEdge = _y_edge.bNextEdge.bNextEdge
        #         else :
        #             _y_edge = _y_edge.bNextEdge

        if 0 == ct :
            raise EdgeNotFindError('Can not find this edge in computation graph.')
        
        return ct

    # 针对前向传播
    # 获得子图中入度为0的点（源点）
    # 如果点的入边数为零，或所有入边都有负载，则此点入度为0
    def _getStartingNodeInForwardPropagation(self, nodes : list[node]) -> list[node]:
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
    def _getEndedNodeInForwardPropagation(self, nodes : list[node]) -> list[node] :
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

    # 针对反向传播
    # 获得子图中入度为0的点（源点）
    # 如果点的入边数为零，或所有入边都有负载，则此点入度为0
    def _getStartingNodeInBackwardPropagation(self, nodes: list[node]) -> list[node] :
        res = []
        for n in nodes:
            p = n.fFirstEdge
            while p is not None:
                # 如果此入边的负载存在，则边不计入入度
                if p.carry is not None :
                    p = p.fNextEdge
                # 如果此入边的负载不存在，则计入入度，此点的入度大于零
                else :
                    break
            else :
                res.append(n)

        return res
    

            

    # 反向传播
    # 此算法目前只适用于有向无环图，对于可能带环状结构的循环网络不兼容
    def bprop (self) : 
        apply_node = self.__nodes

        # 从还未计算的节点中筛选出入度为0的节点
        s_node : list[node] = self._getStartingNodeInBackwardPropagation(apply_node)
        # 将入度为0的节点从未计算节点列表中去掉
        apply_node = list(set(apply_node).difference(set(s_node)))

        while len(s_node) :
            for n in s_node :
                # 从此节点的入边中获取所有运载
                # 一般来说，运算符的入边只有一条，只有变量的入边有一或多条，表明此变量在前向计算中发生了分发
                # 但是没关系，反向传播时变量的每一条入边的运载都是这个变量的偏导
                # 只需要将所有偏导加起来就是这个这个变量的导数（指代价函数J对变量w的导数dJ/dw）
                args = None
                p = n.fFirstEdge
                while p is not None :
                    if args is None :
                        args = p.carry
                    else :
                        args += p.carry
                    # 取得运载后将运载清空，方便后续计算
                    p.carry = None
                    p = p.fNextEdge
                
                # 将运载投入节点负载计算反向传播，根据不同的负载类型会有一到多个结果
                res = n.bprop(args)

                i = 0
                q = n.bFirstEdge
                # 将结果按顺序分发给每一条出边
                while q is not None:
                    q.carry = res[i]
                    q = q.bNextEdge
                    i += 1
            
            # 去掉图中所有入度为0的点后，会产生新的入度为0的点，继续此步骤直到遍历完所有点
            # 从还未计算的节点中筛选出入度为0的节点
            s_node = self._getStartingNodeInBackwardPropagation(apply_node)
            # 将入度为0的节点从未计算节点中去掉
            apply_node = list(set(apply_node).difference(set(s_node)))
        
        # 反向传播无需返回值，因为最后对常数或变量求导的值都为0，整体偏导即为0
        

    # 设置值
    # 根据node.data.name查找并设置节点的值
    def setData (self, name : str, value : any) :
        for n in self.__nodes :
            if n.data.name == name :
                n.data.data = value
                break
        else :
            raise NodeNotFindError('Can not find node named %s.' % (name))
    
    # 获取值
    def getData (self, name: str ) -> any :
        for n in self.__nodes :
            if n.data.name == name :
                return n.data.data
        else :
            raise NodeNotFindError('Can not find node named %s.' % (name))

    # 设置更新函数
    def setUpdateFunc(self, name : str, func : any) :
        for n in self.__nodes :
            if n.data.name == name :
                n.data.update_func = func
                break
        else :
            raise NodeNotFindError('Can not find node named %s.' % (name))

    # 十字链表法输出计算图结构
    def __str__ (self) :
        str = "\n"
        for node in self.__nodes :
            str += "| %s |\t" %(node.code)
            p = node.fFirstEdge
            while p != None:
                str += "[->%s] " %(p.fNode.code )
                p = p.fNextEdge

            p = node.bFirstEdge
            while p != None:
                str += "[<-%s] " %(p.bNode.code )
                p = p.bNextEdge

            str += "\n"
        return str