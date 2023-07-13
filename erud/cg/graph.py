from erud.cg.edge import ComputationEdge as edge
from erud.cg.node import ComputationNode as node
from erud.errors import *
import json
import os

# 计算图
# 使用十字链表存储
class ComputationGraph: 
    """
    # 构建计算图

    计算图（Computation Graph）是神经网络的核心，用于构建神经网络计算的脚手架

    ## 术语辨析
    1. 用户：使用此框架的开发者
    1. 客户：使用框架的一般人
    1. 数据：用户或客户传入的样本数据，可能为数量、向量、矩阵或张量等
    1. 参数：用户传入的数据，可能为数量、向量、矩阵或张量等
    1. 变量：用户看不见的数据，隐藏在实现细节里
    1. 控制参数：不参与计算图运算，只是传递给操作符或者初始化函数来改变运算规则

    ## 计算图的特征

    1. 叶节点：存储张量的节点
        * 固定参数，包括数据和超参数
        * 学习参数

        > 所以需要一个变量区分参数类型，或者说是否允许更新

    2. 非页节点：存储操作和缓存
        
        组图阶段只存储“操作”，留一个或者两个缓存单位空着，正向传播的时候顺便计算要反向传播的变量，然后存在单位里，等反向传播的时候取出来用

        * 基本二元运算
            * 加add（完成）
            * 减sum（完成）
            * 乘mul（完成）
            * 除div（完成）
            * 矩阵乘matmul（完成）
            * ...
        * 复合二元操作
            * 卷积（完成）
            * 交叉熵cross_entropy（完成）
            * L1（完成）
            * L2（完成）
            * ...
        * 基本一元运算
            * 次方（用其他方式代替）
            * 张量求和sum（完成）
            * 总代价cost（完成）
            * ...
        * 复合一元操作
            * relu（完成）
            * sigmoid（完成）
            * tanh（完成）
            * softmax（完成）：softmax需要另一个操作数轴，所以虽然是一元操作，但在nous中的调用方法为``-> softmax 1 ->``或``-> softmax [0,1] ->``
            * <del>dropout（完成）: dropout需要一个概率值表示单元失效概率，在0~1之间`-> dropout 0.5 ->`</del>
            * batchnorm（完成）
            * 最大池化max_pool（完成）
            * 修改变量的维度flatten（完成）
            * ...
        * 复合三元操作
            * 也许有...
        * 带控制参数的一元或多元操作
            * softmax（完成）：softmax需要一个元组来区分样本集合`X`中，每个样本延哪个轴存放，并根据存放的轴号进行概率计算，调用方法为`-> softmax((1)) ->` 或`-> softmax((0, 1)) ->`
            * dropout（完成）: dropout需要一个概率值表示单元失效概率，在0~1之间，调用方法改为`-> dropout(0.5) ->`
            * softmax_cross_entropy（完成） : `softmax`及其交叉熵的复合运算符，调用方法为`-> softmax_cross_entropy(1) Y ->`
        
        > 一开始可以搞点基本操作组计算图，组完了再加复合操作做后期优化

        > 复合操作可以由基本操作组合，或者用户自己往里加，加的时候注意传入参数和计算反向传播就行

    3. 边：双向边

        <del>虽然计算图对前向传播来说是一个有向无环图</del>（见条目8），但毕竟反向传播还是要用边来找前向节点的，所以往图里塞节点的时候需要给后面的节点加前节点的指向

        |节点类型|前向入度|前向出度|
        |-|-|-|
        |叶节点|0|n，看参与的计算数量|
        |一元操作节点|1|1|
        |二元操作节点|2|1|
        |三元操作节点|3|1|
        |根节点，一般是损失函数|n，看参与的参数量|0|

        反向出入度和前向相反

    4. 组图

        搭建图：

        手动搭建“图”类型的数据结构，以图插入节点的方式构建图，接口的话可以设计成图插入，可以是普通的方式，也可以是运算符重载，前期可以只用图插入的方式测试，等前后向传播写完了再扩充接口

        如果以前向传播的有向无环图作为搭建基准，那么
        1. 准备叶结点（数据）
        2. 根据出度插入多个操作节点
        3. 操作节点中会带上叶结点（参数）
        4. 操作节点后续插入多个操作节点，直到根节点


    5. 前向传播

        前向传播使用有向无环图的拓扑排序作为遍历算法，算法如下

        1. 找到图中入度为0的节点，如果节点为叶节点则向后传递数据，如果节点为操作节点则计算操作、存储缓存数据、向后传递数据
        2. 将此节点从图中标记删除（不是真删），删除后的图中会产生新的入度为0的节点
        3. 重复1，直到标记删除根节点或图里节点全部删除（两个条件通常都一样）

        前向传播算法通常是能够访问整个图的算法，可以是图类的方法
        
        前向传播的调用条件是所有叶结点都有数据填充，包括表示样本的叶结点和表示参数的叶结点，后者通常为固定值（超参数）或随机值（参数）

    6. 验证

        图在搭建完成后可以有验证方法进行验证，主要验证目标是针对各个张量的计算是否合法，验证算法模拟前向传播的流程，但只做计算的断言，而不传递真实值

    7. 反向传播

        反向传播是全自动计算算法，通常在前向传播输出代价值后由用户显示调用，反向传播遵从导数计算法则：
        
        * 单路全导
        * 多路偏导
        * 分叉相加
        * 分段相乘

        其中需要参数会像前向传播一样，沿着边传递给上一个计算节点，计算节点拿到上一个参数以后会先计算本节点的导数，然后和之前的导数相乘（单路）或相乘相加（多路），继续传递给下一个节点

    8. 环

        <b style="color: red">循环神经网络是否存在环</b>，如果存在环的话还能不能正常工作，这个需要验证，但至少从基础结构来说，目前实现的图是允许环的存在的


        > 第一阶段就是构建整个计算图，增加简单四则运算作为操作节点，然后完成前向和反向传播。测试用例要包括变量、向量和张量，其中验证功能可以放在最后加

    9. 一些细节

        计算图节点node含有属性data，类型为payload，所有张量类和操作符类都继承自payload，并实现fprop方法和bprop方法，其中fprop会在计算图执行前向传播时陆续调用，bprop会在执行反向传播时调用。

        计算图边edge含有属性carry，类型为标量、向量、矩阵或张量，正在计算的节点会从入度边中一次性取得所有运载并进行计算，已计算完成的节点node会将值分发给出度中所有边的运载


        目前版本为两个节点增加重复边会抛出异常，即对x和y来说，x -> y只可以存在一条边（但可以同时存在x -> y和y -> x）。但考虑到如果有一个节点x，执行x * x运算，那么数据节点x指向操作节点*的边可以有多条。所以边的限制将在后续版本删除

    10. 复杂图

        最开始设想计算图是有向无环图简单图，后来觉得无环的限制可能不适用于循环网络或者递归网络，现在觉得简单图可能不适用于节点的灵活性，所以计算图的类型应该是……额……图。

        对于复杂图，两个之间的节点可以有不止一条边，比如对于表达式``X + X``来说，节点``X``可以指向加法节点``+``两次，存在两条边且不影响目前的计算图计算逻辑，只需要开放一下限制即可

        环是一个例外，环可以满足类似于``X = X + 1``的效果，也就是节点``X``与节点``1``相加后又指向了节点``X``，这对计算图是一个挑战，因为无环图可以通过拓扑排序的遍历算法一次性处理完所有节点，但一旦计算图中存在环，现有的拓扑排序将是死循环。

        一种方法是将环独立于计算图之外，让用户去处理循环网络的事情，保持计算图的可计算性，另一种方法就是规定循环的次数，但这样无疑是将整个图的处理变得复杂，在没有证据证明图中带有环能够使计算图的表达更加高效，更让人接受之前，还是先这样吧。

    """
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
    
    # 表格的方式输出时间开销
    def tableTimespend(self) :
        str = "\n"
        str += "| code \t | fprop last(s) \t | fprop total(s) \t | bprop last(s) \t | bprop total(s) \t|\n"
        for node in self.__nodes :
            str += "| %s \t | %.3f \t | %.3f \t | %.3f \t | %.3f \t |" %(node.code, node.ftimespend, node.ftimetotal, node.btimespend, node.btimetotal)
            str += "\n"
        
        return str
    

    # 将计算图数据保存在文件里
    def exports(self, path) :
        export_nodes = []
        for n in self.__nodes :
            obj = n.exports()
            export_nodes.append(obj)
        
        export_str = json.dumps(export_nodes)

        with open(path, "w", encoding="utf-8") as f :
            f.write(export_str)
        
    
    # 从文件中读取计算图
    def imports(self, path) :
        imports_str = ''
        with open(path, "r", encoding="utf-8") as f :
            imports_str = f.read()
        
        nodes = json.loads(imports_str)
        for i in range(len(nodes)) :
            self.__nodes[i].imports(nodes[i])
        