from erud.cg.graph import ComputationGraph as graph
from erud.cg.node import ComputationNode as node
from erud.errors import *
import numpy as np
import random

import pytest

# def setup_function(self) :
    # print(self.__name__ + ' printed out:')

# 插入节点
def test_graph_insert_node() :
    g = graph()
    n1 = node()
    g.insertNode(n1)

    assert g.nodeCount== 1

    n2 = node()
    g.insertNode(n2)

    assert g.nodeCount == 2

    n3 = node()
    g.insertNode(n3)

    assert g.nodeCount == 3

    with pytest.raises(NodeRepeatError) :
        # 重复插入节点
        g.insertNode(n1)

    with pytest.raises(NodeRepeatError) :
        # 重复插入节点
        g.insertNode(n3)

# 添加边
def test_graph_add_edge() :
    g = graph()
    assert g.nodeCount == 0
    n1 = node(1)
    n2 = node(2)
    n3 = node(3)
    n4 = node(4)

    assert n1.data == 1
    assert n2.data == 2
    assert n3.data == 3
    assert n4.data == 4

    g.insertNode(n1)
    g.insertNode(n2)
    g.insertNode(n3)
    g.insertNode(n4)

    assert g.nodeCount == 4

    g.addEdge(n1, n3)

    assert n1.fFirstEdge.fNode.data == n3.data
    assert n1.fFirstEdge.fNode is n3
    assert n1.fFirstEdge.bNode.data == n1.data
    assert n1.fFirstEdge.bNode is n1
    assert n3.bFirstEdge.bNode.data == n1.data
    assert n3.bFirstEdge.bNode is n1
    assert n3.bFirstEdge.fNode.data == n3.data
    assert n3.bFirstEdge.fNode is n3

    g.addEdge(n1, n4)

    assert n1.fFirstEdge.fNextEdge.fNode is n4
    assert n1.fFirstEdge.fNextEdge.bNode is n1
    assert n4.bFirstEdge.bNode is n1
    assert n4.bFirstEdge.fNode is n4

    # print(g)

# 添加边时如果相对应的节点不存在则抛出异常
def test_graph_raise_unfind_node() :
    g = graph()
    n1 = node(1)
    n2 = node(2)
    n3 = node(3)
    n4 = node(4)
    n5 = node(5)
    n6 = node(6)

    g.insertNode(n1)
    g.insertNode(n2)

    with pytest.raises(NodeNotFindError) :
        # n3 没有添加到图
        g.addEdge(n1, n3)

    g.insertNode(n3)
    g.insertNode(n4)
    g.insertNode(n5)
    g.insertNode(n6)

    g.addEdge(n1, n2)
    g.addEdge(n1, n3)
    g.addEdge(n1, n4)
    g.addEdge(n2, n5)
    g.addEdge(n2, n6)

    with pytest.raises(EdgeRepeatError) :
        # 重复添加n1->n2的边
        g.addEdge(n1, n2)

    with pytest.raises(EdgeRepeatError) :
        # 重复添加n1->n4的边
        g.addEdge(n1, n4)

    with pytest.raises(EdgeRepeatError) :
        # 重复添加n2->n6的边
        g.addEdge(n2, n6)
    
    # print(g)


# 删除边
def test_graph_remove_edge() :
    g = graph()
    n1 = node(1)
    n2 = node(2)
    n3 = node(3)
    n4 = node(4)
    n5 = node(5)
    n6 = node(6)

    g.insertNode(n1)
    g.insertNode(n2)
    g.insertNode(n3)
    g.insertNode(n4)
    g.insertNode(n5)
    g.insertNode(n6)

    g.addEdge(n1, n2)
    g.addEdge(n1, n3)
    g.addEdge(n1, n4)
    g.addEdge(n1, n5)
    g.addEdge(n2, n5)
    g.addEdge(n2, n6)

    g.removeEdge(n1, n2)

    assert n1.fFirstEdge.fNode is n3
    assert n1.fFirstEdge.bNode is n1
    assert n3.bFirstEdge.bNode is n1
    assert n3.bFirstEdge.fNode is n3

    g.removeEdge(n1, n4)

    assert n1.fFirstEdge.fNextEdge.fNode is n5
    assert n1.fFirstEdge.fNextEdge.bNode is n1
    assert n5.bFirstEdge.bNode is n1
    assert n5.bFirstEdge.fNode is n5

    # print(g)


# 删除不存在的边抛出异常
def test_graph_raise_unfind_edge () :
    g = graph()
    n1 = node(1)
    n2 = node(2)
    n3 = node(3)
    n4 = node(4)

    g.insertNode(n1)
    g.insertNode(n2)
    g.insertNode(n3)
    g.insertNode(n4)

    g.addEdge(n1, n3)
    g.addEdge(n1, n4)

    with pytest.raises(EdgeNotFindError) :
        # 删除不存在的边
        g.removeEdge(n2, n4)

    with pytest.raises(EdgeNotFindError) :
        # 删除不存在的边
        g.removeEdge(n1, n2)


    g.removeEdge(n1, n3)
    with pytest.raises(EdgeNotFindError) :
        # 删除已删除的边
        g.removeEdge(n1, n3)

# 删除节点
def test_graph_remove_node () :
    g = graph()
    n1 = node(1)
    n2 = node(2)
    n3 = node(3)
    n4 = node(4)
    n5 = node(5)
    n6 = node(6)
    n7 = node(7)

    g.insertNode(n1)
    g.insertNode(n2)
    g.insertNode(n3)
    g.insertNode(n4)
    g.insertNode(n5)
    g.insertNode(n6)

    g.addEdge(n1, n2)
    g.addEdge(n1, n3)
    g.addEdge(n1, n5)
    g.addEdge(n1, n6)
    g.addEdge(n2, n3)
    g.addEdge(n2, n5)
    g.addEdge(n2, n6)
    g.addEdge(n4, n1)
    g.addEdge(n5, n6)

    g.removeNode(n1)

    assert n2.bFirstEdge is None
    assert n3.bFirstEdge.bNode is n2
    assert n5.bFirstEdge.bNode is n2
    assert n4.fFirstEdge is None

    g.removeNode(n5)

    assert n2.fFirstEdge.fNextEdge.fNode is n6
    assert n6.bFirstEdge.bNextEdge is None

    # print(g)

    with pytest.raises(NodeNotFindError) :
        # 删除不存在的节点
        g.removeNode(n7)


# 获取入度、出度为零的点的内部方法
def test_getStartingNodebalabalabala() :
    g = graph()

    n1 = node(1)
    n2 = node(2)
    n3 = node(3)
    n4 = node(4)
    n5 = node(5)
    n6 = node(6)
    n7 = node(7)

    g.insertNode(n1)
    g.insertNode(n2)
    g.insertNode(n3)
    g.insertNode(n4)
    g.insertNode(n5)
    g.insertNode(n6)
    g.insertNode(n7)

    g.addEdge(n1, n3)
    g.addEdge(n2, n3)
    g.addEdge(n3, n4)
    g.addEdge(n5, n4)
    g.addEdge(n4, n6)

    # 获取前向传播入度为0的点
    init_node = g._getStartingNodeInForwardPropagation(g.nodes)

    assert len(init_node) == 4
    assert np.all(init_node == [n1, n2, n5, n7])

    # 获取前向传播出度为0的点
    end_node = g._getEndedNodeInForwardPropagation(g.nodes)

    assert len(end_node) == 2
    assert np.all(end_node == [n6, n7])

    # 如果n1 -> n3和n2 -> n3的边运载有值，说明n1和n2已经经过了运算
    # 此时n3虽然有入边，依然可以视为入度为0进行下一层运算
    n1.fFirstEdge.carry = 5
    n2.fFirstEdge.carry = 10

    init_node = g._getStartingNodeInForwardPropagation(g.nodes)
    assert len(init_node) == 5
    assert np.all(init_node == [n1, n2, n3, n5, n7])


from erud.opts.add import add
from erud.opts.sub import sub
from erud.opts.mul import mul
from erud.opts.div import div
from erud.tensor.var import var
from erud.tensor.rest import rest

# 前向传播，即计算图计算
def test_fprop() :

    # ret = (5 + 10) * (6 - 19) / 3
    g = graph()

    n1 = node(var(5))
    n2 = node(var(10))
    n3 = node(var(6))
    n4 = node(var(19))
    n5 = node(var(3))

    n6 = node(add())
    n7 = node(mul())
    n8 = node(sub())
    n9 = node(div())
    n10 = node(rest())

    g.insertNode(n1)
    g.insertNode(n2)
    g.insertNode(n3)
    g.insertNode(n4)
    g.insertNode(n5)
    g.insertNode(n6)
    g.insertNode(n7)
    g.insertNode(n8)
    g.insertNode(n9)
    g.insertNode(n10)

    # 5 + 10
    g.addEdge(n1, n6)
    g.addEdge(n2, n6)

    # 6 - 19
    g.addEdge(n3, n8)
    g.addEdge(n4, n8)

    # (5 + 10) * (6 - 19)
    g.addEdge(n6, n7)
    g.addEdge(n8, n7)

    # (5 + 10) * (6 - 19) / 3
    g.addEdge(n7, n9)
    g.addEdge(n5, n9)

    # ret = (5 + 10) * (6 - 19) / 3
    g.addEdge(n9, n10)

    [res] = g.fprop()

    assert res.data == -65.0

# 反向传播，即计算图梯度下降
def test_bprop () :

    # 此用例为深度学习的超级简化版
    # 其中神经网络为y = (5 + w) * (6 - 19) / 3
    # 标签值为y_hat = -65
    # 代价函数为J = (y / y_hat) - 1
    # 前向传播计算图为ret = [(5 + w) * (6 - 19) / 3] / (-65) - 1，其中w为需要学习的参数
    # 学习步骤如下：
    # 1. 随机初始化w
    # 2. 代入计算图前向传播计算ret（在随机w的情况下J大概率不为0）
    # 3. 计算图反向传播计算d(ret)/dw
    # 4. 使用学习率r更新w，即w := w - r * d(ret)/dw
    # 5. 重复2，直到ret等于零或接近零，此时的w即为深度学习的最终结果

    # 初始化计算图
    # ret = [(5 + w) * (6 - 19) / 3] / (-65) - 1
    g = graph()
    # 学习率
    r = 0.01

    # 固定随机数种子，保证测试用例正确运行
    random.seed(1)

    c1 = node(var(5))
    # 参数在10-15中随机，并利用梯度下降更新
    # 因为代价函数并非二次函数，所以梯度下降只能沿着一个方向，故随机值需要大于真实值
    def update_rate (z, dz) :
        return z - r * dz
    w = node(var(random.randint(10, 15), update_rate))
    c2 = node(var(6))
    c3 = node(var(19))
    c4 = node(var(3))
    c5 = node(var(-65))
    c6 = node(var(1))

    o1 = node(add())
    o2 = node(sub())
    o3 = node(mul())
    o4 = node(div())
    o5 = node(div())
    o6 = node(sub())

    s = node(rest())

    g.insertNode(c1)
    g.insertNode(c2)
    g.insertNode(c3)
    g.insertNode(c4)
    g.insertNode(c5)
    g.insertNode(c6)

    g.insertNode(o1)
    g.insertNode(o2)
    g.insertNode(o3)
    g.insertNode(o4)
    g.insertNode(o5)
    g.insertNode(o6)

    g.insertNode(w)
    g.insertNode(s)

    # 5 + 10
    g.addEdge(c1, o1)
    g.addEdge(w, o1)

    # 6 - 19
    g.addEdge(c2, o2)
    g.addEdge(c3, o2)

    # (5 + 10) * (6 - 19)
    g.addEdge(o1, o3)
    g.addEdge(o2, o3)

    # (5 + 10) * (6 - 19) / 3
    g.addEdge(o3, o4)
    g.addEdge(c4, o4)

    # [(5 + 10) * (6 - 19) / 3 ] / (-65)
    g.addEdge(o4, o5)
    g.addEdge(c5, o5)

    # [(5 + 10) * (6 - 19) / 3 ] / (-65) - 1
    g.addEdge(o5, o6)
    g.addEdge(c6, o6)

    # ret = [(5 + 10) * (6 - 19) / 3 ] / (-65) - 1
    g.addEdge(o6, s)

    [res] = g.fprop()

    assert res.data == 0.06666666666666665

    print('There is start to learn parameter of "w"')

    # 多次梯度下降
    for i in range(1500) :
        # 前向传播
        [res]  = g.fprop()
        if (i + 1) % 100 == 0 :
            print('Cost is becoming %s in %s iteration' % (res.data, i+1))
        if (abs(res.data) < 0.0002) :
            print('Cost is becoming %s in %s iteration' % (res.data, i+4))
            break
        # 反向传播，其中var变量会在传播过程中内部调用update_rate方法实现参数本身的更新
        g.bprop()
    
    print('The end of "w" is %s.' % (w.data))
    
