from erud.cg.graph import ComputationGraph as graph
from erud.cg.node import ComputationNode as node
from erud.cg.errors import *

import pytest

def setup_function(self) :
    print(self.__name__ + ' printed out:')

# 插入节点
def test_graph_insert_node() :
    g = graph()
    n1 = node()
    g.insertNode(n1)

    assert g.hasNode == 1

    n2 = node()
    g.insertNode(n2)

    assert g.hasNode == 2

    n3 = node()
    g.insertNode(n3)

    assert g.hasNode == 3

    with pytest.raises(NodeRepeatError) :
        # 重复插入节点
        g.insertNode(n1)

    with pytest.raises(NodeRepeatError) :
        # 重复插入节点
        g.insertNode(n3)

# 添加边
def test_graph_add_edge() :
    g = graph()
    assert g.hasNode == 0
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

    assert g.hasNode == 4

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

    print(g)

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
    
    print(g)


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

    print(g)


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

    print(g)

    with pytest.raises(NodeNotFindError) :
        # 删除不存在的节点
        g.removeNode(n7)


