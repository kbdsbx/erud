from erud.graph import ComputationGraph as graph
from erud.node import ComputationNode as node

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

# 添加边
def test_graph_add_edge() :
    g = graph()
    assert g.hasNode == 0
    n1 = node(1)
    n2 = node(2)
    n3 = node(3)
    n4 = node(4)

    assert n1.info == 1
    assert n2.info == 2
    assert n3.info == 3
    assert n4.info == 4

    g.insertNode(n1)
    g.insertNode(n2)
    g.insertNode(n3)
    g.insertNode(n4)

    assert g.hasNode == 4

    g.addEdge(n1, n3)

    assert n1.fFristEdge.fNode.info == n3.info
    assert n1.fFristEdge.fNode is n3
    assert n1.fFristEdge.bNode.info == n1.info
    assert n1.fFristEdge.bNode is n1
    assert n3.bFristEdge.bNode.info == n1.info
    assert n3.bFristEdge.bNode is n1
    assert n3.bFristEdge.fNode.info == n3.info
    assert n3.bFristEdge.fNode is n3

    g.addEdge(n1, n4)

    assert n1.fFristEdge.fNextEdge.fNode is n4
    assert n1.fFristEdge.fNextEdge.bNode is n1
    assert n4.bFristEdge.bNode is n1
    assert n4.bFristEdge.fNode is n4

    print(g)

# 添加边时如果相对应的节点不存在则抛出异常
def test_graph_raise_unfind_node() :
    g = graph()
    n1 = node(1)
    n2 = node(2)
    n3 = node(3)

    g.insertNode(n1)
    g.insertNode(n2)

    try :
        g.addEdge(n1, n3)
    except Exception as e :
        assert isinstance(e, IndexError)