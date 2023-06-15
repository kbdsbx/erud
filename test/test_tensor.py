from erud.tensor.var import var
from erud.tensor.const import const
import numpy as np

# 测试变量
def test_variable () :
    v1 = var(5)

    assert v1.fprop() == 5
    assert v1.bprop(7) == [0]

    v1.data = 7

    assert v1.fprop() == 7

    np.random.seed(2)

    v2 = var(np.random.randn(3,5,4))

    np.random.seed(2)
    assert np.all(np.equal( v2.fprop(), np.random.randn(3,5,4) ))
    assert np.all(np.equal( v2.bprop(np.ones((1,2,3))), [np.zeros((3,5,4))]))
    assert np.all(np.equal( v2.bprop(), [np.zeros((3,5,4))]))

    np.random.seed(2)
    assert np.all(np.equal( v2.data, np.random.randn(3,5,4)))


# 测试常量
def test_constant () :
    c1 = const(5)

    assert c1.fprop() == 5
    assert c1.bprop(7) == [0]

    np.random.seed(2)

    c2 = const(np.random.randn(3,5,4))

    np.random.seed(2)
    assert np.all(np.equal( c2.fprop(), np.random.randn(3,5,4) ))
    assert np.all(np.equal( c2.bprop(np.ones((1,2,3))), [np.zeros((3,5,4))]))
    assert np.all(np.equal( c2.bprop(), [np.zeros((3,5,4))]))

    np.random.seed(2)
    assert np.all(np.equal( c2.data, np.random.randn(3,5,4)))
