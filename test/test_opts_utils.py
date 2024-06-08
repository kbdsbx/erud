from erud.opts._utils import broadcast_axis
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
import pytest as test

def test_broadcast_axis () :
    z = np.random.randn(5,8,2)

    # 广播第一个维度
    x = np.random.randn(8,2)
    z + x
    assert broadcast_axis(z, x) == [0]

    # 广播前两个维度
    x = np.random.randn(1, 2)
    z + x
    assert broadcast_axis(z, x) == [0, 1]

    # 维度不匹配则异常
    x = np.random.randn(2, 2)
    with test.raises(ValueError) :
        broadcast_axis(z, x)
    
    with test.raises(ValueError) :
        z + x

    # 相同纬度
    x = np.random.randn(5,8,2)
    z + x
    assert broadcast_axis(z, x) == []

    # 广播中间维度
    x = np.random.randn(5,1,2)
    z + x
    assert broadcast_axis(z, x) == [1]

    # 结果数比操作数维度更高则异常
    x = np.random.randn(5, 5, 8, 2)
    with test.raises(ValueError) :
        broadcast_axis(z, x)
    
    # 广播后两个维度
    x = np.random.randn(5,1,1)
    z + x
    assert broadcast_axis(z, x) == [1,2]

    # 广播全部维度
    x = np.random.randn(1,1,1)
    z + x
    assert broadcast_axis(z, x) == [0,1,2]
