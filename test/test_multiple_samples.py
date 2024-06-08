"""
测试各个组件中各个样本的处理是否独立，相同样本在单次处理和批量处理中，各个步骤得出的结果是否相同
"""

from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
import pytest as test

np.set_printoptions(precision=99, suppress=True)

from erud.opts.conv2d_v3_same import conv2d_v3_same

def get_test_samples() :

    np.random.seed(1)
    X = np.random.randn(3,3,3,3)
    X1 = X[0:1,:,:,:]
    X2 = X[0:2,:,:,:]
    X3 = X[0:3,:,:,:]

    return X1, X2, X3

def test_conv2d_v3_same () :
    X1, X2, X3 = get_test_samples()
    W = np.random.randn(2,2,3,4)

    opt = conv2d_v3_same()
    Z1 = opt.fprop(X1, W)
    Z2 = opt.fprop(X2, W)
    Z3 = opt.fprop(X3, W)

    assert np.all(Z1[0] == Z2[0])
    assert np.all(Z1[0] == Z3[0])

from erud.opts.batchnorm2d import batchnorm2d

def test_batchnorm2d() :
    X1, X2, X3 = get_test_samples()
    opt = batchnorm2d()

    Z1 = opt.fprop(X1)
    Z2 = opt.fprop(X2)
    Z3 = opt.fprop(X3)
    # print(Z1)
    # print(Z2)
    # print(Z3)

    # assert np.all(Z1[0] == Z2[0])
    # assert np.all(Z1[0] == Z3[0])

from erud.opts.relu import relu

def test_relu () :
    X1, X2, X3 = get_test_samples()
    opt = relu()

    Z1 = opt.fprop(X1)
    Z2 = opt.fprop(X2)
    Z3 = opt.fprop(X3)

    assert np.all(Z1[0] == Z2[0])
    assert np.all(Z1[0] == Z3[0])

from erud.opts.max_pool_v3 import max_pool_v3

def test_max_pool_v3 () :
    X1, X2, X3 = get_test_samples()
    opt = max_pool_v3()

    Z1 = opt.fprop(X1)
    Z2 = opt.fprop(X2)
    Z3 = opt.fprop(X3)

    assert np.all(Z1[0] == Z2[0])
    assert np.all(Z1[0] == Z3[0])

from erud.opts.flatten import flatten

def test_flatten () :
    X1, X2, X3 = get_test_samples()
    opt = flatten()

    Z1 = opt.fprop(X1)
    Z2 = opt.fprop(X2)
    Z3 = opt.fprop(X3)

    assert np.all(Z1[0] == Z2[0])
    assert np.all(Z1[0] == Z3[0])

from erud.opts.matmul import matmul

def test_matmul () :
    X1, X2, X3 = get_test_samples()
    W = np.random.randn(27, 10)
    opt1 = flatten()
    opt = matmul()

    X1 = opt1.fprop(X1)
    X2 = opt1.fprop(X2)
    X3 = opt1.fprop(X3)

    Z1 = opt.fprop(X1, W)
    Z2 = opt.fprop(X2, W)
    Z3 = opt.fprop(X3, W)

    assert np.all((Z1[0] - Z2[0]) < 1e8)
    assert np.all((Z1[0] - Z3[0]) < 1e8)

from erud.opts.add import add

def test_add () :
    X1, X2, X3 = get_test_samples()
    W = np.random.randn(27)
    opt1 = flatten()
    opt = add()

    X1 = opt1.fprop(X1)
    X2 = opt1.fprop(X2)
    X3 = opt1.fprop(X3)

    Z1 = opt.fprop(X1, W)
    Z2 = opt.fprop(X2, W)
    Z3 = opt.fprop(X3, W)

    assert np.all(Z1[0] == Z2[0])
    assert np.all(Z1[0] == Z3[0])

from erud.opts.softmax_cross_entropy import softmax_cross_entropy

def test_softmax_cross_entropy() :
    X1, X2, X3 = get_test_samples()
    Y = np.random.randn(3, 27)
    opt1 = flatten()
    opt = softmax_cross_entropy(1)

    X1 = opt1.fprop(X1)
    X2 = opt1.fprop(X2)
    X3 = opt1.fprop(X3)

    Z1 = opt.fprop(X1, Y[0:1])
    Z2 = opt.fprop(X2, Y[0:2])
    Z3 = opt.fprop(X3, Y[0:3])

    assert np.all(Z1[0] == Z2[0])
    assert np.all(Z1[0] == Z3[0])




