from erud.opts.add import add
from erud.opts.sub import sub
from erud.opts.mul import mul
from erud.opts.div import div
from erud.opts.matmul import matmul
import numpy as np
import pytest as test

# 加法
def test_operator_add () :
    opt = add()

    res = opt.fprop(5, 7)
    assert res == 12

    res = opt.fprop(5.5, 7.7)
    assert res == 13.2

    np.random.seed(2)
    r = np.random.randn(5,3,2)
    res = opt.fprop(r, r)
    assert np.all(np.equal(res, np.add(r, r)))

    # 不同维度张量相加会异常
    with test.raises(ValueError) :
        res = opt.fprop(np.random.randn(5,3,2), np.random.randn(5,1,3))
    
    opt.fprop(5, 7)
    res = opt.bprop(5)
    assert np.all(np.equal(res, [5, 5]))

    # 左操作数为标量，右操作数为矩阵shape(2, 2)，反向传播对两侧操作数产生不同的维度
    x = 5
    y = np.array([[1,2],[3,4]])
    z = opt.fprop(x, y)
    assert np.all(z == np.array([[6, 7], [8, 9]]))
    dz = np.array([ [11, 10], [9, 8] ])
    res = opt.bprop(dz)
    assert res[0] == 38
    assert res[1].shape == (2, 2)
    assert np.all(res[1] == np.array([[11, 10], [9, 8]]))

    # 左操作数为张量shape(3, 2, 4)，右操作数为矩阵(1, 2, 4)
    np.random.seed(2)
    x = np.array([[[8, 8, 6, 2],
        [8, 7, 2, 1]],

       [[5, 4, 4, 5],
        [7, 3, 6, 4]],

       [[3, 7, 6, 1],
        [3, 5, 8, 4]]])
    y = np.array([[[6, 3, 9, 2],
        [0, 4, 2, 4]]])
    dz = np.array([[[1, 7, 8, 2],
        [9, 8, 7, 1]],

       [[6, 8, 5, 9],
        [9, 9, 3, 0]],

       [[0, 2, 8, 8],
        [2, 9, 6, 5]]])

    z = opt.fprop(x, y)
    [dx, dy] = opt.bprop(dz)
    assert z.shape == (3, 2, 4)
    assert dx.shape == x.shape
    assert dy.shape == y.shape
    assert np.all(z == np.array([[[14, 11, 15,  4],
        [ 8, 11,  4,  5]],

       [[11,  7, 13,  7],
        [ 7,  7,  8,  8]],

       [[ 9, 10, 15,  3],
        [ 3,  9, 10,  8]]]))
    assert np.all(dx == np.array([[[1, 7, 8, 2],
        [9, 8, 7, 1]],

       [[6, 8, 5, 9],
        [9, 9, 3, 0]],

       [[0, 2, 8, 8],
        [2, 9, 6, 5]]]))
    assert np.all(dy == np.array([[ 7, 17, 21, 19],
       [20, 26, 16,  6]]))
   

    # 左操作数为张量shape(2,1,4,5)，右操作数为张量shape(2,3,1,5)
    x = np.array([[[[8, 8, 6, 2, 8],
         [7, 2, 1, 5, 4],
         [4, 5, 7, 3, 6],
         [4, 3, 7, 6, 1]]],

       [[[3, 5, 8, 4, 6],
         [3, 9, 2, 0, 4],
         [2, 4, 1, 7, 8],
         [2, 9, 8, 7, 1]]]])
    y = np.array([[[[6, 8, 5, 9, 9]],
        [[9, 3, 0, 0, 2]],
        [[8, 8, 2, 9, 6]]],

       [[[5, 6, 6, 6, 3]],
        [[8, 2, 1, 4, 8]],
        [[1, 6, 9, 5, 1]]]])
    
    dz = np.array([[[[2, 4, 7, 6, 4],
         [5, 8, 3, 0, 0],
         [5, 7, 5, 0, 8],
         [6, 5, 1, 7, 4]],

        [[3, 6, 1, 4, 0],
         [8, 5, 4, 2, 9],
         [7, 1, 9, 2, 1],
         [0, 7, 1, 8, 9]],

        [[0, 7, 0, 5, 2],
         [5, 1, 3, 3, 1],
         [8, 6, 8, 1, 5],
         [7, 0, 9, 1, 5]]],


       [[[9, 2, 0, 0, 4],
         [6, 3, 1, 8, 5],
         [9, 5, 4, 2, 7],
         [8, 7, 3, 4, 8]],

        [[6, 3, 8, 8, 5],
         [1, 3, 3, 3, 2],
         [6, 5, 5, 2, 5],
         [6, 1, 5, 0, 5]],

        [[4, 8, 3, 7, 5],
         [9, 4, 5, 2, 6],
         [0, 5, 7, 1, 6],
         [7, 0, 1, 2, 4]]]])
    
    z = opt.fprop(x, y)
    [dx, dy] = opt.bprop(dz)

    assert dx.shape == x.shape
    assert dy.shape == y.shape
    assert z.shape == (2, 3, 4, 5)
    assert np.all(z == np.array([[[[14, 16, 11, 11, 17],
         [13, 10,  6, 14, 13],
         [10, 13, 12, 12, 15],
         [10, 11, 12, 15, 10]],

        [[17, 11,  6,  2, 10],
         [16,  5,  1,  5,  6],
         [13,  8,  7,  3,  8],
         [13,  6,  7,  6,  3]],

        [[16, 16,  8, 11, 14],
         [15, 10,  3, 14, 10],
         [12, 13,  9, 12, 12],
         [12, 11,  9, 15,  7]]],

       [[[ 8, 11, 14, 10,  9],
         [ 8, 15,  8,  6,  7],
         [ 7, 10,  7, 13, 11],
         [ 7, 15, 14, 13,  4]],

        [[11,  7,  9,  8, 14],
         [11, 11,  3,  4, 12],
         [10,  6,  2, 11, 16],
         [10, 11,  9, 11,  9]],

        [[ 4, 11, 17,  9,  7],
         [ 4, 15, 11,  5,  5],
         [ 3, 10, 10, 12,  9],
         [ 3, 15, 17, 12,  2]]]]))

    assert np.all(dx == np.array([[[[ 5, 17,  8, 15,  6],
         [18, 14, 10,  5, 10],
         [20, 14, 22,  3, 14],
         [13, 12, 11, 16, 18]]],

       [[[19, 13, 11, 15, 14],
         [16, 10,  9, 13, 13],
         [15, 15, 16,  5, 18],
         [21,  8,  9,  6, 17]]]]))
    assert np.all(dy == np.array([[[[18, 24, 16, 13, 16]],
        [[18, 19, 15, 16, 19]],
        [[20, 14, 20, 10, 13]]],

       [[[32, 17,  8, 14, 24]],
        [[19, 12, 21, 13, 17]],
        [[20, 17, 16, 12, 21]]]]))

# 乘法
def test_operator_mul () :
    opt = mul()

    # 数乘数
    x = 3
    y = 6
    z = opt.fprop(x, y)

    assert z == 18

    # 数乘矩阵
    x = np.array([[2,1], [3,4]])
    y = 5

    z = opt.fprop(x, y)

    assert z.shape == (2, 2)
    assert np.all(z == np.array([[10, 5], [15, 20]]))

    # 张量shape(1,2,2)乘张量shape(3,1,2)
    x = np.array([[[2, 2],
        [2, 8]]])
    y = np.array([[[7, 7]],
       [[2, 9]],
       [[7, 9]]])
    
    z = opt.fprop(x, y)

    assert z.shape == (3,2,2)
    assert np.all(z == np.array([[[14, 14],
        [14, 56]],
       [[ 4, 18],
        [ 4, 72]],
       [[14, 18],
        [14, 72]]]))
    
    dz = np.array([[[8, 1],
        [1, 7]],
       [[5, 1],
        [2, 8]],
       [[2, 9],
        [0, 1]]])
    
    # 反向传播
    dx, dy = opt.bprop(dz)

    assert dx.shape == x.shape
    assert dy.shape == y.shape
    assert np.all(dx == np.array([[[ 80,  97],
        [ 11, 130]]]))
    assert np.all(dy == np.array([[[18, 58]],
       [[14, 66]],
       [[ 4, 26]]]))

# 矩阵乘法
def test_operator_matmul () :
    m = matmul()

    x = np.array([[1, 4], [2, 5], [3, 6]])
    assert x.shape == (3, 2)

    y = np.array([[7, 8, 9], [10, 11, 12]])
    assert y.shape == (2, 3)

    z = m.fprop(x, y)
    assert z.shape == (3, 3)
    assert np.all(z == np.array([[47, 52, 57], [64, 71, 78], [81, 90, 99]]))

    dz = np.array([[5, 4, 3], [2, 1, 0], [-1, -2, -3]])

    # 矩阵乘法的梯度
    dx, dy = m.bprop(dz)
    assert np.all(dx == np.array([[94, 130], [22, 31], [-50, -68]]))
    assert np.all(dy == np.array([[6, 0, -6], [24, 9, -6]]))
