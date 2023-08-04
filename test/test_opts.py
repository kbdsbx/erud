from erud.opts.add import add
from erud.opts.sub import sub
from erud.opts.mul import mul
from erud.opts.div import div
from erud.opts.matmul import matmul
from erud.opts.relu import relu
from erud.opts.sigmoid import sigmoid
from erud.opts.softmax import softmax
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np
import pytest as test

np.set_printoptions(precision=99, suppress=True)


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

# relu
def test_operator_relu() :
    opt = relu()

    assert opt.fprop(5) == 5
    assert np.all(opt.bprop(15) == [15])
    assert opt.fprop(-5) == 0
    assert np.all(opt.bprop(15) == [0])
    assert np.all(opt.fprop(np.array([[1, -4], [3, 0]])) == np.array([[1, 0], [3, 0]]))
    assert np.all(opt.bprop(np.array([[5, 3], [-6, 0]])) == np.array([[5, 0], [-6, 0]]))

# sigmoid
def test_operator_sigmoid() :
    opt = sigmoid()

    assert opt.fprop(0) == 0.5
    assert np.all(opt.bprop(4) == [1])
    
    assert opt.fprop(3) == 0.9525741268224334
    assert np.all(opt.bprop(4) == [0.180706638923648])

    assert np.all(opt.fprop(np.array([1,2,3])) == np.array([0.7310585786300049, 0.8807970779778823, 0.9525741268224334]))

    assert np.all(opt.bprop(np.array([1,1,1]))[0] == np.array([0.19661193324148185, 0.10499358540350662, 0.045176659730912]))


from erud.opts.cross_entropy import cross_entropy

# softmax
def test_softmax() :
    z = np.array([3, 1, -3])
    y = np.array([1, 0, 0])
    softmax_opt = softmax(0)
    a = softmax_opt.fprop(z)
    cross_entropy_opt = cross_entropy()
    j = cross_entropy_opt.fprop(a, y)

    assert np.all(a == np.array([0.8788782427321509, 0.11894323591065209, 0.002178521357197023]))

    assert np.all(j == np.array([0.1291089088298506, 0.1266332236921202, 0.002180897786878049]))

    [da, _] = cross_entropy_opt.bprop(np.ones(3))

    assert np.all(da == np.array([-1.137814035413279, 1.1350006500813723, 1.002183277674239 ]))

    [dz, _] = softmax_opt.bprop(da)

    assert np.all(dz == np.array([-0.2416897266247951,  0.23762678570983872, 0.004062940914956294]))

from erud.opts.cost import cost
from erud.nous import nous

# cost
def test_cost() :
    w = np.array([[1], [2]])
    b = 2
    X = np.array([[1,2], [3,4]])
    Y = np.array([[1,0]])
    sigmoid_opt = sigmoid()
    cross_entropy_opt = cross_entropy()
    cost_opt = cost()

    A = sigmoid_opt.fprop(np.dot(w.T, X) + b)
    j = cross_entropy_opt.fprop(A, Y)
    c = cost_opt.fprop(j)

    assert c == 6.000064773192205

    # ↑
    # 上下等价
    # ↓

    g = nous(
        """
        W:[[1, 2]] matmul X:[[1, 2], [3, 4]] add b:2 ->
        sigmoid ->
        cross_entropy Y:[[1,0]] ->
        cost ->
        j:$$
        """
    ).parse()

    # 前向传播

    g.fprop()
    c = g.getData('j')
    assert c == 6.000064773192205

    # 反向传播计算梯度

    def update_w_func (w, dw) :
        assert np.all(dw == np.array([[0.9999321585374046, 1.999802619786816 ]]))
    def update_b_func (b, db) :
        assert db == 0.4999352306247057

    # 给W和B设置更新参数，但此处不更新而是断言dw和db的值是否正确
    g.setUpdateFunc('W', update_w_func )
    g.setUpdateFunc('b', update_b_func )

    g.bprop()

from erud.opts.batchnorm import batchnorm

def test_batchnorm() :
    x = np.array([[1,2],[3,4]])
    b = batchnorm()
    z = b.fprop(x)

    assert np.sum(z) == 0.
    assert np.sum(np.power(z, 2)) == 3.999999968000002

    [dx] = b.bprop(np.array([[1, 1],[1, 1]]))

    assert np.all(dx == np.array([[0, 0], [0, 0]]))

from erud.opts.batchnorm2d import batchnorm2d

def test_batchnorm2d() :
    np.random.seed(1)
    x = np.random.randn(2, 3, 4, 5)
    b = batchnorm2d()
    z = b.fprop(x)
    assert np.sum(z) == 4.440892098500626e-15
    assert z.shape == (2, 3, 4, 5)
    assert np.sum(np.power(z, 2)) == 119.99999818952824

    [dx] = b.bprop(np.random.randn(2, 3, 4, 5))
    assert dx.shape == (2, 3, 4, 5)

from erud.opts_extend.max_index import max_index

def test_max_index() :
    m = np.array([[1,2], [3,4]])
    assert np.all(max_index(1).fprop(m) == np.array([1, 1]))

from erud.opts.conv2d import conv2d

def test_conv2d() :
    np.random.seed(1)
    X = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    conv = conv2d(1, 2)
    Z = conv.fprop(X, W)
    assert np.mean(Z) == -0.011100360859202631

    np.random.seed(1)
    [dx, dw] = conv.bprop(Z)
    assert np.mean(dx) == 3.6150899802828027
    assert np.mean(dw) == 9.078398646602041

from erud.opts.max_pool import max_pool

def test_pooling_max_2d() :
    np.random.seed(1)

    X = np.random.randn(2, 4, 4, 3)
    opt = max_pool(1, 4, 4)
    Z = opt.fprop(X)
    assert np.all(Z == np.array([[[[1.74481176421648, 1.6924546010277466, 2.100255136478842]]], [[[1.198917879901507, 1.5198168164221988, 2.1855754065331614]]]]))

    np.random.seed(1)
    X = np.random.randn(5, 5, 3, 2)
    opt = max_pool(1, 2, 2)
    Z = opt.fprop(X)
    dZ = np.random.randn(5, 4, 2, 2)
    [dX] = opt.bprop(dZ)
    assert np.mean(dZ) == 0.14571390272918056
    assert np.mean(dX) == 0.07771408145556294
    assert np.all(dX[1,1] == np.array([[ 0., 0.,],[ 5.058443935342465,-1.6828270215556507],[ 0.,0.,]]))

from erud.opts.max_pool_v3 import max_pool_v3

def test_pooling_max_2d_for_cpp() :

    X = np.array([(i+1) for i in range(144)], dtype = np.float32)
    X = X.reshape((2, 4, 6, 3))
    opt = max_pool(2, 2, 2)
    Z = opt.fprop(X)
    # print('Z:' +str(Z))

    dZ = np.array([((i+1) * 0.01) for i in range(36)])
    dZ = dZ.reshape((2, 2, 3, 3))

    [dW] = opt.bprop(dZ)
    # print('dW:' +str(dW))

    opt2 = max_pool_v3(2, 2, 2)
    Zn = opt2.fprop(X)
    # print('Zn:' +str(Zn))

    [dWn] = opt2.bprop(dZ)
    # print('dWn:' +str(dWn))




from erud.opts.conv2d_v2 import conv2d_v2

def test_conv2d_v2() :
    np.random.seed(1)
    X = np.random.randn(10, 4, 4, 3)
    W = np.random.randn(2, 2, 3, 8)
    conv = conv2d_v2(1, 2)
    Z = conv.fprop(X, W)
    assert np.mean(Z) == -0.011100360859202626

    np.random.seed(1)
    [dx, dw] = conv.bprop(Z)
    assert np.mean(dx) == 3.615089980282802
    assert np.mean(dw) == 9.078398646602034

from erud.opts.conv2d_v3 import conv2d_v3

def test_conv2d_v2_for_cpp() :
    X = np.array([(i+1) for i in range(144)], dtype = np.float32)
    W = np.array([-(i+1) for i in range(24)], dtype = np.float32)
    # print('X : ' + str(X))
    # print('W : ' + str(W))

    X = X.reshape((2, 4, 6, 3))
    W = W.reshape((2, 2, 3, 2))

    conv = conv2d_v2(1, 2)
    Z = conv.fprop(X, W)
    # print('Z : ' + str(Z))

    dZ = np.array([((i+1) * 0.01) for i in range(252)])

    dZ = dZ.reshape((2, 7, 9, 2))

    [dX, dW] = conv.bprop(dZ)
    # print('dX :' + str(dX))
    # print('dW :' + str(dW))

    conv_v3 = conv2d_v3(1, 2)
    Zn = conv_v3.fprop(X, W)
    # print('Z :' + str(Zn))

    [dXn, dWn] = conv_v3.bprop(dZ)
    # print('dX :' + str(dXn))
    # print('dW :' + str(dWn))
    






