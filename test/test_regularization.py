import scipy.io
from erud.nous import nous
import numpy as np
   
np.set_printoptions(precision=99, suppress=True)

def load_2D_dataset():
    path = __file__[:__file__.rfind('\\')]
    data = scipy.io.loadmat(path + '/datasets/data.mat')
    train_X = data['X'].T
    train_Y = data['y'].T
    test_X = data['Xval'].T
    test_Y = data['yval'].T

    return train_X, train_Y, test_X, test_Y

def no_test_none_regularization () :
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    assert train_X.shape == (2, 211)
    assert train_Y.shape == (1, 211)
    assert test_X.shape == (2, 200)
    assert test_Y.shape == (1, 200)

    rate = 0.3
    num_iterations = 40000 

    np.random.seed(1)
    g = nous(
        '''
        X:(211, 2) ->

            matmul W1:he((2, 20), 4) add b1:(20) -> relu ->
            matmul W2:he((20, 3), 40) add b2:(3) -> relu ->
            matmul W3:he((3, 1), 6) add b3:(1) -> sigmoid ->
        
        cross_entropy Y:(211, 1) -> cost -> J:$$
        '''
    ).parse()

    g.setData('X', train_X.T)
    g.setData('Y', train_Y.T)

    def update_func (z, dz) :
        return z - rate * dz

    g.setUpdateFunc('W1', update_func)
    g.setUpdateFunc('W2', update_func)
    g.setUpdateFunc('W3', update_func)
    g.setUpdateFunc('b1', update_func)
    g.setUpdateFunc('b2', update_func)
    g.setUpdateFunc('b3', update_func)

    for i in range(num_iterations) :
        g.fprop()
        g.bprop()

        if i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, g.getData('J')))
    print("Cost after iteration {}: {}".format(num_iterations, g.getData('J')))
    

    # 测试

    gtest = nous(
        '''
        X ->

            matmul W1 add b1 -> relu ->
            matmul W2 add b2 -> relu ->
            matmul W3 add b3 -> sigmoid ->
        
        threshold(0.5) -> accuracy Y -> J:$$
        '''
    ).parse()

    # 迁移学习好的参数
    gtest.setData('W1', g.getData('W1'))
    gtest.setData('W2', g.getData('W2'))
    gtest.setData('W3', g.getData('W3'))
    gtest.setData('b1', g.getData('b1'))
    gtest.setData('b2', g.getData('b2'))
    gtest.setData('b3', g.getData('b3'))


    # 计算训练集精度
    gtest.setData('X', train_X.T)
    gtest.setData('Y', train_Y.T)

    gtest.fprop()

    print('train accuracy: %s' %(gtest.getData('J')))

    # 计算测试集精度
    gtest.setData('X', test_X.T)
    gtest.setData('Y', test_Y.T)

    gtest.fprop()

    print('test accuracy: %s' %(gtest.getData('J')))



def no_test_l2_regularization () :
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    assert train_X.shape == (2, 211)
    assert train_Y.shape == (1, 211)
    assert test_X.shape == (2, 200)
    assert test_Y.shape == (1, 200)

    rate = 0.3
    num_iterations = 30000 

    np.random.seed(1)
    g = nous(
        '''
        X:(211, 2) ->

            matmul W1:he((2, 20), 4) add b1:(20) -> relu ->
            matmul W2:he((20, 3), 40) add b2:(3) -> relu ->
            matmul W3:he((3, 1), 6) add b3:(1) -> sigmoid ->
        
        cross_entropy Y:(211, 1) -> cost as entropy_cost

        # L2 正则
        # 1/m * lambda / 2 * sum_all(w**2), w in W1, W2, W3

        (1.0 div 211) mul (lambd:0.7 div 2) mul ((W1 mul W1 -> sum) add (W2 mul W2 -> sum) add (W3 mul W3 -> sum)) as l2_reg_cost

        entropy_cost add l2_reg_cost -> J:$$
        '''
    ).parse()

    g.setData('X', train_X.T)
    g.setData('Y', train_Y.T)

    def update_func (z, dz) :
        return z - rate * dz

    g.setUpdateFunc('W1', update_func)
    g.setUpdateFunc('W2', update_func)
    g.setUpdateFunc('W3', update_func)
    g.setUpdateFunc('b1', update_func)
    g.setUpdateFunc('b2', update_func)
    g.setUpdateFunc('b3', update_func)

    for i in range(num_iterations) :
        g.fprop()
        g.bprop()

        if i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, g.getData('J')))
    print("Cost after iteration {}: {}".format(num_iterations, g.getData('J')))
    

    # 测试

    gtest = nous(
        '''
        X ->

            matmul W1 add b1 -> relu ->
            matmul W2 add b2 -> relu ->
            matmul W3 add b3 -> sigmoid ->
        
        threshold(0.5) -> accuracy Y -> J:$$
        '''
    ).parse()

    # 迁移学习好的参数
    gtest.setData('W1', g.getData('W1'))
    gtest.setData('W2', g.getData('W2'))
    gtest.setData('W3', g.getData('W3'))
    gtest.setData('b1', g.getData('b1'))
    gtest.setData('b2', g.getData('b2'))
    gtest.setData('b3', g.getData('b3'))


    # 计算训练集精度
    gtest.setData('X', train_X.T)
    gtest.setData('Y', train_Y.T)

    gtest.fprop()

    print('train accuracy: %s' %(gtest.getData('J')))

    # 计算测试集精度
    gtest.setData('X', test_X.T)
    gtest.setData('Y', test_Y.T)

    gtest.fprop()

    print('test accuracy: %s' %(gtest.getData('J')))




def no_test_dropout_regularization () :
    train_X, train_Y, test_X, test_Y = load_2D_dataset()

    assert train_X.shape == (2, 211)
    assert train_Y.shape == (1, 211)
    assert test_X.shape == (2, 200)
    assert test_Y.shape == (1, 200)

    rate = 0.3
    num_iterations = 30000

    np.random.seed(1)
    g = nous(
        '''
        X:(211, 2) ->

            matmul W1:he((2, 20), 4) add b1:(20) -> relu -> dropout(0.86) ->
            matmul W2:he((20, 3), 40) add b2:(3) -> relu -> dropout(0.86) ->
            matmul W3:he((3, 1), 6) add b3:(1) -> sigmoid ->
        
        cross_entropy Y:(211, 1) -> cost -> J:$$
        '''
    ).parse()

    g.setData('X', train_X.T)
    g.setData('Y', train_Y.T)

    def update_func (z, dz) :
        return z - rate * dz

    g.setUpdateFunc('W1', update_func)
    g.setUpdateFunc('W2', update_func)
    g.setUpdateFunc('W3', update_func)
    g.setUpdateFunc('b1', update_func)
    g.setUpdateFunc('b2', update_func)
    g.setUpdateFunc('b3', update_func)

    for i in range(num_iterations) :
        g.fprop()
        g.bprop()

        if i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, g.getData('J')))
    print("Cost after iteration {}: {}".format(num_iterations, g.getData('J')))
    

    # 测试

    gtest = nous(
        '''
        X ->

            matmul W1 add b1 -> relu ->
            matmul W2 add b2 -> relu ->
            matmul W3 add b3 -> sigmoid ->
        
        threshold(0.5) -> accuracy Y -> J:$$
        '''
    ).parse()

    # 迁移学习好的参数
    gtest.setData('W1', g.getData('W1'))
    gtest.setData('W2', g.getData('W2'))
    gtest.setData('W3', g.getData('W3'))
    gtest.setData('b1', g.getData('b1'))
    gtest.setData('b2', g.getData('b2'))
    gtest.setData('b3', g.getData('b3'))


    # 计算训练集精度
    gtest.setData('X', train_X.T)
    gtest.setData('Y', train_Y.T)

    gtest.fprop()

    print('train accuracy: %s' %(gtest.getData('J')))

    # 计算测试集精度
    gtest.setData('X', test_X.T)
    gtest.setData('Y', test_Y.T)

    gtest.fprop()

    print('test accuracy: %s' %(gtest.getData('J')))
