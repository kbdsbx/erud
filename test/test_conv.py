import h5py
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np
import erud
import math
import time

def load_dataset():
    path = __file__[:__file__.rfind('\\')]
    print(path)
    
    train_dataset = h5py.File(path + '/datasets/train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(path + '/datasets/test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def set_mini_batch(train_X, train_Y, m, s) :

    permutation = list(np.random.permutation(m))
    shuffled_X = train_X[permutation, :, :, :]
    shuffled_Y = train_Y[permutation, :].reshape((m, 6))

    n = math.floor(m/s)
    batches = []

    for k in range(0, n) :
        mX = shuffled_X[k*s : (k+1) * s, :, :, :]
        mY = shuffled_Y[k*s : (k+1) * s, :]

        miniB = (mX, mY)
        batches.append(miniB)
    
    if m % s != 0 :
        mX = shuffled_X[m - (m % s):, :, :, :]
        mY = shuffled_Y[m - (m % s):, :]

        miniB = (mX, mY)
        batches.append(miniB)

    
    return batches

def no_test_conv () :
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

    assert train_set_x_orig.shape == (1080, 64, 64, 3)
    assert train_set_y_orig.shape == (1, 1080)
    assert test_set_x_orig.shape == (120, 64, 64, 3)
    assert test_set_y_orig.shape == (1, 120)

    # 中心化、向量化
    train_set_x = train_set_x_orig / 255 - 0.5
    test_set_x = test_set_x_orig / 255 - 0.5
    # 标签转为onehot向量
    train_set_y = np.eye(6)[train_set_y_orig.reshape(-1)]
    test_set_y = np.eye(6)[test_set_y_orig.reshape(-1)]

    assert train_set_x.shape == (1080, 64, 64, 3)
    assert train_set_y.shape == (1080, 6)
    assert test_set_x.shape == (120, 64, 64, 3)
    assert test_set_y.shape == (120, 6)

    batches = set_mini_batch(train_set_x, train_set_y, 1080, 32)

    assert batches[0][0].shape == (32, 64, 64, 3)
    assert batches[0][1].shape == (32, 6)

    num_iterations = 1
    rate = 0.04

    g = erud.nous(
        '''
        X:(1080, 64, 64, 3) ->

            ##### 一层卷积
            ##### (1080, 64, 64, 3) -> (1080, 64, 64, 8) -> (1080, 8, 8, 8)
            conv2d(1, 2) W1:xavier((4, 4, 3, 8), 16) -> relu -> max_pool(8, 8, 8) ->
            ##### 二层卷积
            ##### (1080, 8, 8, 8) -> (1080, 8, 8, 16) -> (1080, 2, 2, 16)
            conv2d(1, 1) W2:xavier((2, 2, 8, 16), 4) -> relu -> max_pool(4, 4, 4) ->

            ##### 全连接
            flatten -> matmul W3:xavier((64, 6), 64) add b3:(6) ->

        softmax_cross_entropy(1) Y:(1080, 6) -> cost -> J:$$
        '''
    ).parse()

    g.setUpdateFunc('W1', erud.upf.norm(rate))
    g.setUpdateFunc('W2', erud.upf.norm(rate))
    g.setUpdateFunc('W3', erud.upf.norm(rate))
    g.setUpdateFunc('b3', erud.upf.norm(rate))

    tic = time.time()

    for i in range (num_iterations) :
        for b in batches :
            g.setData('X', b[0])
            g.setData('Y', b[1])

            g.fprop()
            g.bprop()

        # if i % 1 == 0 :
            print("Cost after iteration {}: {}.".format(i, g.getData('J')))
    print("Cost after iteration {}: {}.".format(num_iterations, g.getData('J')))

    toc = time.time()


    print(g.tableTimespend())
    print("Cost of time is %fs." % ((toc - tic)))


    gtest = erud.nous(
        '''
        X ->

            ##### 一层卷积
            ##### (1080, 64, 64, 3) -> (1080, 64, 64, 8) -> (1080, 8, 8, 8)
            conv2d(1, 2) W1 -> relu -> max_pool(8, 8, 8) ->
            ##### 二层卷积
            ##### (1080, 8, 8, 8) -> (1080, 8, 8, 16) -> (1080, 2, 2, 16)
            conv2d(1, 1) W2 -> relu -> max_pool(4, 4, 4) ->

            ##### 全连接
            flatten -> matmul W3 add b3 ->

        max_index(1) -> accuracy Y -> J:$$
        '''
    ).parse()


    # 迁移参数
    gtest.setData('W1', g.getData('W1'))
    gtest.setData('W2', g.getData('W2'))
    gtest.setData('W3', g.getData('W3'))
    gtest.setData('b3', g.getData('b3'))


    # 计算训练集精度
    gtest.setData('X', train_set_x)
    gtest.setData('Y', train_set_y_orig.T)

    gtest.fprop()

    print('train accuracy: %s' %(gtest.getData('J')))

    # 计算测试集精度
    gtest.setData('X', test_set_x)
    gtest.setData('Y', test_set_y_orig.T)

    gtest.fprop()

    print('test accuracy: %s' %(gtest.getData('J')))



        

