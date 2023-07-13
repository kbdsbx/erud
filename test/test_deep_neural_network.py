from erud.nous import nous
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np
import h5py

def load_dataset():
    path = __file__[:__file__.rfind('\\')]
    

    train_dataset = h5py.File(path + '/datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(path + '/datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes


def no_test_deep_neural_network () :
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, _ = load_dataset()

    assert train_set_x_orig.shape == (209, 64, 64, 3)
    assert train_set_y_orig.shape == (1, 209)
    assert test_set_x_orig.shape == (50, 64, 64, 3)
    assert test_set_y_orig.shape == (1, 50)

    # 中心化、向量化
    train_set_x = train_set_x_orig.reshape((209, 64 * 64 * 3)).T / 255
    test_set_x = test_set_x_orig.reshape((50, 64 * 64 * 3)).T / 255

    assert train_set_x.shape == (12288, 209)
    assert test_set_x.shape == (12288, 50)

    # 训练
    num_iterations = 1000
    rate = 0.005

    g = nous('''
    X:(209, 12288) ->

        matmul W1:(12288, 20) add b1:(20) -> relu ->
        matmul W2:(20, 7) add b2:(7) -> relu ->
        matmul W3:(7, 5) add b3:(5) -> relu ->
        matmul W4:(5, 1) add b4:(1) -> sigmoid ->
    
    cross_entropy Y:(209, 1) -> cost -> J:$$
    ''').parse()

    # 初始化样本和训练参数
    np.random.seed(1)
    g.setData('X', train_set_x.T)
    g.setData('W1', np.random.randn(12288, 20) / np.sqrt(12288))
    g.setData('W2', np.random.randn(20, 7) / np.sqrt(20))
    g.setData('W3', np.random.randn(7, 5) / np.sqrt(7))
    g.setData('W4', np.random.randn(5, 1) / np.sqrt(5))
    g.setData('Y', train_set_y_orig.T )
    
    # 添加更新方法
    def update_func(z, dz) :
        return z - rate * dz

    g.setUpdateFunc('W1', update_func)
    g.setUpdateFunc('W2', update_func)
    g.setUpdateFunc('W3', update_func)
    g.setUpdateFunc('W4', update_func)
    g.setUpdateFunc('b1', update_func)
    g.setUpdateFunc('b2', update_func)
    g.setUpdateFunc('b3', update_func)
    g.setUpdateFunc('b4', update_func)

    for i in range(num_iterations) :
        g.fprop()
        g.bprop()

        if i % 100 == 0 :
            print('Cost after iteration %i : %f' % (i, g.getData('J') ) )
        
    
    # 测试

    gtest = nous( '''
    X ->

        matmul W1 add b1 -> relu ->
        matmul W2 add b2 -> relu ->
        matmul W3 add b3 -> relu ->
        matmul W4 add b4 -> sigmoid ->
    
    threshold(0.5) -> accuracy Y -> J:$$
    ''').parse()
    

    # 迁移学习好的参数
    gtest.setData('W1', g.getData('W1'))
    gtest.setData('W2', g.getData('W2'))
    gtest.setData('W3', g.getData('W3'))
    gtest.setData('W4', g.getData('W4'))
    gtest.setData('b1', g.getData('b1'))
    gtest.setData('b2', g.getData('b2'))
    gtest.setData('b3', g.getData('b3'))
    gtest.setData('b4', g.getData('b4'))

    # 计算训练集精度
    gtest.setData('X', train_set_x.T)
    gtest.setData('Y', train_set_y_orig.T)

    gtest.fprop()

    print('train accuracy: %s' %(gtest.getData('J')))

    # 计算测试集精度
    gtest.setData('X', test_set_x.T)
    gtest.setData('Y', test_set_y_orig.T)

    gtest.fprop()

    print('test accuracy: %s' %(gtest.getData('J')))
