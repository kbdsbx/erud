from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
import h5py
from erud.nous import nous

def load_dataset():
    path = __file__[:__file__.rfind('\\')]
    print(path)
    

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

def test_logistic_regression () :
    train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes = load_dataset()

    assert train_set_x_orig.shape == (209, 64, 64, 3)
    assert train_set_y_orig.shape == (1, 209)
    assert test_set_x_orig.shape == (50, 64, 64, 3)
    assert test_set_y_orig.shape == (1, 50)

    # 中心化、向量化
    train_set_x = train_set_x_orig.reshape((209, 64 * 64 * 3)).T / 255 - 0.5
    test_set_x = test_set_x_orig.reshape((50, 64 * 64 * 3)).T / 255 - 0.5

    assert train_set_x.shape == (12288, 209)
    assert test_set_x.shape == (12288, 50)

    # 训练
    num_iterations = 1000
    rate = 0.005

    g = nous('''
    W:zeros(1, 12288) matmul X add b:0 -> sigmoid as temp -> cross_entropy Y -> cost -> J:$$
    ''').parse()

    g.show()

    g.setData('X', train_set_x)
    g.setData('Y', train_set_y_orig)
    g.setUpdateFunc('W', lambda z, dz : z - rate * dz)
    g.setUpdateFunc('b', lambda z, dz : z - rate * dz)

    for i in range(num_iterations) :

        g.fprop()
        g.bprop()

        if i % 100 == 0 :
            print('Cost after iteration %i : %f' % (i, g.getData('J') ) )


    # 测试

    g1 = nous('''
    W matmul X add b -> sigmoid -> threshold(0.5) -> accuracy Y -> J:$$
    ''').parse()

    g1.setData('X', train_set_x)
    g1.setData('Y', train_set_y_orig)
    g1.setData('W', g.getData('W'))
    g1.setData('b', g.getData('b'))
    
    g1.fprop()

    print('train accuracy: %s' %(g1.getData('J')))

    g1.setData('X', test_set_x)
    g1.setData('Y', test_set_y_orig)

    g1.fprop()

    print('test accuracy: %s' %(g1.getData('J')))
