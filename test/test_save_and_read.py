import erud
import os
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

def test_exports() :
    np.random.seed(1)
    g = erud.nous(
        '''
        X ->

            ##### 一层卷积
            ##### (1080, 64, 64, 3) -> (1080, 64, 64, 8) -> (1080, 8, 8, 8)
            conv2d(1, 2) W1:xavier((4, 4, 3, 8), 16) -> relu -> max_pool(8, 8, 8) ->
            ##### 二层卷积
            ##### (1080, 8, 8, 8) -> (1080, 8, 8, 16) -> (1080, 2, 2, 16)
            conv2d(1, 1) W2:xavier((2, 2, 8, 16), 4) -> relu -> max_pool(4, 4, 4) ->

            ##### 全连接
            flatten -> matmul W3:xavier((64, 6), 64) add b3:(6) ->

        softmax_cross_entropy(1) Y -> cost -> J:$$
        '''
    ).parse()

    path = __file__[:__file__.rfind('\\')]
    filename = path + '/datasets/test_export_caches.json'

    g.exports(filename)

    assert os.path.exists(filename) == True

def test_imports() :
    g = erud.nous(
        '''
        X ->

            ##### 一层卷积
            ##### (1080, 64, 64, 3) -> (1080, 64, 64, 8) -> (1080, 8, 8, 8)
            conv2d(1, 2) W1:xavier((4, 4, 3, 8), 16) -> relu -> max_pool(8, 8, 8) ->
            ##### 二层卷积
            ##### (1080, 8, 8, 8) -> (1080, 8, 8, 16) -> (1080, 2, 2, 16)
            conv2d(1, 1) W2:xavier((2, 2, 8, 16), 4) -> relu -> max_pool(4, 4, 4) ->

            ##### 全连接
            flatten -> matmul W3:xavier((64, 6), 64) add b3:(6) ->

        softmax_cross_entropy(1) Y -> cost -> J:$$
        '''
    ).parse()

    path = __file__[:__file__.rfind('\\')]
    filename = path + '/datasets/test_export_caches.json'

    g.imports(filename)

    assert g.getData('W1')[0][0][0][0] == 0.4060863409158104
    assert g.getData('W2')[0][0][0][0] == 0.30899276696016736

    # os.remove(filename)


def test_exports_from_nous() :
    np.random.seed(1)
    n = erud.nous(
        '''
        X ->

            ##### 一层卷积
            ##### (1080, 64, 64, 3) -> (1080, 64, 64, 8) -> (1080, 8, 8, 8)
            conv2d(1, 2) W1:xavier((4, 4, 3, 8), 16) -> relu -> max_pool(8, 8, 8) ->
            ##### 二层卷积
            ##### (1080, 8, 8, 8) -> (1080, 8, 8, 16) -> (1080, 2, 2, 16)
            conv2d(1, 1) W2:xavier((2, 2, 8, 16), 4) -> relu -> max_pool(4, 4, 4) ->

            ##### 全连接
            flatten -> matmul W3:xavier((64, 6), 64) add b3:(6) ->

        softmax_cross_entropy(1) Y -> cost -> J:$$
        '''
    )
    n.parse()

    path = __file__[:__file__.rfind('\\')]
    filename = path + '/datasets/test_export_nous_caches.json'

    erud.nous_exports(n, filename)

    assert os.path.exists(filename) == True

def test_imports_from_nous() :
    n = erud.nous()
    path = __file__[:__file__.rfind('\\')]
    filename = path + '/datasets/test_export_nous_caches.json'

    erud.nous_imports(n, filename)

    assert n.g.getData('W1')[0][0][0][0] == 0.4060863409158104
    assert n.g.getData('W2')[0][0][0][0] == 0.30899276696016736

    # os.remove(filename)



