import erud
import os
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np
import pytest as test

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

    n.g.setUpdateFunc('W1', erud.upf.norm(0.66))
    n.g.setUpdateFunc('W2', erud.upf.adam(0.99))

    path = __file__[:__file__.rfind('\\')]
    filename = path + '/datasets/test_export_nous_caches.json'

    erud.exports(n, filename, {
        'a' : 0,
        'b' : 1,
    })

    assert os.path.exists(filename) == True

def test_imports_from_nous() :
    path = __file__[:__file__.rfind('\\')]
    filename = path + '/datasets/test_export_nous_caches.json'

    n, obj = erud.imports(filename)

    # n.g.setUpdateFunc('W1', erud.upf.norm(0.5))
    # n.g.setUpdateFunc('W2', erud.upf.norm(0.5))

    assert obj['a'] == 0
    assert obj['b'] == 1
    assert n.g.getData('W1')[0, 0, 0, 0] == 0.4060863409158104
    assert n.g.getData('W2')[0, 0, 0, 0] == 0.30899276696016736
    assert isinstance(n.g.nodes[2].data.update_func, erud.upf.norm_class)
    assert isinstance(n.g.nodes[6].data.update_func, erud.upf.adam_class)
    assert n.g.nodes[2].data.update_func.rate == 0.66
    assert n.g.nodes[6].data.update_func.rate == 0.99

    # os.remove(filename)

def test_imports_from_error_file() :
    path = __file__[:__file__.rfind('\\')]
    filename = path + '/datasets/test_export_nous_caches_none_file.json'

    with test.raises(FileNotFoundError) :
        erud.imports(filename)



