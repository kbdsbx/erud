import erud
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

import random
from random import shuffle


def transData (data) :
    ix_to_char = {0: '\n', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}

    data = data.reshape((27, 27))
    
    st = ''
    for di in data :
        mx = np.argmax(di)
        if mx == 0 :
            break
        st = st + ix_to_char[mx]
    
    return st


def getXY() :
    path = __file__[:__file__.rfind('\\')]
    with open(path + "/datasets/dinos.txt", 'r') as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    ix_to_char = {0: '\n', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
    char_to_ix = {ix_to_char[k]:k for k in ix_to_char}
    examples.sort(reverse=True, key = lambda ele : len(ele))
    print(examples[0])
    print(examples[len(examples) - 1])
    # 将最长样本长度定为rnn循环次数
    max = len(examples[0])

    X = []
    Y = []

    onehotmap = np.eye(27)

    shuffle(examples)

    # \n
    edp = np.zeros((27)).tolist()
    edp[0] = 1.

    # 补充向量
    nv = np.zeros((27)).tolist()
    nv[0] = 1.

    for exp in examples :
        samp = []
        for i in range(max + 1):
            if i == 0 :
                # 将x的第一个字符设置为0
                samp.append(np.zeros((27)).tolist())
            elif i <= len(exp) :
                samp.append(onehotmap[char_to_ix[exp[i - 1]]].tolist())
            elif i == len(exp) + 1 :
                samp.append(edp)
            else :
                # samp.append(np.zeros((27)).tolist())
                samp.append(nv)
        X.append(samp)
        s = samp[:]
        # s.append(np.zeros((27)).tolist())
        s.append(nv)
        Y.append(s[1:])
    
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)
    # print(X[16])
    # print(Y[16])

    return X, Y



def test_rnn() :
    X, Y = getXY()
    num_iterations = 35001
    rate = 0.01
    
    g = erud.nous(
    '''
    X:(1, 27, 27) scatter(1) -> loop t = 1 to 3 (
        x<t - 1> matmul Wax:xavier(27, 50):adam(0.001) ->
        add (a<t - 1>:(1, 50) matmul Waa:xavier(50, 50):adam(0.001)) ->
        add ba:(50):adam(0.001) ->
        tanh => a<t> ->
        matmul Wya:xavier(50, 27):adam(0.001) add by:(27):adam(0.001) -> softmax(1)
    ) -> gather(1) ->

    cross_entropy Y:(1, 27, 27) -> cost -> J:$$
'''
    ).parse()

    g.setGradientClip('Wax', 5)
    g.setGradientClip('Waa', 5)
    g.setGradientClip('Wya', 5)
    g.setGradientClip('ba', 5)
    g.setGradientClip('by', 5)

    # print(g)
    g.show()

    for j in range(num_iterations) :
        index = j % len(X)
        _x = X[index:index + 1, :, :]
        _y = Y[index:index + 1, :, :]
        g.setData('X', _x)
        g.setData('Y', _y)
        g.fprop()
        g.bprop()

        if j % 1000 == 0 :
            print('Cost after {} iteration is {}'.format(j, g.getData('J')))
            # print(g.getData('Wya'))
            simpling(g)
        
    # simpling(g)



def simpling (g = None) :
    gsimpling = erud.nous(
    '''
    loop t = 1 to 27 (
        x<t - 1> matmul Wax -> add (a<t - 1> matmul Waa) -> add ba -> tanh => a<t> -> matmul Wya add by -> softmax(1) -> choise_to_index => x<t>
    ) -> gather(1) -> to Yhat
'''
    ).parse()

    gsimpling.setData('x0', np.zeros((1, 27)))
    gsimpling.setData('a0', np.zeros((1, 50)))
    gsimpling.setData('Wax', g.getData('Wax'))
    gsimpling.setData('Waa', g.getData('Waa'))
    gsimpling.setData('Wya', g.getData('Wya'))
    gsimpling.setData('ba', g.getData('ba'))
    gsimpling.setData('by', g.getData('by'))

    # print(gsimpling)

    gsimpling.fprop()

    # print(gsimpling.getData('Yhat'))
    d = transData(gsimpling.getData('Yhat'))
    print(d)



