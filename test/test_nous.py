from erud.nous import nous
from erud.errors import ParseError
import pytest as test
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np


# 获取代码串中的第一个元素和剩余的子串
def test_nous_get_next_el() :
    n = nous()

    assert n._getNextEl('X:1332') == ('X:1332', '')
    assert n._getNextEl(' X:') == ('X:', '')
    assert n._getNextEl('   X[]') == ('X[]', '')
    assert n._getNextEl('X[[1, (2)], [(3), 4]]') == ('X[[1, (2)], [(3), 4]]', '')
    assert n._getNextEl('x:42 error sse') == ('x:42', 'error sse')
    assert n._getNextEl('X:(1000, 10) matmul W1:r(10, 50) -> ') == ('X:(1000, 10)', 'matmul W1:r(10, 50) ->')
    assert n._getNextEl('(((((test)))))') == ('((((test))))', '')
    assert n._getNextEl('((([((test))])))') == ('(([((test))]))', '')
    assert n._getNextEl('(((((test))))) rest') == ('(((((test)))))', 'rest')
    assert n._getNextEl('((([((test))]))) rest') == ('((([((test))])))', 'rest')
    assert n._getNextEl('(a add b) rest') == ('(a add b)', 'rest')
    assert n._getNextEl('') == ('', '')
    assert n._getNextEl('(a add b)') == ('a add b', '')

    # 错误的表达式无法解析，半个中括号
    with test.raises(ParseError) :
        n._getNextEl('   X[')

    # 错误的表达式无法解析，右侧少一个小括号
    with test.raises(ParseError) :
        n._getNextEl('''(((([[{{43ss}}]])))''')


def test_nous_is_block() :
    n = nous()

    assert n._isBlock('X -> Y') == True
    assert n._isBlock('X:1122') == False
    assert n._isBlock('X:') == False
    assert n._isBlock('X:[[1, (2)], [(3), 4]]') == False
    assert n._isBlock('(((a add b)))') == True
    assert n._isBlock('X:(1000, 10) matmul W1:r(10, 50)') == True
    assert n._isBlock('') == False
    assert n._isBlock('(((([((test))]))))') == True
    assert n._isBlock('X:rand(10, 15)') == False
    assert n._isBlock('    X[1, 4]') == False
    assert n._isBlock('(a add b)') == True
    assert n._isBlock('_1:(5, 2, 4)') == False

def test_nous_is_operator () :
    n = nous ()
    assert n._isOperator('then') == False
    assert n._isOperator('as') == False
    assert n._isOperator('rest') == False
    assert n._isOperator('') == False
    assert n._isOperator('add') == True
    assert n._isOperator('sub') == True
    assert n._isOperator('mul') == True
    assert n._isOperator('div') == True
    assert n._isOperator('div(1, 2)') == True
    assert n._isOperator('div((1, 2), [[4, 2], [1, 3]])') == True
    assert n._isOperator('div(-5, 4, 1e-8)') == True

from erud.cg.node import ComputationNode as node
from erud.opts_extend.threshold import threshold as threshold
from erud.opts.add import add

def test_nous_make_operator () :
    n = nous ()

    nd = n._makeOperator('add')
    assert isinstance(nd, node )
    assert isinstance(nd.data, add)

    nd = n._makeOperator('threshold(0.7)')
    assert isinstance(nd, node)
    assert isinstance(nd.data, threshold)
    assert nd.data.threshold == 0.7

    with test.raises(ParseError) :
        n._makeOperator('then')


from erud.cg.graph import ComputationGraph as graph

def test_nous_is_reference() :
    g = graph()
    n1 = node(add())
    n1.data.name = "X-add"
    g.insertNode(n1)

    n = nous()

    assert n._isReference('X', g) == False
    assert n._isReference('X-add', g) == True
    assert n._isReference('then', g) == False

def test_nous_get_reference() :
    g = graph()
    n1 = node(add())
    n1.data.name = "X-add"
    g.insertNode(n1)

    n = nous()

    d = n._getReference('X-add', g)

    assert isinstance(d, node)
    assert isinstance(d.data, add)
    assert d.data.name == 'X-add'

    with test.raises(ParseError) :
        n._getReference('X', g)
    

def test_make_arguments() :
    n = nous()

    assert n._makeArguments('') == []
    assert n._makeArguments('(5, 3, 2)') == [(5,3,2)]
    assert n._makeArguments('5, 3, 2') == [5,3,2]
    assert n._makeArguments('5, [1, 2], 2') == [5,[1,2],2]
    assert n._makeArguments('5, [[1,2], [3,4]], 2, (5.5, 7, -10)') == [5,[[1, 2], [3,4]],2, (5.5, 7, -10)]

    with test.raises(NameError) :
        n._makeArguments('abc')

    with test.raises(ParseError) :
        n._makeArguments('5,3,')

    with test.raises(ParseError) :
        n._makeArguments('[1,2],,5.5')


def test_is_init_func () :
    n = nous()

    assert n._isInitFunc('randn(5, 3, 2)') == True
    assert n._isInitFunc('randn(1)') == True
    assert n._isInitFunc('add(1, 4)') == False
    assert n._isInitFunc('randn(1, )') == True
    assert n._isInitFunc('randn(1, abc)') == False
    assert n._isInitFunc('randn(1.1, 5)') == True
    assert n._isInitFunc('randn(-1.1, -5)') == True
    assert n._isInitFunc('randn(-1.1, 1e-8)') == True
    assert n._isInitFunc('randn(-1.1, 1e-8, [[1,2], [3,4]], (5, 6))') == True
    assert n._isInitFunc('randn((1, 2), 1e-8, [[1,2], [3,4]], (5, 6))') == True
    assert n._isInitFunc('adam(0.008, 0.88, 0.8)') == False

def test_is_update_func () :
    n = nous()
    
    assert n._isUpdateFunc('randn(5,3,2)') == False
    assert n._isUpdateFunc('norm') == False
    assert n._isUpdateFunc('norm(0.9)') == True
    assert n._isUpdateFunc('adam(0.9, 0.99, 0.999)') == True
    assert n._isUpdateFunc('momentum(2, 1e-8, 0.3)') == True
    assert n._isUpdateFunc('momentum(2, [[1,2], [3,4]], 0.3)') == True

def test_is_name() :
    n = nous()

    assert n._isName('') == True
    assert n._isName('X') == True
    assert n._isName('_4322') == True
    assert n._isName('abc_123') == True
    assert n._isName('5s34') == False
    assert n._isName('randn(1)') == False
    assert n._isName('zeros(1, 2, 3)') == False
    assert n._isName('then') == False
    assert n._isName('as') == False
    assert n._isName('add') == False
    assert n._isName('X Y') == False

def test_is_tensor() :
    n = nous()

    assert n._isTensor('[1,2]') == True 
    assert n._isTensor('123') == True 
    assert n._isTensor('0.52341') == True 
    assert n._isTensor('[[1,2], [4,5]]') == True 
    assert n._isTensor('[[1,2], [4,5]][]') == False
    assert n._isTensor('') == False
    assert n._isTensor('abc') == False
    assert n._isTensor('adam(0.00008, 0.88, 0.8)') == False

def test_is_value () :
    n = nous()

    assert n._isValue('5') == True
    assert n._isValue('123') == True
    assert n._isValue('[[1,2], [9.1, 4]]') == True
    assert n._isValue('randn(5,2, 4 )') == True
    assert n._isValue('zeros(5,2, 4 )') == True
    assert n._isValue('155.4') == True
    assert n._isValue('-155.4') == True
    assert n._isValue('[[1,2], [9.1, 4]], []') == False
    assert n._isValue('-1.55.4') == False
    assert n._isValue('a add b') == False
    assert n._isValue('then') == False
    assert n._isValue('add') == False
    assert n._isValue('$$')== False
    assert n._isValue('_1:(5, 2, 4)') == False
    assert n._isValue('adam(0.0008, 0.88, 0.8)') == False

def test_make_value () :
    n = nous()
    assert n._makeValue('155.4') == 155.4
    assert n._makeValue('-155.4') == -155.4

    np.random.seed(1)
    t = n._makeValue('(5, 2, 4)')
    assert isinstance(t, np.ndarray)
    assert t.shape == (5,2,4)
    assert t[0][0][0] == 0
    assert t[4][1][3] == 0

    np.random.seed(1)
    t = n._makeValue('randn(5, 2, 4)')
    assert isinstance(t, np.ndarray)
    assert t.shape == (5,2,4)
    assert t[0][0][0] == 1.6243453636632417
    assert t[4][1][3] == 0.7420441605773356

    t = n._makeValue('zeros( 2, 3, 8, 1 )')
    assert isinstance(t, np.ndarray)
    assert t.shape == (2,3,8,1)
    assert t[0][0][0][0] == 0
    assert t[1][2][7][0] == 0

    t = n._makeValue('ones( 2, 3, 8, 1 )')
    assert isinstance(t, np.ndarray)
    assert t.shape == (2,3,8,1)
    assert t[0][0][0][0] == 1
    assert t[1][2][7][0] == 1

    t = n._makeValue('[[ -1, 55.2 ], [ 0, 5.554 ]]')
    assert isinstance(t, np.ndarray)
    assert t.shape == (2,2)
    assert t[0][0] == -1
    assert t[1][1] == 5.554

from erud.upf.norm import norm
from erud.upf.adam import adam

def test_make_update_func() :
    n = nous()
    uf = n._makeUpdateFunc('norm(0.8)')
    assert isinstance(uf, norm)
    assert uf.rate == 0.8

# 变量格式
def test_make_variable() :
    n = nous()
    
    nd = n._makeVariable('X')
    assert nd.data.name == 'X'
    assert nd.data.data == None

    nd = n._makeVariable('X:')
    assert nd.data.name == 'X'
    assert nd.data.data == None

    nd = n._makeVariable('5')
    assert nd.data.name == None
    assert nd.data.data == 5

    nd = n._makeVariable(':5')
    assert nd.data.name == ''
    assert nd.data.data == 5

    nd = n._makeVariable('X:5')
    assert nd.data.name == "X"
    assert nd.data.data == 5

    nd = n._makeVariable('X:-5.5')
    assert nd.data.name == "X"
    assert nd.data.data == -5.5

    nd = n._makeVariable('X:[[1.1, 0], [-4, 0.33]]')
    assert nd.data.name == "X"
    assert isinstance(nd.data.data, np.ndarray)
    assert nd.data.data.shape == (2,2)
    assert nd.data.data[0][0] == 1.1
    assert nd.data.data[1][1] == 0.33

    nd = n._makeVariable('[[1.1, 0], [-4, 0.33]]')
    assert nd.data.name == None
    assert isinstance(nd.data.data, np.ndarray)
    assert nd.data.data.shape == (2,2)
    assert nd.data.data[0][0] == 1.1
    assert nd.data.data[1][1] == 0.33

    np.random.seed(1)
    nd = n._makeVariable('_X:randn(5, 2, 4)')
    assert nd.data.name == "_X"
    assert isinstance(nd.data.data, np.ndarray)
    assert nd.data.data.shape == (5,2,4)
    assert nd.data.data[0][0][0] == 1.6243453636632417
    assert nd.data.data[4][1][3] == 0.7420441605773356

    nd = n._makeVariable('_1:(5, 2, 4)')
    assert nd.data.name == "_1"
    assert isinstance(nd.data.data, np.ndarray)
    assert nd.data.data.shape == (5,2,4)
    assert nd.data.data[0][0][0] == 0
    assert nd.data.data[4][1][3] == 0

    nd = n._makeVariable('中文:(5, 2, 4)')
    assert nd.data.name == '中文'
    assert isinstance(nd.data.data, np.ndarray)
    assert nd.data.data.shape == (5,2,4)
    assert nd.data.data[0][0][0] == 0
    assert nd.data.data[4][1][3] == 0

    nd = n._makeVariable('(5, 2, 4)')
    assert nd.data.name == None
    assert isinstance(nd.data.data, np.ndarray)
    assert nd.data.data.shape == (5,2,4)
    assert nd.data.data[0][0][0] == 0
    assert nd.data.data[4][1][3] == 0

    nd = n._makeVariable('_1:(5):norm(0.0008)')
    assert nd.data.name == '_1'
    assert isinstance(nd.data.data, np.ndarray)
    assert nd.data.data.shape == (5,)
    assert nd.data.data[0] == 0
    assert isinstance(nd.data.update_func, norm)
    assert nd.data.update_func.rate == 0.0008

    nd = n._makeVariable('X:adam(0.0008, 0.9, 0.99)')
    assert nd.data.name == 'X'
    assert nd.data.data == None
    assert isinstance(nd.data.update_func, adam)
    assert nd.data.update_func.rate == 0.0008


    nd = n._makeVariable('randn(2, 5, 4):adam(0.0008, 0.9, 0.99)')
    assert nd.data.name == None
    assert isinstance(nd.data.data, np.ndarray)
    assert nd.data.data.shape == (2, 5, 4)
    assert isinstance(nd.data.update_func, adam)
    assert nd.data.update_func.rate == 0.0008

    with test.raises(ParseError) :
        n._makeVariable('X:X:X')

    with test.raises(ParseError) :
        n._makeVariable('X:X:5')

    with test.raises(ParseError) :
        n._makeVariable('[[1.2],],[]')

    with test.raises(ParseError) :
        n._makeVariable(':[[1.2],],[]')

    with test.raises(ParseError) :
        n._makeVariable('add(1,2, 3)')

    with test.raises(ParseError) :
        n._makeVariable('zeros(1,,3)')

    with test.raises(ParseError) :
        n._makeVariable('J:$$')
    

def test_nous_is_rest () :
    n = nous()

    assert n._isRest('J:XX') == False
    assert n._isRest('J:$$') == True
    assert n._isRest('_:$$') == True
    assert n._isRest(':$$') == True
    assert n._isRest('$$') == True
    assert n._isRest('rest')== True
    assert n._isRest('J:rest')== True
    assert n._isRest('J:[1,2]')== False

from erud.tensor.rest import rest

def test_make_rest() :
    n = nous()

    nd = n._makeRest('$$')
    assert isinstance(nd.data, rest)
    assert nd.data.name == None

    nd = n._makeRest('J:$$')
    assert isinstance(nd.data, rest)
    assert nd.data.name == 'J'

    nd = n._makeRest('J:rest')
    assert isinstance(nd.data, rest)
    assert nd.data.name == 'J'

    nd = n._makeRest(':rest')
    assert isinstance(nd.data, rest)
    assert nd.data.name == None

    with test.raises(ParseError) :
        n._makeRest('J:#')

    with test.raises(ParseError) :
        n._makeRest('6')

from erud.tensor.var import var
from erud.tensor.rest import rest

def test_process_block() :

    n = nous()
    g = graph()

    # 左操作数和子块并列，中间没有操作符
    with test.raises(ParseError) :
        n._processBlock('X (a add b)', g)

    # 右操作数和子块并列，中间没有操作符
    with test.raises(ParseError) :
        n._processBlock('X add Y (a add b)', g)

    # 操作符不能在整个表达式首部
    with test.raises(ParseError) :
        n._processBlock('add Y', g)

    # 两个操作符并排出现
    with test.raises(ParseError) :
        n._processBlock('(a add b) add mul c', g)

    # 两个合法变量同时出现
    with test.raises(ParseError) :
        n._processBlock('X:5 Y:[1, 2] add Z', g)
    with test.raises(ParseError) :
        n._processBlock('X:5 add Y:[1, 2] Z', g)

    # 只有操作符可以使用as来命名
    with test.raises(ParseError) :
        n._processBlock(':5 as X', g)

    # 非法名称
    with test.raises(ParseError) :
        n._processBlock('X add Y as 5', g)

    # 层连接符放在首部
    with test.raises(ParseError) :
        n._processBlock('->', g)

    # 终止符放在首部
    with test.raises(ParseError) :
        n._processBlock('$$ add X', g)
    
    g = graph()

    n._processBlock('X:(1) add Y:[[1,2], [3.4, 0]] -> sub Z -> $$', g)
    assert isinstance( g.nodes[0].data, var )
    assert isinstance( g.nodes[0].data.data, np.ndarray)
    assert g.nodes[0].data.data[0] == 0
    assert isinstance( g.nodes[1].data, add )
    assert isinstance( g.nodes[2].data, var )
    assert isinstance( g.nodes[2].data.data, np.ndarray )
    assert g.nodes[2].data.data[0][0] == 1
    assert g.nodes[2].data.data[1][1] == 0
    assert isinstance( g.nodes[5].data, rest )
    assert g.nodes[5].data.name == None

    g = graph()
    n._processBlock('5 add 10 as u1', g)
    assert g.nodes[1].data.name == 'u1'
    n._processBlock('6 sub 19 as u2', g)
    assert g.nodes[4].data.name == 'u2'
    n._processBlock('u1 mul u2 -> div 3 -> res:$$', g)
    
    [res] = g.fprop()
    assert res.data == -65

    g = graph()
    np.random.seed(1)
    n._processBlock('X:[[1,2], [3,4]] add Y:randn(2, 2) sub :[-1, -2] -> div 0.01 -> $$', g)
    [res] = g.fprop()

    assert res.data.shape == (2,2)

    assert np.all(res.data == np.array([[362.43453636632415, 338.82435863499245], [347.18282477365443, 492.70313778438293]]))

    g = graph()
    n._processBlock('5 add 10 as u', g)
    n._processBlock('6 sub 19 as v', g)
    n._processBlock('u mul v as t', g)
    n._processBlock('t div 3 then y:$$', g)
    [res] = g.fprop()

    assert res.data == -65

    g = graph()
    n._processBlock('5 add 10 as u mul (6 sub 19 as v) as t div 3 then y:$$', g)
    assert g.nodes[1].data.name == 'u'
    assert g.nodes[4].data.name == 'v'
    assert g.nodes[6].data.name == 't'
    assert g.nodes[9].data.name == 'y'
    [res] = g.fprop()

    assert res.data == -65

    # g = graph()
    # n._processBlock('a:5 add b:10 as u mul (c:6 sub d:19 as v) as w div e:3 as t then y:$$', g)
    # print(g)

def test_nous_parse() :
    n = nous('''

    # 这里是注释
    5 add 10 as u
    6 sub 19 as v

        # 下面两个是一行
        u mul v ->
        div 3 as t

    # 结束
    t -> y:$$
    ''')

    g = n.parse()
    [res] = g.fprop()

    assert res.data == -65

    # X被多次引用计算
    n = nous('''
    X:5 add X mul X sub X mul X -> $$
    ''')

    [res] = n.parse().fprop()

    assert res.data == 225

    # 测试一元运算符
    n = nous(
        '''
        5 add -10 -> relu -> sigmoid -> J:$$
        '''
    )
    g = n.parse()
    g.fprop()
    assert g.getData('J') == 0.5

def test_nous_parse_difficult () :
    n = nous(
        '''
        X:(211, 2) ->

            matmul W1:he((2, 20), 4) add b1:(20) -> relu ->
            matmul W2:he((20, 3), 40) add b2:(3) -> relu ->
            matmul W3:he((3, 1), 6) add b3:(1) -> sigmoid ->
        
        cross_entropy Y:(211, 1) -> cost as entropy_cost

        # L2 正则

        (1.0 div 2) mul (lambd:0.1 div 2) mul ((W1 mul W1 -> sum) add (W2 mul W2 -> sum) add (W3 mul W3 -> sum)) as l2_reg_cost

        entropy_cost add l2_reg_cost -> J:$$
        '''
    ).parse()

