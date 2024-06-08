from erud.cg.graph import ComputationGraph as graph
from erud.cg.node import ComputationNode as node
from erud.errors import ParseError
from erud.opts.add import add
from erud.opts.sub import sub
from erud.opts.mul import mul
from erud.opts.div import div
from erud.opts.sum import sum
from erud.opts.matmul import matmul
from erud.opts.relu import relu
from erud.opts.sigmoid import sigmoid
from erud.opts.L1 import L1
from erud.opts.L2 import L2
from erud.opts.softmax import softmax
from erud.opts.tanh import tanh
from erud.opts.cross_entropy import cross_entropy
from erud.opts.cost import cost
from erud.opts.dropout import dropout
from erud.opts.batchnorm import batchnorm
from erud.opts.batchnorm2d import batchnorm2d
from erud.opts.softmax_cross_entropy import softmax_cross_entropy
# from erud.opts.conv2d import conv2d
from erud.opts.conv2d_v2 import conv2d_v2
from erud.opts.conv2d_v3 import conv2d_v3
from erud.opts.conv2d_same import conv2d_same
from erud.opts.conv2d_v3_same import conv2d_v3_same
from erud.opts.max_pool import max_pool
from erud.opts.max_pool_v3 import max_pool_v3
from erud.opts.flatten import flatten
from erud.opts.L2_regularization import L2_regularization
from erud.opts.scatter import scatter
from erud.opts.gather import gather
from erud.opts.leaky_relu import leaky_relu
from erud.opts.reshape import reshape

from erud.tensor.var import var
from erud.tensor.rest import rest
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
import re

from erud.opts_extend.accuracy import accuracy
from erud.opts_extend.threshold import threshold
from erud.opts_extend.max_index import max_index
from erud.opts_extend.choise_to_index import choise_to_index
from erud.opts_extend.yolo1_loss import yolo1_loss

import erud.upf as upf

# 解析代码，构造计算图
class nous :
    """
    ## nous

    nous 是一套格式化语言，可以帮助各位走上智识的命途，领略星神博识尊践行的命运真谛（计算图快速搭建语言）

    示例：

    ```python
    graph = '''
    X matmul W1 then relu then matmul W2 then cross_entropy Y as J
    W2 then sqr then sum as C
    W1 then sqr then sum then plus C then times LAMBDA then plus J then REST
    '''

    # corss_entropy(sigmoid(relu(relu(relu(relu(X * W1 + B1) * W2 + B2) * W3 + B3) * W4 + B4)))
    graph = '''
    X:(1000, 10) matmul W1:r(10, 50) -> add B1:z(40) -> relu -> matmul W2:z(40, 20) -> add B2:z(20) -> relu -> matmul W3:r(20, 10) -> add B3:z(10) -> matmul W4:r(10, 1) -> add B4:z(1) -> sigmoid -> cross_entropy y:(1000) -> J:$$
    '''

    # rest = [(5 + 10) * (6 - 19) / 3 ] / (-65) - 1
    graph = '''
    5 add 10 as u1
    6 sub 19 as u2
    u1 mul u2 then div 3 then div -65 then sub 1 then REST
    '''

    graph = '''
    # 这里是注释

    # 下面两个各是一行
    5 add 10 as u
    6 sub 19 as v

        # 下面两个是一行
        u mul v ->
        div 3 as t

    # 结束
    t -> y:$$
    '''
    ```

    ### 语法规则

    1. 声明变量
        * ``X``: 声明一个变量，命名为'X'，不赋值
        * ``:5`` 或 ``5``: 声明一个匿名变量，赋值``5``。如果没有冒号``':'``，当声明不满足于变量命名规范时视为值，否则视为变量
        * ``X:5``: 声明一个变量，命名为'X'，并赋值``5``
        * ``X:randn(10, 15)``: 声明一个张量变量，维度为(10,15)，命名为'X'，并随机赋值
        * ``X:zero(10, 15)`` 或 ``X:(10,15)``: 声明一个张量变量，维度为(10,15)，命名为'X'，并赋值为0
        * ``X:ones(10, 15)``: 声明一个张量变量，维度为(10,15)，命名为'X'，并赋值为1
        * ``X:he((10, 15), 10)``: 声明一个张量变量，维度为(10,15)，命名为'X'，使用`he`进行初始化
        * ``X:[[1,2], [3,4]]``: 声明一个张量变量，维度为(2,2)，命名为'X'，并赋值为`[[1,2],[3,4]]`

        目前已有的初始化函数
        1. `zero(<tuple>)`或`(<tuple>)`: 执行零初始化
        2. `ones(<tuple>)`: 执行1初始化
        3. `randn(<tuple>)`: 执行正态随机初始化
        4. `he(<tuple>, layer_num)`: 执行he初始化

    2. 运算符操作

        操作有默认操作和扩展操作两种，其中默认操作为框架内置操作，包括加减乘除、矩阵乘、relu、sigmoid、softplus、softmax、maxout、tanh、rbf、cross_entropy等

        > 其中部分随后实现

        > 未来已来，实现已现

        扩展操作为用户添加操作，高级用户可在默认操作的基础上扩展自己的操作，只需要继承``erud.cg.payload``类，声明关键词，实现``fprop``和``bprop``方法即可

        扩展操作可以覆盖默认操作，如果用户想要替换默认操作，比如想要替换cross_entropy操作，可以自行声明一个类``class cross_entropy (payload)``并将关键词改为``cross_entropy``，框架在装载操作符时就会将用户声明的``cross_entropy``操作替换掉内部默认的``cross_entropy``操作

        具体步骤参考“高级用户的高级玩法”

        * ``5 add 10``: 包含左右操作数的计算声明，操作数为任意类型变量，但用户需要自己检查变量是否满足操作运算的规则（例如矩阵乘法）
        * ``add W1``: 只包含一个操作数的计算声明，如果操作符需要两个操作数，则左操作数为上一层的结果
        * ``relu``: 没有操作数的计算声明，如果操作符需要一个操作数，则左操作数为上一层的结果
        * ``dropout(0.5)``: 带参数的操作符

    3. 层操作

        层操作并不严格定义层，只定义计算图中节点与节点之间的前后关系

        * ``then`` 或 ``->``: 层分隔符，运算时将左边式子的计算结果传递给右边式子的左操作数

    4. 赋值操作

        赋值操作并不产生新的节点，只修改原本节点的命名，以便可以在计算图中按名称找到此节点
        
        * ``a add b as res``: 将加法操作``add``重命名为`res`

    5. 声明结果变量

        计算图中可以有一个或多个结果变量，结果变量存放结果值，计算图的前向传播`fprop`会以数组的方式返回所有结果，结果变量为计算图或子图的汇点

        匿名变量只能由`fprop`获得，命名变量也可以由查找节点的方法`getData`在前向传播完成后取得对应的值

        * ``... -> rest``或``... -> $$``: 声明匿名结果变量节点
        * ``... -> J:rest``或``... -> J:$$``: 声明结果变量节点，并命名为``J``

    6. 声明子块

        * ``(a add b) then mul c``: 小括号用于声明子块
        * 使用换行来声明子块，每一行为一个子块

            ```python
            code = '''
            5 add 10 as u1
            6 sub 19 as u2
            u1 mul u2 then REST
            '''
            ```

    7. 换行

        * 当尾部为层连接符时，此行与下一行属于同一个子块

            ```python
            code = '''
            # 前五行为同一子块
            X ->
                matmul W1 add B1 -> relu ->
                matmul W2 add B2 -> relu ->
                matmul W3 add B3 -> relu ->
            sigmoid as u
            # 最后一行为一子块
            u cross_entropy y -> J:$$
            '''
            ```

    8. 注释
        * 以`#`开始的行是注释

    9. 引用
        
        代码会从上到下从左到右依次扫描，其中第一次声明的变量或操作符将会被新建，而后续找到的所有同名变量和操作符会被视为第一个变量的引用，指向同一个节点
        
        根据这个特性，可以创建`X mul X`的代码，使用乘法代替平方

        目前为止，代码没有作用域和子域的功能，所有变量和操作符在同一计算图中平等存在，任何引用都会全局搜索计算图节点
    """
    # 可用的操作符
    __operators = {
        'add' : add,
        'sub' : sub,
        'mul' : mul,
        'div' : div,
        'sum' : sum,
        'matmul': matmul,
        'relu' : relu,
        'sigmoid' : sigmoid,
        'L1' : L1,
        'L2' : L2,
        'softmax' : softmax,
        'tanh' : tanh,
        'dropout' : dropout,
        'batchnorm' : batchnorm,
        'batchnorm2d' : batchnorm2d,
        'conv2d' : conv2d_v2,
        'conv2d_same' : conv2d_same,
        'conv2d_v3' : conv2d_v3,
        'conv2d_v3_same' : conv2d_v3_same,
        'max_pool' : max_pool,
        'max_pool_v3' : max_pool_v3,
        'flatten' : flatten,
        'L2_regularization': L2_regularization,
        'scatter' : scatter,
        'gather' : gather,
        'leaky_relu' : leaky_relu,
        'reshape' : reshape,

        'cross_entropy' : cross_entropy,
        'softmax_cross_entropy' : softmax_cross_entropy,
        'cost' : cost,

        'accuracy' : accuracy,
        'threshold' : threshold,
        'max_index' : max_index,
        'choise_to_index' : choise_to_index,
        'yolo1_loss' : yolo1_loss,
    }

    # 所有语句关键词
    __keywords = [
        'as',
        '=>',
        'to',
        'then',
        '->',
        'loop',
        'rest',
        '$$'
    ]

    # 循环关键词
    __loop_keywords = [
        'loop',
        '@@'
    ]

    # 赋值关键词
    __assign_keywords = [
        '=>',
        'to'
    ]

    # 命名关键词
    __named_keywords = [
        'as'
    ]

    # 连接关键词
    __link_keywords = [
        'then',
        '->'
    ]

    # 结束符关键词
    __rest_keywords = [
        'rest',
        '$$'
    ]

    # 可用的初始化函数
    __init_func = {
        '' : (lambda x : np.zeros(tuple(x))),
        'randn' : (lambda x : np.random.randn(*x)),
        'ones' : (lambda x : np.ones(tuple(x))),
        'zeros' : (lambda x : np.zeros(tuple(x))),
        'he' : (lambda x : np.random.randn(*x[0]) * np.sqrt(2. / x[1])),
        'xavier_cnn' : (lambda x : np.random.randn(*x) * 2. / np.sqrt(x[0] * x[1] * x[2] + x[0] * x[1] * x[3])),
        'xavier' : (lambda x : np.random.randn(*x) * 2. / np.sqrt(np.sum(list(x)))),
    }

    # 可用的更新函数
    __update_func = {
        'norm' : upf.norm,
        'momentum' : upf.momentum,
        'adam' : upf.adam,
    }

    # 输入语句
    __code : str = ''

    # 计算图
    __g : graph = None

    @property
    def code(self) :
        return self.__code
    
    @code.setter
    def code(self, value) :
        self.__code = value

    @property
    def g (self) :
        return self.__g

    def __init__ (self, code = '') :
        self.__code = code

    # 去掉一层小括号
    def _stripBrackets(self, el : str) -> str:
        if el.startswith('(') and el.endswith(')') :
            el = el[1:-1]
        return el

    # 将任意格式的多层数组拉平
    def _flatten (self, lst, save =[]):
        if type(lst) is list :
            for x in lst:
                if type(x) is list:
                    self._flatten(x, save)
                else:
                    save.append(x)
        else :
            save.append(lst)
        return save
    
    # 获取嵌套列表中的第一个值
    def _getListFirst (self, lst):
        save = []
        _temp = []
        if type(lst) is list :
            for x in lst:
                if type(x) is list :
                    self._flatten(x, _temp)
                    save.append(_temp[0])
                    _temp = []
                else :
                    save.append(x)
        else :
            save.append(lst)
        
        return save
    
    # 获取嵌套列表中的最后一个
    def _getListLast(self, lst) :
        save = []
        _temp = []
        if type(lst) is list :
            for x in lst :
                if type(x) is list :
                    self._flatten(x, _temp)
                    save.append(_temp[-1])
                    _temp = []
                else :
                    save.append(x)
        else :
            save.append(lst)
        
        return save

    # 将节点插入图
    def _insert(self, node, g:graph) :
        # 旧的方法把深度列表拉平成一个列表
        _nodes = []
        _nodes = self._flatten(node, _nodes)
        # 新的方法只处理列表当前深度的第一个值
        # _nodes = self._getListFirst(node)


        if not isinstance(_nodes, list) :
            _nodes = [_nodes]

        for n in _nodes :
            if not g.hasNode(n) :
                g.insertNode(n)

        return _nodes

    # 连接两个节点，无论节点是单个节点还是节点数组
    # 如果节点不在图里，则插入节点
    def _link(self, left, right, g:graph) :
        leftArr = self._insert(left, g)
        rightArr = self._insert(right, g)
        
        # 一个或多个左节点和一个或多个右节点进行交叉映射，类似于全连接
        for l in leftArr :
            for r in rightArr :
                g.addEdge(l, r)
    
    # 从代码块中获得下一个元素
    # str : str 代码块
    # el : str 返回获得的元素字符串
    #   * 关键词
    #   * 变量
    #   * 操作符
    #   * 子块
    #   * 等
    # rest_str : str 返回去掉元素后的代码子串
    # 支持Unicode
    def _getNextEl(self, str: str) :
        # 在遍历字符的过程中暂存括号
        stacks = []
        str = str.strip()
        i = 0
        # 上一个字符
        sprev = ''
        for s in str :
            if s in " \n\t\v\f" :
                if 0 == len(stacks) :
                    # 如果遇到空格且前面没有括号则元素到此为止
                    el = str[0:i]
                    rest_str = str[i+1:]
                    break
            # 如果有括号则进栈
            if s in "[({<":
                stacks.append(s)
            # 如果遇到另一半括号，且括号不是操作符的组成部分（->和=>），则出栈，验证括号是否匹配，不匹配则抛出异常
            if s in "])}>" and sprev not in '-=' :
                p = stacks.pop()
                if ('[' == p and ']' != s) or ('(' == p and ')' != s) or ('{' == p and '}' != s or ('<' == p and '>' != s)) :
                    hope = {'[' : ']', '{' : '}', '(' : ')', '<' : '>'}[p]
                    raise ParseError('Parsing code error, can not find "%s" to match "%s".' % (hope, p))
            sprev = s
            i += 1
        # 如果整体元素是一体的则一整块都是元素
        else :
            el = str.strip()
            rest_str = ""
            # 只去掉一层括号
            if el.startswith('(') and el.endswith(')') :
                el = el[1:-1]

        
        # 如果所有字符处理完成，栈里依然有括号，则说明表达式尾部缺少对应结束括号
        if 0 != len(stacks) :
            p = stacks.pop()
            hope = {'[' : ']', '{' : '}', '(' : ')'}[p]
            raise ParseError('Parsing code error, can not find "%s" to match "%s".' % (hope, p))
        
        el = el.strip()
        rest_str = rest_str.strip()

        # 去掉外层多余小括号
        # el = self._stripBrackets(el)
        
        return el, rest_str


    # 是否是子块
    # el元素字符串
    # 子块由不止一个元素组成，所以使用getNextEl判断
    def _isBlock(self, el : str) -> bool:
        el = el.strip()

        # 被小括号包裹的一整块元素是子块
        if el.startswith('(') and el.endswith(')') and self._getNextEl(el)[1] == '':
            return True
        # 没有被小括号包裹却由多个元素组成的是子块
        return self._getNextEl(el)[1] != ''
    
    
    # 是否是操作符
    def _isOperator(self, el : str) -> bool :
        # 先判断是否是关键词
        if el in self.__keywords :
            return False

        mp = "|".join(self.__operators.keys())
        # re.compile(r"^(add|sub|div|mul){0,1}(\((\s*\d+\s*,\s*)*(\s*\d+\s*){1}\))$", re.U)
        # mc = re.compile(r"^(" + mp + r"){0,1}(\((\s*\d+\s*,\s*)*(\s*\d+\s*){1}\))$", re.U)
        mc = re.compile(r"^(" + mp + r"){1}(\((\s*[0-9\.e\(\)\[\],\s\-]+?\s*,\s*)*(\s*[0-9\.e\(\)\[\],\s\-]+?\s*){0,1}\)){0,1}$", re.U)

        return mc.search(el) is not None
    

    # 创建操作符节点
    def _makeOperator(self, el:str) -> node:
        if not self._isOperator(el) :
            raise ParseError('"%s" is a illegal operator.' % (el))
        
        mp = "|".join(self.__operators.keys())
        mc = re.compile(r"^(" + mp + r"){1}(\((\s*[0-9\.e\(\)\[\],\s\-]+?\s*,\s*)*(\s*[0-9\.e\(\)\[\],\s\-]+?\s*){0,1}\)){0,1}$", re.U)
        mg = mc.search(el)
        # mg[1] 为操作符名
        # mg[2] 为初始化参数，可能为空
        f = mg[1]
        arg_str = mg[2]
        if arg_str is None :
            return node(self.__operators[f](), code = el)
        else :
            arg_str = self._stripBrackets(mg[2])
            args = self._makeArguments(arg_str)
            return node(self.__operators[f](*args), code = el)

    

    # 是否是合法引用
    # el : str 元素字符
    # g : graph 当前运算图
    def _isReference(self, el:str, g:graph) -> bool :
        if el in self.__keywords :
            return False
        
        if el in self.__operators.keys() :
            return False

        name = el.split(":")
        
        # 在计算图所有节点中寻找同名节点
        for n in g.nodes :
            if n.data.name == name[0].strip() :
                return True
        else :
            return False
    

    # 获得引用节点
    # el : str 元素字符
    # g : graph 当前运算图
    def _getReference(self, el:str, g:graph) -> node :
        if not self._isReference(el, g) :
            raise ParseError('Can not find node named "%s" in computation graph.' % (el))

        name = el.split(":")

        for n in g.nodes :
            if n.data.name == name[0].strip() :
                return n
    

    # 是否是合法初始化表达式
    # 合法的初始化表达式是init_func( 5, 4, 33 )
    def _isInitFunc(self, el:str) -> bool :
        mp = "|".join(self.__init_func.keys())
        # re.compile(r"^(add|sub|div|mul){0,1}(\((\s*\d+\s*,\s*)*(\s*\d+\s*){1}\))$", re.U)
        # mc = re.compile(r"^(" + mp + r"){0,1}(\((\s*\d+\s*,\s*)*(\s*\d+\s*){1}\))$", re.U)
        mc = re.compile(r"^(" + mp + r"){0,1}(\((\s*[0-9\.e\(\)\[\],\s\-]+?\s*,\s*)*(\s*[0-9\.e\(\)\[\],\s\-]+?\s*){0,1}\))$", re.U)

        return mc.search(el) is not None


    # 是否是合法名称
    def _isName(self, el:str) -> bool :
        # 不能是关键词
        if el in self.__keywords :
            return False
        
        # 不能是操作符
        if el in self.__operators.keys() :
            return False

        # 不能以数字开头
        if el.startswith(('0', '1', '2', '3', '4', '5', '6', '8', '9')) :
            return False
        
        # 只允许出现文字、字母、下划线和数字
        # 允许为空
        km = re.compile(r'^[\w_\d]*$', re.U)
        if km.search(el) is None :
            return False
        
        # 不能是初始化表达式
        if self._isInitFunc(el) :
            return False
        
        return True

    
    # 是否是合法标量或张量
    # ------ 错误的实现方式
    def _isTensor(self, el:str) -> bool :
        try :
            np.array(eval(el))
        except :
            return False

        return True
    

    # 是否是合法数字
    def _isNumber(self, el:str) -> bool :
        mc = re.compile(r"^[\-]?\d*[\.]?\d+$", re.U)
        return mc.match(el) is not None

    

    # 是否是合法值，包括标量、张量和初始化函数
    def _isValue(self, el:str) -> bool :
        if el in self.__keywords :
            return False
        
        if el in self.__operators.keys() :
            return False
        
        if self._isInitFunc(el) :
            return True

        if self._isNumber(el) :
            return True
        
        if self._isTensor(el) :
            return True
        
        if el == '' :
            return True
        
        return False


    def _isUpdateFunc(self, el:str) -> bool :
        mp = "|".join(self.__update_func.keys())
        mc = re.compile(r"^(" + mp + r"){0,1}(\((\s*[0-9\.e\(\)\[\],\s\-]+?\s*,\s*)*(\s*[0-9\.e\(\)\[\],\s\-]+?\s*){0,1}\))$", re.U)

        return mc.search(el) is not None
    

    # 创建合法参数数组，用于变量的初始化方法
    # 只允许标量、数组（包括多维数组）、元组三种类型的参数，其他类型比如字符串类型的参数为非法参数
    def _makeArguments(self, el:str) -> any :
        args = []

        if el == '' :
            return args

        p = ''
        stack = []
        for s in el :
            if s == ',' :
                if len(stack) > 0 :
                    p += s
                else :
                    if p.strip() == '' :
                        raise ParseError('Parsing code error, can not parse "%s" as legal arguments.' %(el))
                    args.append(eval(p))
                    p = ''
            elif s in '[(':
                p += s
                stack.append(s)
            elif s in '])':
                b = stack.pop()
                if (b == '[' and s == ']') or (b == '(' and s == ')') :
                    p += s
                else :
                    raise ParseError('Parsing code error, can not parse "%s" as legal arguments.' %(el))
            else :
                p += s
        
        if len(stack) != 0 :
            raise ParseError('Parsing code error, can not parse "%s" as legal arguments.' %(el))
        
        if p.strip() == '' :
            raise ParseError('Parsing code error, can not parse "%s" as legal arguments.' %(el))

        args.append(eval(p))
        
        return args

            


    # 创建合法值，依据元素类型返回张量或标量，如果元素是初始化函数，则执行此函数
    def _makeValue(self, el:str) -> any :
        if self._isInitFunc(el) :
            mp = "|".join(self.__init_func.keys())
            # mc = re.compile(r"^(" + mp + r"){1}(\((\s*\d+\s*,\s*)*(\s*\d+\s*){1}\))$", re.U)
            mc = re.compile(r"^(" + mp + r"){0,1}(\((\s*[0-9\.e\(\)\[\],\s-]+?\s*,\s*)*(\s*[0-9\.e\(\)\[\],\s-]+?\s*){0,1}\))$", re.U)
            mg= mc.search(el)
            # mg[1] 为方法名
            # mg[2] 为初始化参数
            f = mg[1]
            # tu = tuple( int(i) for i in self._stripBrackets( mg[2].rstrip() ).split(',') )
            # 可能会出问题的地方
            arg_str = self._stripBrackets(mg[2])
            args = self._makeArguments(arg_str)
            return self.__init_func[f](args)
        
        if self._isNumber(el) :
            if el.find('.') == -1 :
                return int(el)
            else :
                return float(el)
        
        if self._isTensor(el) :
            return np.array(eval(el))
        
        if el == '' :
            return None

        raise ParseError('"%s" is a illegal form of value.' % (el))


    def _makeUpdateFunc(self, el:str) -> any :
        mp = "|".join(self.__update_func.keys())
        mc = re.compile(r"^(" + mp + r"){0,1}(\((\s*[0-9\.e\(\)\[\],\s-]+?\s*,\s*)*(\s*[0-9\.e\(\)\[\],\s-]+?\s*){0,1}\))$", re.U)
        mg= mc.search(el)
        # mg[1] 为方法名
        # mg[2] 为初始化参数
        f = mg[1]
        arg_str = self._stripBrackets(mg[2])
        args = self._makeArguments(arg_str)
        return self.__update_func[f](*args)
    
    # 是否是合法变量
    def _isVariable(self, el:str) -> bool:
        
        sp_block = el.split(":")

        if len(sp_block) == 1 :
            if self._isName(sp_block[0]) or self._isValue(sp_block[0]) or self._isUpdateFunc(sp_block[0]) :
                return True
            else :
                return False
        elif len(sp_block) == 2 :
            if self._isName(sp_block[0]) and self._isValue(sp_block[1]) or self._isName(sp_block[0]) and self._isUpdateFunc(sp_block[1]) or self._isValue(sp_block[0]) and self._isUpdateFunc(sp_block[1]) :
                return True
            else :
                return False
        elif len(sp_block) == 3 :
            if self._isName(sp_block[0]) and self._isValue(sp_block[1]) and self._isUpdateFunc(sp_block[2]) :
                return True
            else :
                return False
        
        return False
        

    # 创建变量节点
    # el : str 变量表达式
    # 1. X或X: 名称X，无值
    # 2. 5或:5 无名称，值为5
    # 2. X:5 名称X，值为5
    # 3. X:(1) 名称X，类型为(1)张量
    # 4. X:(10, 20) 名称X，类型为10x20矩阵
    # 5. X:[[1, 2], [3, 4]] 名称X，类型为2x2矩阵，值为[[1, 2], [3, 4]]
    # 6. X:randn(5, 10) 名称X，类型为5x10矩阵，值为rand函数产生的随机数张量
    # 7. X:zeros(5, 10) 名称X，类型为5x10矩阵，值为zeros函数产生的全零张量
    # 8. X:ones(5, 10) 名称X，类型为5x10矩阵，值为ones函数产生的全一张量
    def _makeVariable(self, el:str) -> node :
        sp_block = el.split(":")

        name = None
        value = None
        updateFunc = None
        
        # 如果没有分号，则匹配名称或值
        if len(sp_block) == 1:
            if self._isName(sp_block[0]) :
                name = sp_block[0]
            elif self._isValue(sp_block[0]) :
                value = self._makeValue(sp_block[0])
            elif self._isUpdateFunc(sp_block[0]) :
                updateFunc = self._makeUpdateFunc(sp_block[0])
            else :
                raise ParseError('Can not create variable from "%s".' % (el))
        # 如果有分号，则分别匹配名称和值
        elif len(sp_block) == 2 :
            if self._isName(sp_block[0]) and self._isValue(sp_block[1]) :
                name = sp_block[0]
                value = self._makeValue(sp_block[1])
            elif self._isName(sp_block[0]) and self._isUpdateFunc(sp_block[1]) :
                name = sp_block[0]
                updateFunc = self._makeUpdateFunc(sp_block[1])
            elif self._isValue(sp_block[0]) and self._isUpdateFunc(sp_block[1]) :
                value = self._makeValue(sp_block[0])
                updateFunc = self._makeUpdateFunc(sp_block[1])
            else :
                raise ParseError('Can not create variable from "%s".' % (el))
        elif len(sp_block) == 3 :
            if self._isName(sp_block[0]) and self._isValue(sp_block[1]) and self._isUpdateFunc(sp_block[2]) :
                name = sp_block[0]
                value = self._makeValue(sp_block[1])
                updateFunc = self._makeUpdateFunc(sp_block[2])
            else :
                raise ParseError('Can not create variable from "%s".' % (el))
        else :
            raise ParseError('Can not create variable from "%s".' % (el))

        
        n = node(var(value, updateFunc), code = el)
        self._setOperatorName(n, name)

        return n


    # 是否是合法变量组，包含未创建的变量和已创建的命名引用
    def _isVariableArray(self, el:str, g:graph) -> bool :
        ell = el.split('|')

        if len(ell) <= 1 :
            return False
        
        for i in ell :
            if i == '' :
                continue
            if  not self._isReference(i, g) and not self._isVariable(i) :
                return False
        
        return True
    
    # 返回节点数组
    # 已有的同名节点返回引用，没有的创建
    def _makeVariableArray(self, el:str, g:graph) -> list :
        ell = el.split('|')

        vlist = []
        for i in ell :
            if i == '' :
                continue

            if self._isReference(i, g) :
                vlist.append(self._getReference(i, g))
            
            elif self._isVariable(i) :
                vlist.append(self._makeVariable(i))

        return vlist

     

    # 给操作符命名
    def _setOperatorName(self, n:node, el:str) :
        n.data.name = el
    

    # 是否是休止符
    def _isRest(self, el:str) -> bool:
        sp_idx = el.find(":")

        # 匿名休止符
        if -1 == sp_idx :
            if el in self.__rest_keywords :
                return True
            else :
                return False
        # 命名休止符
        else :
            name_str = el[:sp_idx]
            value_str = el[sp_idx + 1:]
            if self._isName(name_str) and (value_str in self.__rest_keywords) :
                return True
            else :
                return False
            
    
    # 创建休止符（结果节点）
    def _makeRest(self, el:str) -> node :
        sp_idx = el.find(":")
        name = None

        if -1 == sp_idx :
            if el not in self.__rest_keywords :
                raise ParseError('"%s" is a illegal operator of rest.' % (el))
        
        else :
            name_str = el[:sp_idx]
            value_str = el[sp_idx + 1:]
            if self._isName(name_str) and (value_str in self.__rest_keywords) :
                if name_str == '' :
                    name = None
                else :
                    name = name_str
            else :
                raise ParseError('"%s" is a illegal operator of rest.' % (el))
        
        n = node(rest(), code = el)
        self._setOperatorName(n, name)

        return n



    # 处理块逻辑，并返回解析块的计算图的汇点（终点）
    # line : str 代码子串
    # g: 计算图
    # 返回源点与汇点的数组
    def _processBlock(self, line:str, g:graph) :
        # 左操作数
        left : node = None
        # 右操作数
        right : node = None
        # 操作符
        opt : node = None

        # 源点
        first : node = None
        # 汇点
        last : node = None

        rest_str = line
        el, rest_str = self._getNextEl(rest_str)
        
        # 对于每一层（由then或->分隔的部分），可能的结构由以下几种
        # 1. X opt Y 此时左右操作数与操作符均显式给出
        # 2. opt Y 此时左操作数为上一层隐式给出
        # 3. opt 此为一元操作符，左操作数由上一层给出
        # 4. X opt1 Y opt2 Z 此时构建树状结构图
        # 5. (X opt1 Y) opt2 Z 此时子块递归
        # 6. X1|X2|X3 opt X4 此为多元操作符，操作符需要两个以上（不包括两个）时需要使用"|"分割操作数
        while el :
            # 如果元素是子块，满足结构5
            if self._isBlock(el) :
                el = el.strip()
                # 处理子块时，需要去掉外层小括号
                # 如果块为整块，则去掉小括号，否则视为多个块的拼接，不去掉小括号
                _b, _rst = self._getNextBlock(el)
                if _rst == '' :
                    el = self._stripBrackets(el)
                # 递归调用块处理逻辑来处理子块
                # 子块位于首部，则此块源点是子块的源点
                if left is None and first is None:
                    first, left = self._processSection(el, g)
                
                # 左操作数和子块并列，中间没有操作符
                # 考虑`X scatter -> (x0 matmul W1)`，scatter会在->后变为左操作数，然后将值传递给x0，即如果两个变量或操作符与变量相连，那么后者将会接收并存储前者的值
                # 源点与前节点相连
                # 返回汇点
                elif opt is None :
                    opt, _ = self._processSection(el, g)
                    if left is not None :
                        self._link(left, opt, g)
                    # 汇点为每一个子块的最后一个节点
                    # todo : change _ as last node of each block
                    left = _
                    opt = None
                    right = None

                # ... add (Y1 matmul Z1)
                # 右操作数，则使用汇点
                elif right is None :
                    _, right = self._processSection(el, g)

                # 右操作数和子块并列，中间没有操作符
                else :
                    raise ParseError('Can not arrange two of both variable (%s, %s) and block in one place.' % (right.code, el))

            # 判断是否是操作符
            elif self._isOperator(el) :
                # 操作符不能在整个表达式首部
                # v0.3 操作符可以在子块首部
                # X -> ( matmul W1 ) -> 
                if left is None and first is None:
                    opt = self._makeOperator(el)
                    first = opt
                    # raise ParseError('Operator "%s" after null variable is mistake.' % (el) )
                
                elif opt is None :
                    opt = self._makeOperator(el)
                
                # 两个操作符并排出现
                elif right is None :
                    raise ParseError('Can not arrange two operators (%s, %s) in one place.' % (opt.code, el))

                # 如果左操作数、右操作数和操作符同时存在，此时又出现一个新的操作符，则满足结构4
                else : 
                    self._insert(opt, g)

                    if left is not None :
                        self._link(left, opt, g)

                    if right is not None :
                        self._link(right, opt, g)
                    # 操作符成为下一层的左操作数
                    left = opt
                    opt = self._makeOperator(el)
                    right = None
                
            # 判断是否是合法引用
            # 引用通常来自于已经命名的操作符（通过关键字as）或者已经命名的变量
            # 变量通常在第一次发现时创建，后续使用则为同名引用
            elif self._isReference(el, g) :
                if left is None and first is None:
                    left = self._getReference(el, g)
                    first = left
                # 引用只能为操作数
                elif opt is None :
                    raise ParseError('Can not arrange two variables (%s, %s) in one place.' % (left.code, el))
                elif right is None :
                    right = self._getReference(el, g)

            # 判断是否是合法变量组
            # 如果是变量组，则满足表达式6
            elif self._isVariableArray(el, g) :
                if left is None and first is None:
                    left = self._makeVariableArray(el, g)
                    first = left
                    # 将所有左操作数加入图
                    self._insert(left, g)
                
                elif opt is None :
                    raise ParseError('Can not arrange two variables (%s, %s) in one place.' % (left.code, el))
                
                elif right is None :
                    right = self._makeVariableArray(el)

                else :
                    raise ParseError('Can not arrange two variables (%s, %s) in one place.' % (right.code, el))
            
            # 判断是否是合法变量
            elif self._isVariable(el) :
                if left is None and first is None:
                    left = self._makeVariable(el)
                    first = left
                    # 将首个左操作数加入图，后续的所有左操作数都是上一层操作符，是已经加入图的节点
                    self._insert(left, g)
                # 两个变量并排出现
                elif opt is None :
                    raise ParseError('Can not arrange two variables (%s, %s) in one place.' % (left.code, el))
                elif right is None :
                    right = self._makeVariable(el)
                else :
                    raise ParseError('Can not arrange two variables (%s, %s) in one place.' % (right.code, el))
            
            # 赋值关键词
            # 将之前的表达式的值保存在一个变量里
            elif el in self.__assign_keywords :
                oel = el
                el, rest_str = self._getNextEl(rest_str)

                if not self._isVariable(el):
                    raise ParseError('Only variable can be assigned by "%s".' % (oel))
                
                v = None
                if self._isReference(el, g) :
                    v = self._getReference(el, g)
                else :
                    v = self._makeVariable(el)
                
                # v0.3赋值操作符可以出现在子块首部
                if left is None and first is None:
                    left = v
                    first = left
                    # raise ParseError('Keyword "%s" before all of statements is mistake.' % (el))
                elif opt is None :
                    opt = v
                    if left is not None :
                        self._link(left, opt, g)
                    
                    left = opt
                    opt = None
                    right = None
                
                elif right is None :
                    if left is not None :
                        self._link(left, opt, g)
                    self._link(opt, v, g)
                    left = v
                    opt = None
                    right = None
                else :
                    if left is not None :
                        self._link(left, opt, g)
                    if right is not None :
                        self._link(right, opt, g)
                    self._link(opt, v, g)
                    left = v
                    opt = None
                    right = None



            
            # 对操作符命名
            elif el in self.__named_keywords :
                oel = el
                # 获取下一个元素
                el, rest_str = self._getNextEl(rest_str)

                if opt is None :
                    raise ParseError('Only operator can be named by "%s".' % (oel))
                
                # 非法的命名
                if not self._isName(el) :
                    raise ParseError('"%s" is a illegal name.' % (el))
                
                # 给操作符命名
                self._setOperatorName(opt, el)
                # opt.data.name = el
            

            # 如果遇到loop语句，则交由loop处理程序循环展开源码，而不是按某种语法处理，随后将处理好的源码重新投入块解析器
            elif el in self.__loop_keywords :
                rest_str = self._processLoop(el + ' ' + rest_str)


            # 层连接符
            elif el in self.__link_keywords :
                # 层连接符不能放在表达式首部
                if left is None and first is None:
                    raise ParseError('Keyword "%s" before all of statements is mistake.' % (el))
                
                # 操作符不为空，则构建此层，此层的操作符成为下一层的左操作数
                elif opt is not None :
                    # g.insertNode(opt)
                    self._insert(opt, g)
                    # 如果是多个左操作数，则全部连接到操作符

                    if left is not None :
                        self._link(left, opt, g)

                    if right is not None:
                        # 如果是多个右操作数，则全部连接到操作符
                        self._link(right, opt, g)

                    left = opt
                    opt = None
                    right = None
            
            # 终止符（结果变量）
            elif self._isRest(el) :
                # 休止符（结果变量）不能放在表达式首部
                if left is None and first is None:
                    raise ParseError('REST arranged before all of statements is mistake.')
                
                else :
                    opt = self._makeRest(el)


            # 处理下一个元素
            el, rest_str = self._getNextEl(rest_str)
        

        # 对还未加入图的节点进行收尾，并返回汇点
        if opt is not None :
            if left is not None :
                self._link(left, opt, g)

            if right is not None :
                self._link(right, opt, g)

            last = opt
        else :
            last = left
        
        return first, last
    
    # 获取下一个块
    def _getNextBlock(self, str:str) :
        stacks = []
        str = str.strip()
        i = 0
        # 注释标记
        comment = 0
        # 要返回的块
        block = ""
        for s in str :
            # 如果发现注释标记，则不记录后面的所有字符，直到遇到'\n'
            if s in "#" :
                comment = 1
                continue
            # 默认的块分割符为换行
            elif s in "\n" :
                if comment == 1:
                    comment = 0
                    continue
                # 如果换行时处于括号内，则视为子块语句，不处理，否则当做块分割
                if 0 == len(stacks) :
                    # 如果行以层连接符结束，则表示此块未结束
                    if block.strip().endswith(tuple(self.__link_keywords)) :
                        block = block + ' '
                        i += 1
                        continue
                    else :
                        rest_str = str[i+1:]
                        break
            elif s in "(" and comment == 0 :
                stacks.append(s)
            elif s in ")" and comment == 0 :
                p = stacks.pop()
                if '(' == p and ')' != s :
                    raise ParseError('Parsing code error, can not find ")" to match "(".')
            if comment == 0 :
                block = block + s

            i += 1
        # 表示传的全部字符串是一个块
        else :
            block = block.strip()
            rest_str = ""
            # 只去掉一层括号
            if block.startswith('(') and block.endswith(')') :
                block = block[1:-1]
        
        # 如果所有字符处理完成，栈里依然有括号，则说明表达式尾部缺少对应结束括号
        if 0 != len(stacks) :
            raise ParseError('Parsing code error, can not find ")" to match "(".')
        
        block = block.strip()
        rest_str = rest_str.strip()
        
        return block, rest_str
        
    # 处理代码段落
    # 1. 如果换行不在括号里，则将换行符作为块的分割依据
    # 2. 如果换行在括号里，则当成一整块不处理
    # 3. 处理空行
    # 4. 处理注释
    # 5. 块内处理交给_processBlock
    def _processSection(self, section: str, g: graph) :
        # 代码段生成的源点
        sources = []
        # 代码段生成的汇点
        focus = []

        rest_str = section
        block, rest_str = self._getNextBlock(rest_str)

        while block:
            # 如果block已经是一个无法分割的块，则调用块处理
            _b, _rst = self._getNextBlock(block)
            if _rst == '' :
                sou, foc = self._processBlock(block, g)
            # 否则继续调用句子处理
            else :
                sou, foc = self._processSection(block, g)

            sources.append(sou)
            focus.append(foc)

            block, rest_str = self._getNextBlock(rest_str)
        

        # todo 收集完所有子图的源点和汇点后，对每个源点和汇点进行二次检查，后续处理的子图中对先前处理的子图中的点进行连接，导致先前子图中的汇点不再是汇点，找出这些点并排除
        # 源为列表中的第一个元素，无论这个元素嵌套有多深
        # 汇为列表中的最后一个元素，无论这个元素嵌套有多深
        sources = self._getListFirst(sources)
        focus = self._getListLast(focus)

        return sources, focus


    def _processLoop(self, code : str) -> str:
        code = code.strip()
        mp = "|".join(self.__loop_keywords)
        mc = re.compile(r"^(" + mp + r"){1}\s+([a-zA-Z0-9_]+){1}\s*=\s*([0-9]+){1}\s+to\s+([0-9]+){1}")
        mg = mc.search(code)

        if not mg or mg[1] not in self.__loop_keywords:
            return code
        
        # 从循环语句中提取循环关键词，循环变量，循环起始值，循环结束值
        # loop t = 1 to 5
        keyword, variable, idx_from, idx_to = mg[1], mg[2], int(mg[3]), int(mg[4])
        rest_str = code[len(mg[0]):]
        loop_range = list(range(idx_from, idx_to + 1))

        el, rest_str = self._getNextEl(rest_str)
        if not self._isBlock(el) :
            raise ParseError('Looping keyword "%s" must be set before a block.' % (keyword) )
        
        # 剥离括号
        if el.startswith('(') and el.endswith(')') :
            el = el[1:-1]
        new_el = ""

        # 遍历循环变量，计算并替换源码中所有尖括号<>中的值
        for i in loop_range :
            cp_el = self._replaceAndTryCompute(el, variable, i)
            new_el +='(' + cp_el + ")\n"
        
        new_el = "(" + new_el + ")"
        # print(new_el + ' ' + rest_str.strip())
        return new_el + ' ' + rest_str.strip()

    # 将由尖括号`<>`中包裹的代码中的变量`name`替换成对应的值`value`
    # 如果此尖括号`<>`内的表达式中所有的变量都已经被替换，则尝试计算表达式，用结果替换掉尖括号中的所有内容，并去掉尖括号
    # `X<a + b + 1>`，当a = 1时，替换为`X<1 + b + 1>`
    # 当a = 1，b = 3时，替换为`X5`
    def _replaceAndTryCompute(self, code, vname, vvalue) :
        start = -1
        end = -1
        rvalue = ''
        replaceAllMatch = []
        for i in range(len(code)) :
            if code[i] in '<' :
                if start == -1:
                    start = i
                else :
                    raise Exception('暂不支持嵌套')
            elif code[i] in '>' :
                if start == -1 :
                    # 没有开始括号，只有结束括号，表示这是一个其他运算符，而不是表达式分隔符
                    continue
                if end == -1 :
                    end = i + 1
                else :
                    raise Exception('暂不支持嵌套')
                rvalue = self._computeExpr(code[start:end], vname, vvalue)
                # 未防止替换和遍历同时进行出错，将需要替换的值暂存，等完成遍历后再进行替换
                replaceAllMatch.append([start, end, rvalue])

                start = -1
                end = -1
            
        code_arr = list(code)
        # 将表达式从后向前依次替换（通过数组替换元素的方式）
        for repObj in reversed(replaceAllMatch) :
            code_arr[repObj[0]:repObj[1]] = list(str(repObj[2]))
        
        return ''.join(code_arr)
    
    def _computeExpr(self, expr, src, des) -> str:
        # 剥离尖括号，增加结束符
        expr = expr[1:-1]
        
        start = -1
        end = -1
        replaceAllMatch = []

        for i in range(len(expr)) :
            if expr[i] in '+-*/() \n\f\v\r\t' :
                if start != -1 and end == -1 :
                    end = i
                    if expr[start:end] == src :
                        replaceAllMatch.append([start, end])

                start = -1
                end = -1
            else :
                if start == -1 :
                    start = i
        
        if start != -1 and expr[start:] == src :
            replaceAllMatch.append([start, None])
        
        code_arr = list(expr)
        for repObj in reversed(replaceAllMatch) :
            code_arr[repObj[0]:repObj[1]] = list(str(des))

        end_code = ''.join(code_arr)

        # 如果表达式的所有变量都已经替换，则计算表达式并返回值
        try :
            return eval(end_code, {}, {})
        # 如果表达式无法执行，则表示其中有未经替换的变量，则返回
        except:
            return '<' + (''.join(code_arr)) + '>'
            
        
        
      
    # 解析代码
    def parse(self, code : str = None) :
        """
        解析代码

        ### 参数
        * code : str, 代码，可为空

        ### 返回值
        * g : ComputationGraph, 计算图对象
        """
        if code is not None :
            self.__code = code
        self.__g = graph()

        self._processSection(self.__code, self.__g)
        
        return self.g


    # 注册新的操作符，如果有同名操作符则覆盖
    def registerOperator (self, name : str, class_type) :
        self.__operators[name] = class_type
    
    # 注册新的初始化函数，如果有同名函数则覆盖
    def registerInitFunc (self, name : str, func ) :
        self.__init_func[name] = func
    
    # 注册新的参数更新函数，如果有同名函数则覆盖
    def registerUpdateFunc(self, name : str, func) :
        self.__update_func[name] = func

    
    # 导出nous数据
    def exports (self) :
        obj = {}
        if self.g is not None :
            obj['nodes'] = self.g.exports()

        if self.code is not None :
            obj['code'] = self.code
        
        return obj
    
    # 导入nous数据
    def imports (self, value) :
        if value['code'] :
            self.code = value['code']

        if value['nodes'] :
            self.parse()
            self.g.imports(value['nodes'])



