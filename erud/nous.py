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
from erud.tensor.var import var
from erud.tensor.rest import rest
import numpy as np
import re

from erud.opts_extend.accuracy import accuracy
from erud.opts_extend.threshold import threshold

# 解析代码，构造计算图
class nous :
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
        'cross_entropy' : cross_entropy,
        'cost' : cost,

        'accuracy' : accuracy,
        'threshold' : threshold,
    }

    # 所有语句关键词
    __keywords = [
        'as',
        'then',
        '->',
        'rest'
        '$$'
    ]

    # 层连接关键词
    __layer_keywords = [
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
        'he' : (lambda x : np.random.randn(*x[0]) * np.sqrt(2.0 / x[1]))
    }

    # 输入语句
    __code : str = ''

    # 计算图
    __g : graph

    @property
    def g (self) :
        return self.__g

    def __init__ (self, code = '') :
        self.__code = code

    # 去掉外层多余小括号
    def _stripBrackets(self, el : str) -> str:
        while el.startswith('(') and el.endswith(')') :
            el = el[1:-1]
        return el

    
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
        for s in str :
            if s in " \n\t\v\f" :
                if 0 == len(stacks) :
                    # 如果遇到空格且前面没有括号则元素到此为止
                    el = str[0:i]
                    rest_str = str[i+1:]
                    break
            # 如果有括号则进栈
            if s in "[({" :
                stacks.append(s)
            # 如果遇到另一半括号则出栈，验证括号是否匹配，不匹配则抛出异常
            if s in "])}" :
                p = stacks.pop()
                if ('[' == p and ']' != s) or ('(' == p and ')' != s) or ('{' == p and '}' != s) :
                    hope = {'[' : ']', '{' : '}', '(' : ')'}[p]
                    raise ParseError('Parsing code error, can not find "%s" to match "%s".' % (hope, p))
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
        
        if el in self.__operators.keys() :
            return True
        
        return False
    

    # 创建操作符节点
    def _makeOperator(self, el:str) -> node:
        if not self._isOperator(el) :
            raise ParseError('"%s" is a illegal operator.' % (el))
        
        return node(self.__operators[el](), code = el)
    

    # 是否是合法引用
    # el : str 元素字符
    # g : graph 当前运算图
    def _isReference(self, el:str, g:graph) -> bool :
        if el in self.__keywords :
            return False
        
        if el in self.__operators.keys() :
            return False
        
        # 在计算图所有节点中寻找同名节点
        for n in g.nodes :
            if n.data.name == el :
                return True
        else :
            return False
    

    # 获得引用节点
    # el : str 元素字符
    # g : graph 当前运算图
    def _getReference(self, el:str, g:graph) -> node :
        if not self._isReference(el, g) :
            raise ParseError('Can not find node named "%s" in computation graph.' % (el))
        
        for n in g.nodes :
            if n.data.name == el :
                return n
    

    # 是否是合法初始化表达式
    # 合法的初始化表达式是init_func( 5, 4, 33 )
    def _isInitFunc(self, el:str) -> bool :
        mp = "|".join(self.__init_func.keys())
        # re.compile(r"^(add|sub|div|mul){0,1}(\((\s*\d+\s*,\s*)*(\s*\d+\s*){1}\))$", re.U)
        # mc = re.compile(r"^(" + mp + r"){0,1}(\((\s*\d+\s*,\s*)*(\s*\d+\s*){1}\))$", re.U)
        mc = re.compile(r"^(" + mp + r"){0,1}(\((\s*[0-9\.e\(\)\[\],\s]+?\s*,\s*)*(\s*[0-9\.e\(\)\[\],\s]+?\s*){0,1}\))$", re.U)

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
    
    # 是否是合法变量
    def _isVariable(self, el:str) -> bool:
        sp_idx = el.find(":")

        if -1 == sp_idx :
            if self._isName(el):
                return True
            elif self._isValue(el):
                return True
            else :
                return False
        else :
            name_str = el[:sp_idx]
            value_str = el[sp_idx + 1:]
            if self._isName(name_str) and self._isValue(value_str) :
                return True
            else :
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
        sp_idx = el.find(":")

        name = None
        value = None
        
        # 如果没有分号，则匹配名称或值
        if -1 == sp_idx :
            if self._isName(el) :
                name = el
            elif self._isValue(el) :
                value = self._makeValue(el)
            else :
                raise ParseError('Can not create variable from "%s".' % (el))
        # 如果有分号，则分别匹配名称和值
        else :
            name_str = el[:sp_idx]
            value_str = el[sp_idx + 1:]
            if self._isName(name_str) :
                if name_str == '' :
                    name = None
                else :
                    name = name_str
            else :
                raise ParseError('Can not create variable from "%s".' % (el))

            if self._isValue(value_str) :
                value = self._makeValue(value_str)
            else :
                raise ParseError('Can not create variable from "%s".' % (el))
        
        n = node(var(value), code = el)
        self._setOperatorName(n, name)

        return n


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
    def _processBlock(self, line:str, g:graph) :
        # 左操作数
        left : node = None
        # 右操作数
        right : node = None
        # 操作符
        opt : node = None

        rest_str = line
        el, rest_str = self._getNextEl(rest_str)
        
        # 对于每一层（由then或->分隔的部分），可能的结构由以下几种
        # 1. X opt Y 此时左右操作数与操作符均显式给出
        # 2. opt Y 此时左操作数为上一层隐式给出
        # 3. opt 此为一元操作符，左操作数由上一层给出
        # 4. X opt1 Y opt2 Z 此时构建树状结构图
        # 5. (X opt1 Y) opt2 Z 此时子块递归
        while el :
            # 如果元素是子块，满足结构5
            if self._isBlock(el) :
                # 递归调用块处理逻辑来处理子块
                if left is None :
                    left = self._processBlock(el, g)
                
                # 左操作数和子块并列，中间没有操作符
                elif opt is None :
                    raise ParseError('Can not arrange two of both variable (%s, %s) and block in one place.' % (left.code, el))

                elif right is None :
                    right = self._processBlock(el, g)

                # 右操作数和子块并列，中间没有操作符
                else :
                    raise ParseError('Can not arrange two of both variable (%s, %s) and block in one place.' % (right.code, el))

            # 判断是否是操作符
            # elif el in self.__options.keys() :
            elif self._isOperator(el) :
                # 操作符不能在整个表达式首部
                if left is None :
                    raise ParseError('Operator "%s" after null variable is mistake.' % (el) )
                
                elif opt is None :
                    opt = self._makeOperator(el)
                    # opt = node(self.__options[el]())
                
                # 两个操作符并排出现
                elif right is None :
                    raise ParseError('Can not arrange two operators (%s, %s) in one place.' % (opt.code, el))

                # 如果左操作数、右操作数和操作符同时存在，此时又出现一个新的操作符，则满足结构4
                else : 
                    g.insertNode(opt)
                    if not g.hasNode(right) :
                        g.insertNode(right)
                    g.addEdge(left, opt)
                    g.addEdge(right, opt)
                    # 操作符成为下一层的左操作数
                    left = opt
                    opt = self._makeOperator(el)
                    right = None
                
            # 判断是否是合法引用
            # 引用通常来自于已经命名的操作符（通过关键字as）或者已经命名的变量
            # 变量通常在第一次发现时创建，后续使用则为同名引用
            elif self._isReference(el, g) :
                if left is None :
                    left = self._getReference(el, g)
                # 引用只能为操作数
                elif opt is None :
                    raise ParseError('Can not arrange two variables (%s, %s) in one place.' % (left.code, el))
                elif right is None :
                    right = self._getReference(el, g)

            
            # 判断是否是合法变量
            elif self._isVariable(el) :
                if left is None :
                    left = self._makeVariable(el)
                    # 将首个左操作数加入图，后续的所有左操作数都是上一层操作符，是已经加入图的节点
                    g.insertNode(left)
                # 两个变量并排出现
                elif opt is None :
                    raise ParseError('Can not arrange two variables (%s, %s) in one place.' % (left.code, el))
                elif right is None :
                    right = self._makeVariable(el)
                    # 操作符、操作数齐全，生成一层操作
                    # g.insertNode(opt)
                    # g.addEdge(left, opt)
                    # g.insertNode(right)
                    # g.addEdge(right, opt)
                    # # 操作符成为下一层的左操作数
                    # left = opt
                    # opt = None
                    # right = None
                else :
                    raise ParseError('Can not arrange two variables (%s, %s) in one place.' % (right.code, el))
            
            # 对操作符命名
            elif el == 'as' :
                # 获取下一个元素
                el, rest_str = self._getNextEl(rest_str)

                if opt is None :
                    raise ParseError('Only operator can be named by "as".')
                
                # 非法的命名
                if not self._isName(el) :
                    raise ParseError('"%s" is a illegal name.' % (el))
                
                # 给操作符命名
                self._setOperatorName(opt, el)
                # opt.data.name = el

            # 层连接符
            elif el in self.__layer_keywords :
                # 层连接符不能放在表达式首部
                if left is None :
                    raise ParseError('Liner "%s" before all of statements is mistake.' % (el))
                
                # 操作符不为空，则构建此层，此层的操作符成为下一层的左操作数
                elif opt is not None :
                    g.insertNode(opt)
                    g.addEdge(left, opt)
                    if right is not None:
                        if not g.hasNode(right) :
                            g.insertNode(right)
                        g.addEdge(right, opt)
                    left = opt
                    opt = None
                    right = None
            
            # 终止符（结果变量）
            elif self._isRest(el) :
                # 休止符（结果变量）不能放在表达式首部
                if left is None :
                    raise ParseError('REST arranged before all of statements is mistake.')
                
                else :
                    opt = self._makeRest(el)


            # 处理下一个元素
            el, rest_str = self._getNextEl(rest_str)
        

        # 对还未加入图的节点进行收尾，并返回汇点
        if opt is not None :
            g.insertNode(opt)
            g.addEdge(left, opt)
            if right is not None :
                if not g.hasNode(right) :
                    g.insertNode(right)
                g.addEdge(right, opt)
            left = opt
        
        return left
        

    # 解析代码
    def parse(self, code : str = None) :
        if code is not None :
            self.__code = code
        self.__g = graph()

        lines = self.__code.split("\n")

        block = ''
        for b in lines :
            b = b.strip()

            # 空行跳过
            if b == '' :
                continue

            # 以#开头的是注释
            if b.startswith('#') :
                continue

            block += ' ' + b

            # 以层连接符结尾的视为块内换行
            if not ( b.endswith(tuple(self.__layer_keywords)) ) :
                self._processBlock(block, self.__g)
                block = ''
        
        return self.g


    # 注册新的操作符，如果有同名操作符则覆盖
    def registerOperator (self, name : str, class_type) :
        self.__operators[name] = class_type
    
    # 注册新的初始化函数，如果有同名函数则覆盖
    def registerInitFunc (self, name : str, func ) :
        self.__init_func[name] = func

