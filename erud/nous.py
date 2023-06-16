from erud.cg.graph import ComputationGraph as graph
from erud.cg.node import ComputationNode as node
from erud.errors import ParseError
from erud.opts.add import add
from erud.opts.sub import sub
from erud.opts.mul import mul
from erud.opts.div import div
from erud.cg.payload import payload
import re

# 解析代码，构造计算图
class nous :
    # 可用的操作符
    __options = {
        'add' : add,
        'sub' : sub,
        'mul' : mul,
        'div' : div
    }
    # 所有语句关键词
    __keywords = [
        'as',
        'then'
        '->'
        'REST'
    ]
    # 输入语句
    __code : str = ''

    def __init__ (self, code = '') :
        self.__code = code
    
    # 获得下一个元素
    # 返回获得的元素对象
    #   * 关键词
    #   * 变量
    #   * 操作符
    #   * 子块
    #   * 等
    # 返回去掉元素后的代码子串
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
            el = str
            rest_str = ""
        
        # 如果所有字符处理完成，栈里依然有括号，则说明表达式尾部缺少对应结束括号
        if 0 != len(stacks) :
            p = stacks.pop()
            hope = {'[' : ']', '{' : '}', '(' : ')'}[p]
            raise ParseError('Parsing code error, can not find "%s" to match "%s".' % (hope, p))
        
        el.strip()
        rest_str.strip()
        
        # 去掉外层多余小括号
        while el.startswith('(') and el.endswith(')') :
            el = el[1:-1]
        
        return el, rest_str


    # 处理块逻辑，并返回解析块的计算图的汇点（终点）
    # str: str 代码子串
    # g: 计算图
    def _processBlock(self, str:str, g:graph) :
        # 左操作数
        left : node = None
        # 右操作数
        right : node = None
        # 操作符
        opt : node = None

        rest_str = str
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
                    raise ParseError('Can not arrange two of both variable and block in one place.')

                elif right is None :
                    right = self._processBlock(el, g)

                # 右操作数和子块并列，中间没有操作符
                else :
                    raise ParseError('Can not arrange two of both variable and block in one place.')

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
                    raise ParseError('Can not arrange two operators in one place.')

                # 如果左操作数、右操作数和操作符同时存在，此时又出现一个新的操作符，则满足结构4
                # else : 
                #     g.insertNode(left)
                #     g.insertNode(right)
                #     g.insertNode(opt)
                #     g.addEdge(left, opt)
                #     g.addEdge(right, opt)
                #     # 操作符成为下一层的左操作数
                #     left = opt
                #     opt = self.__options[el]()
                #     right = None
                
            # 判断是否是合法引用
            # 引用通常来自于已经命名的操作符（通过关键字as）或者已经命名的变量
            # 变量通常在第一次发现时创建，后续使用则为同名引用
            elif self._isReference(el, g) :
                if left is None :
                    left = self._getReference(el, g)
                elif opt is None :
                    opt = self._getReference(el, g)
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
                    raise ParseError('Can not arrange two variables in one place.')
                elif right is None :
                    right = self._makeVariable(el)
                    # 操作符、操作数齐全，生成一层操作
                    g.insertNode(opt)
                    g.addEdge(left, opt)
                    g.insertNode(right)
                    g.addEdge(right, opt)
                    # 操作符成为下一层的左操作数
                    left = opt
                    opt = None
                    right = None
                else :
                    raise ParseError('Can not arrange two variables in one place.')
            
            # 对操作符命名
            elif el == 'as' :
                # 获取下一个元素
                el, rest_str = self._getNextEl(rest_str)

                if opt is not None :
                    raise ParseError('Only operator can be named by "as".')
                
                # 非法的命名
                if not self._isName(el) :
                    raise ParseError('"%s" is a illegal name.' % (el))
                
                # 给操作符命名
                self._setOperatorName(opt, el)
                # opt.data.name = el

            # 层连接符
            elif el == '->' or el == 'then' :
                # 层连接符不能放在表达式首部
                if left is None :
                    raise ParseError('Liner "%s" before all of statements is mistake.' % (el))
                
                # 操作符不为空，则构建此层，此层的操作符成为下一层的左操作数
                elif opt is not None :
                    g.insertNode(opt)
                    g.addEdge(left, opt)
                    if right is not None :
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
                g.insertNode(right)
                g.addEdge(right, opt)
            left = opt
        
        return left
        




