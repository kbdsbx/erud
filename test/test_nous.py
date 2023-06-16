from erud.nous import nous
from erud.errors import ParseError
import pytest as test


# 获取代码串中的第一个元素和剩余的子串
def test_nous_get_next_el() :
    n = nous()

    assert n._getNextEl('X:1332') == ('X:1332', '')
    assert n._getNextEl(' X:') == ('X:', '')
    assert n._getNextEl('   X[]') == ('X[]', '')
    assert n._getNextEl('X[[1, (2)], [(3), 4]]') == ('X[[1, (2)], [(3), 4]]', '')
    assert n._getNextEl('x:42 error sse') == ('x:42', 'error sse')
    assert n._getNextEl('X:(1000, 10) matmul W1:r(10, 50) -> ') == ('X:(1000, 10)', 'matmul W1:r(10, 50) ->')
    assert n._getNextEl('(((((test)))))') == ('test', '')
    assert n._getNextEl('(((((test))))) rest') == ('test', 'rest')
    assert n._getNextEl('((([((test))]))) rest') == ('[((test))]', 'rest')
    assert n._getNextEl('(a add b) rest') == ('a add b', 'rest')
    assert n._getNextEl('') == ('', '')

    # 错误的表达式无法解析，半个中括号
    with test.raises(ParseError) :
        n._getNextEl('   X[')

    # 错误的表达式无法解析，右侧少一个小括号
    with test.raises(ParseError) :
        n._getNextEl('''(((([[{{43ss}}]])))''')