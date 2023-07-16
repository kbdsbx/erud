"""
Imports and Exports
"""

import json

from erud.nous import nous

def nous_exports(n : nous, path : str, extends : object = {}) :
    """
    通过nous对象导出运行时缓存，导出格式为JSON

    * n : nous, 要导出的`nous`对象
    * path : str, 要导出的目的文件
    * extends : object, 要导出的其他有用信息
    """
    if n.g is not None :
        exp_obj = {
            'code' : n.code,
            'nodes' : n.g.exports(),
            'extends' : extends
        }
    else :
        exp_obj = {
            'code' : n.code,
            'extends' : extends
        }

    export_str = json.dumps(exp_obj)
    with open(path, "w", encoding="utf-8") as f :
        f.write(export_str)

def nous_imports(path) :
    """
    从文件中导入之前运行的缓存并还原

    * path : str, 要导出的缓存源文件

    ### 返回值
    * n : nous, 实例化的`nous`对象
    * extends : object, 之前保存的其他有用信息
    """
    imports_str = ''
    with open(path, "r", encoding="utf-8") as f :
        imports_str = f.read()
    imp_obj = json.loads(imports_str)

    code = imp_obj['code']
    nodes = imp_obj['nodes']
    extends = imp_obj['extends']

    n = nous(code)
    n.parse()
    n.g.imports(value = nodes)

    return n, extends

