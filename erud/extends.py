"""
Imports / Exports / Transfer
"""

import json

from erud.nous import nous

def exports(n : nous, path : str, extends : object = {}) :
    """
    通过nous对象导出运行时缓存，导出格式为JSON

    * n : nous, 要导出的`nous`对象
    * path : str, 要导出的目的文件
    * extends : object, 要导出的其他有用信息
    """
    exp_obj = {
        'nous' : n.exports(),
        'extends' : extends
    }

    export_str = json.dumps(exp_obj)
    with open(path, "w", encoding="utf-8") as f :
        f.write(export_str)

def imports(path) :
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

    nous_data = imp_obj['nous']
    extends = imp_obj['extends']

    n = nous()
    n.imports(nous_data)

    return n, extends

def transfer(n : nous, src : str) :
    """
    从文件中获取另一个结构的网络参数并赋值到本网络

    根据名称匹配（匿名参数无法迁移）

    * n : nous, 当前实例化的nous
    * path : str, 结构文件
    """

    oldnous, _ = imports(src)
    oldg = oldnous.g
    for oldNode in oldg.nodes :
        for newNode in n.g.nodes :
            if (oldNode.data.name is not None) and oldNode.data.name == newNode.data.name :
                newNode.data.imports(oldNode.data.exports())

