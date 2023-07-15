"""
Imports and Exports
"""

import json

from erud.nous import nous

def nous_exports(n : nous, path) :
    exp_obj = {
        'code' : n.code,
        'nodes' : n.g.exports(),
    }

    export_str = json.dumps(exp_obj)
    with open(path, "w", encoding="utf-8") as f :
        f.write(export_str)

def nous_imports(n : nous, path) :
    imports_str = ''
    with open(path, "r", encoding="utf-8") as f :
        imports_str = f.read()
    imp_obj = json.loads(imports_str)

    code = imp_obj['code']
    nodes = imp_obj['nodes']

    n.parse(code)
    n.g.imports(value = nodes)

    return n

