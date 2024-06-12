from core.erud.interface.Igraph import Igraph
from core.erud.interface.Icalculator import Icalculator
from core.erud.var import var
from core.erud.operator import operator
from core.erud.linkable import linkable

def _add (x : var, y : var) -> var:
    return var(x.value + y.value)

class add (operator) :
    _x : var
    _y : var

    def __init__(self, g_x: Igraph, g_y: Igraph) :
        super(add, self).__init__(g_x, g_y)

    def exec (self, x : var, y : var) -> var :
        self._x = x
        self._y = y
        return _add(x, y)

    def dexec (self, z : var) -> var :
        pass
