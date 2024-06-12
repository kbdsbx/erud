from core.erud.node import node
from core.erud.var import var
from core.erud.graph import graph
from core.erud.interface.Igraph import Igraph
from core.erud.basic_operators.add import add

class variable(node) :
    """
    variable is a node that sames as operators. It content vars which is used to calculation.
    """

    payload : var = None

    def __init__ (self, value) :
        super(node, self).__init__()

        self.payload = var(value)
    
    def __add__(self, addend : Igraph) :
        opt = add(self, addend)
        return graph(self, opt, addend)
    
    def forward_value(self) -> var :
        return self.payload

    def backward_value(self) -> var :
        return self.payload
