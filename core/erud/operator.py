from core.erud.node import node
from core.erud.interface.Igraph import Igraph
from core.erud.interface.Icalculator import Icalculator
from core.erud.var import var

class operator (Icalculator, node) :
    """
    Basic class for all type of operator
    """

    def __init__(self, * subgraphs : Igraph):
        super().__init__()

        for subg in subgraphs :
            nodes = subg.forward_terminal_nodes()
            for n in nodes :
                self.backward_link(n, True)

    def fprop(func) :
        def wrapper (*args, **kwargs) :
            return func(*args, **kwargs)
        
        return wrapper
    
    def bprop(func) :
        def wrapper(*args, **kwargs) :
            return func(*args, **kwargs)
        
        return wrapper
    
    def forward_value(self) -> var:
        return self.exec(* [n.forward_value() for n in self.backward_nodes()])
    
    def backward_value(self) -> var:
        return self.dexec(* [n.forward_value() for n in self.forward_nodes()])


