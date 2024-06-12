
from core.erud.interface.Igraph import Igraph
from core.erud.interface.Iedge import Iedge
from core.erud.linkable import linkable

class node (Igraph, linkable) :
    """
    Node abstract class is as the common graph. Its terminal node is itself whenever in forward or backward propagation. It can content many edges. The value of each node depends on its concret type which could realize by inheritance.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward_terminal_nodes(self) -> list[Igraph]:
        return [self]
    
    def backward_terminal_nodes(self) -> list[Igraph]:
        return [self]
    
    # def forward_hang_edges(self) -> list[Iedge]:
    #     return list[filter(lambda e : e.link_to() != None, [e for e in self.forward_edges])]
    
    # def backward_hang_edges(self) -> list[Iedge]:
    #     return list[filter(lambda e : e.link_to() != None, [e for e in self.backward_edges])]

    def forward_value () -> any :
        ...

    def backward_value() -> any :
        ...

