
from core.erud.interface.Igraph import Igraph
from core.erud.interface.Iedge import Iedge
from core.erud.interface.Ilinkable import Ilinkable

from core.erud.edge import edge

class linkable (Ilinkable) :
    """
    Linkable abstract class is for support function to link others using edge. Some inherited this class mean that they have ability to add, remove, count links with node, graph and others
    """


    def __init__(self) -> None:

        self.forward_edges : list[Iedge] = []
        self.backward_edges : list[Iedge] = []

    def forward_nodes (self) -> list[Igraph] :
        """
        return node list linked front of this object.
        """
        return [e.link_to() for e in self.forward_edges]
    
    def backward_nodes (self) -> list[Igraph] :
        """
        return node list linked back of this object.
        """

        return [e.link_to() for e in self.backward_edges]

    def forward_link(self, target: Ilinkable, set_backward : bool = False) -> Iedge:
        """
        Linking between current object and target in forward. The worth attention is that the one aim can set many links more than one because the graph is not a simgle graph.

        Params
        target : Ilinkable
        set_backward : bool

        Return : Iedge, added edge
        """

        if set_backward :
            # avoid recurrent callback
            target.backward_link(self)

        e = edge(target)
        self.forward_edges.append(e)

        return e
    

    def backward_link(self, target: Ilinkable, set_forward: bool = False) -> Iedge:
        """
        Linking between current object and target in backward. The worth attention is that the one aim can set many links more than one because the graph is not a simgle graph.

        Params
        target : Ilinkable
        set_forward : bool

        Return : Iedge, added edge
        """

        if set_forward :
            target.forward_link(self)
        e = edge(target)
        self.backward_edges.append(e)

        return e
    

    def forward_unlink(self, target: Ilinkable, remove_backward: bool = False):
        """
        Unlink operator is to remove all links between current object and target.

        Params
        target : Ilinkable
        remove_backward : bool

        Return : list[Iedge], removed edges
        """

        if remove_backward :
            target.backward_unlink(self)
        
        elist = filter(lambda e : e.link_to() == target, self.forward_edges)

        for e in elist :
            self.forward_edges.remove(e)
        
        return list(elist)
    

    def backward_unlink(self, target: Ilinkable, remove_forward: bool = False):
        """
        Unlink operator is to remove all links between current object and target.

        Params
        target : Ilinkable
        remove_forward : bool

        Return : list[Iedge], removed edges
        """

        if remove_forward :
            target.forward_unlink(self)
        
        elist = filter(lambda e : e.link_to() == target, self.backward_edges)
        
        for e in elist :
            self.backward_edges.remove(e)
        
        return list(elist)
    
    def in_degree(self) -> int:
        return len(self.backward_edges)
    
    def out_degree(self) -> int:
        return len(self.forward_edges)