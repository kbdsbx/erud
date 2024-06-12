
from core.erud.interface.Iedge import Iedge
from core.erud.interface.Igraph import Igraph
from core.erud.interface.Ilinkable import Ilinkable


class edge (Iedge) :
    """
    Edge class is the uni-directional edge that includes only not original but target node.
    """

    to : Ilinkable = None

    def __init__(self, to: Ilinkable) :
        self.to = to

    def link_to(self) -> Igraph:
        return self.to