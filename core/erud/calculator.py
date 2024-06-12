from core.erud.interface.Icalculator import Icalculator
from core.erud.interface.Igraph import Igraph
from core.erud.graph import graph
from core.erud.node import node


class calculator(Icalculator, node) :

    def exec(* subgraphs : Igraph) -> any:
        ...
    
    def dexec(* subgraphs : Igraph) -> any:
        ...

    