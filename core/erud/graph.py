
from core.erud.interface.Igraph import Igraph
from core.erud.interface.Iedge import Iedge
from core.erud.basic_operators.add import add


class graph (Igraph) :
    """
    Normal graph class which is a nested structure. It contents zero or more sub-graphs. Classes of those sub-graphs are as realizing the Igraph interface.
    """

    subgraphs : list[Igraph] = None

    def __init__(self, * subgraphs : Igraph) :
        self.subgraphs = subgraphs


    def forward_terminal_nodes(self) -> list[Igraph] :
        """
        collecting forward terminated nodes from each subgraph and then removing repeated nodes
        """
        nodes = []
        for g in self.subgraphs :
            sub_nodes = g.forward_terminal_nodes()
            for n in sub_nodes :
                if n not in nodes and n.out_degree() == 0 :
                    nodes.append(n)
        
        return nodes
    
    def backward_terminal_nodes(self) -> list[Igraph]:
        """
        collecting backward terminated nodes from each subgraph and then removing repeated nodes
        """

        nodes = []
        for g in self.subgraphs :
            sub_nodes = g.backward_terminal_nodes()
            for n in sub_nodes :
                if n not in nodes and n.in_degree() == 0:
                    nodes.append(n)
        
        return nodes
    
    # def forward_hang_edges(self) -> list[Iedge] :
    #     """
    #     collecting forward hang edges from each subgraph and then removing repeated edges.
    #     """

    #     edges = []
    #     for g in self.subgraphs :
    #         sub_edges = g.forward_hang_edges()
    #         for e in sub_edges :
    #             if e not in edges :
    #                 edges.append(e)
        
    #     return edges
    
    # def backward_hang_edges(self) -> list[Iedge] :
    #     """
    #     collecting backward hang edges from each subgraph and then removing repeated edges.
    #     """

    #     edges = []
    #     for g in self.subgraphs :
    #         sub_edges = g.backward_hang_edges()
    #         for e in sub_edges :
    #             if e not in edges :
    #                 edges.append(e)
        
    #     return edges
    

    def forward_value (self) -> list[any] :
        """
        Forward value is a list that contents calculated results which comes from terminated nodes
        """

        nodes = self.forward_terminal_nodes()

        return [n.forward_value() for n in nodes]
    
    def backward_value(self) -> list[any]:
        """
        Backward value is a list that contents calculated results which comes from terminated nodes
        """

        nodes = self.backward_terminal_nodes()

        return [n.backward_value() for n in nodes]
    

    def __add__(self, addend : Igraph) :
        opt = add(self, addend)
        return graph(self, opt, addend)