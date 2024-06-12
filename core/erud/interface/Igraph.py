from core.erud.interface.Ilinkable import Ilinkable

class Igraph:
    """
    Interface of graph.
    The graph have some genres below:
        1) just includes one node named as common graph
        2) includes a group of nodes but not have any edge
        3) includes a group of nodes and edges but some edges are not having output
        4) a normal graph
    Bacause of that the graph in this framework is directed graph, each graph includes methods below: 
        1) get the forward edges
        2) get the backward edges
        3) get the forward nodes
        4) get the backward nodes
    --------------------------------------------------------------
        5) get the hanging edges in forward propagation
        6) get the hanging edges in backward propagation
    Hanging edge means that a edge has input nodes but haven't output node. A graph that has some hanging edge can't be computed in propagating process.
    """

    ############ working in construction

    def forward_terminal_nodes () -> list[Ilinkable] :
        """
        collecting nodes in this graph. The nodes have no output edge in forward propagation.
        """
        ...


    def backward_terminal_nodes () -> list[Ilinkable] :
        """
        collecting nodes in this graph. The nodes have no output edge in backward propagation.
        """
        ...


    # def forward_hang_edges () -> list[any] :
    #     """
    #     collecting edges in this graph. The edges have no aimed node in forward propagation.
    #     """
    #     ...

    # def backward_hang_edges () -> list[any] :
    #     """
    #     collecting edges in this graph. The edges have no aimed node in backward propagation.
    #     """
    #     ...

    ############ working in propagation

    def forward_value () -> any :
        """
        calculating value that relies on all graph and node back of current graph. The number of returned values equals with the number of output edges.
        """
        ...


    def backward_value () -> any :
        """
        calculating value that relies on all graph and node front of current graph. The number of returned values equals with the number of input edges.
        """
        ...