
class Iedge :
    """
    Interface of edge.

    In the traditional computation graph, edges have no matter on graph's construction and computation. The only thing they could do is let the current node know which node is the next.
    """

    ############ working in construction

    def link_to() :
        """
        return which node is linked.
        """
        ...
