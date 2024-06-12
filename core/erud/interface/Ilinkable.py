
class Ilinkable :
    """
    Ilinkable interface is for support function to link others using edge. It supports functions to define standard operators for node linking.
    """

    ############ working in construction

    def forward_link(self, target : any, set_backward : bool) :
        """
        Link between current object and target in forward

        target : Ilinkable
        set_backward : bool, If it true, It will set backward link when call this function.
        """
        ...

    def backward_link(self, target : any, set_forward : bool) :
        """
        Link between current object and target in backward 
        
        target : Ilinkable
        set_backward : bool, If it true, It will set forward link when call this function.
        """
        ...

    def forward_unlink(self, target : any, remove_backward : bool) :
        """
        Unlink between current object and target in forward

        target : Ilinkable
        remove_backward : bool, If it true, It will remove backward link when call this function.
        """
        ...

    def backward_unlink(self, target : any, remove_forward : bool) :
        """
        Unlink between current object and target in backward 

        target : Ilinkable
        set_backward : bool, If it true, It will remove forward link when call this function.
        """
        ...
    
    def in_degree (self) -> int :
        """
        Return the number of link those which end with this object
        """
        ...
    
    def out_degree(self) -> int :
        """
        Return the number of link those which start with this object
        """
        ...