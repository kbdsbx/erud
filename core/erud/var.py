
class var :
    """
    Variable is a container that contents a concrete value in computer memory.
    """

    value : any

    def __init__(self, val: any) :
        self.value = val

    def __str__(self) :
        return str(self.value)
    
    def __repr__(self) :
        return repr(self.value)