from core.erud.var import var

class Icalculator:
    """
    Interface of calculator.
    A calculator must realize this interface to offer a exection and decline-exection to calculation.
    """

    ############ working in propagation

    def exec() -> var:
        """
        exection.
        executing a operation in forward propagation.
        """
        ...
    

    def dexec() -> var:
        """
        decline exection.
        execting a operation in backward propagation.
        """
        ...