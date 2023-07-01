
class norm :
    __rate : float

    def __init__(self, rate):
        self.__rate = rate

    def updateFunc(self, z, dz) :
        return z - self.__rate * dz