from erud.upf.updateable import updateable

class norm(updateable) :
    __rate : float

    @property
    def rate (self) :
        return self.__rate

    def __init__(self, rate):
        self.__rate = rate

    def updateFunc(self, z, dz) :
        return z - self.__rate * dz
    
    def exports(self) :
        return {
            'class' : 'norm',
            'rate' : self.__rate,
        }
    
    def imports(self, value) :
        self.__rate = value['rate']
