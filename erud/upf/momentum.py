from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np
from erud.upf.updateable import updateable

class momentum(updateable) :
    __velocity : any = None
    __rate : float
    __beta : float

    @property
    def rate (self) :
        return self.__rate

    def __init__ (self, rate, beta = 0.9) :
        self.__rate = rate
        self.__beta = beta
    
    def updateFunc (self, z, dz) : 
        if self.__velocity is None :
            self.__velocity = np.zeros_like(z)
        
        _rate = self.__rate
        _beta = self.__beta
        
        # 移动加权平均
        self.__velocity = (_beta * self.__velocity) + ((1. - _beta) * dz)

        return z - (_rate * self.__velocity)
    

    def exports(self) :
        return {
            'class' : 'momentum',
            'rate' : self.__rate,
            'beta' : self.__beta,
            'velocity' : self.__velocity.tolist(),
        }
    
    def imports(self, value) :
        self.__rate = value['rate']
        self.__beta = value['beta']
        self.__velocity = np.array(value['velocity'])
