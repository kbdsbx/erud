from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

class momentum :
    __velocity : any = None
    __rate : float
    __beta : float

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
