import numpy as np
from erud._utils import epsilon as eps

class adam : 
    __velocity : any = None
    __square : any = None
    __rate : float
    __beta_momentum : float
    __beta_rms : float

    def __init__ (self, rate, beta_momentum = 0.9, beta_rms = 0.999) :
        self.__rate = rate
        self.__beta_momentum = beta_momentum
        self.__beta_rms = beta_rms
    
    def updateFunc(self, z, dz) :
        if self.__velocity is None :
            self.__velocity = np.zeros_like(z)
        if self.__square is None :
            self.__square = np.zeros_like(z)
        
        _rate = self.__rate
        _beta1 = self.__beta_momentum
        _beta2 = self.__beta_rms

        # 移动加权平均
        self.__velocity = (_beta1 * self.__velocity) + ((1. - _beta1) * dz)
        # 平方根调整更新动量
        self.__square = (_beta2 * self.__square) + ((1. - _beta2) * np.power(dz, 2))

        return z - (_rate * self.__velocity / np.sqrt(self.__square + eps))