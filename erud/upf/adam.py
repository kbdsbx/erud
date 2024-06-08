from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
from erud._utils import epsilon as eps
from erud.upf.updateable import updateable

class adam (updateable) : 
    __velocity : any = None
    __square : any = None
    __rate : float
    __beta_momentum : float
    __beta_rms : float

    @property
    def rate (self) :
        return self.__rate

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
    
    def exports(self) :
        obj = {
            'class' : 'adam',
            'rate' : self.__rate,
            'beta_momentum' : self.__beta_momentum,
            'beta_rms' : self.__beta_rms,
        }

        if self.__velocity is None :
            obj['velocity'] = None
        else :
            obj['velocity'] = self.__velocity.tolist()

        if self.__square is None :
            obj['square'] = None
        else :
            obj['square'] = self.__square.tolist()
    
    def imports(self, value) :
        self.__rate = value['rate']
        self.__beta_momentum = value['beta_momentum']
        self.__beta_rms = value['beta_rms']
        if value['velocity'] != None :
            self.__velocity = np.array(value['velocity'])
        if value['square'] != None :
            self.__square = np.array(value['square'])