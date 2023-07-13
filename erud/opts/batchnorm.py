from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np
from erud._utils import epsilon as eps

class batchnorm(payload) :

    __x : any
    __m : int
    __mu : float
    __sigma2 : float

    def fprop(self, x) -> any :
        _m = np.size(x)

        _mu = 1. / _m * np.sum(x)
        _sigma2 = 1. / _m * np.sum(np.power(x - _mu, 2))

        self.__x = x
        self.__m = _m
        self.__mu = _mu
        self.__sigma2 = _sigma2

        return (x - _mu) / np.sqrt(_sigma2 + eps)
    
    def bprop(self, dz) -> list[any] :
        _x = self.__x
        _m = self.__m
        _mu = self.__mu
        _sigma = np.sqrt(self.__sigma2)

        # dx = (dz / _sigma) - np.sum(dz / _m / _sigma) - np.sum(dz * (_x - _mu) / _m / np.power(_sigma, 3)) * (_x - _mu) + np.sum((_x - _mu) * np.sum(_x - _mu) / np.power(_m, 2) / np.power(_sigma, 3))
        dx = (dz / _sigma) - (np.sum(dz) / _m / _sigma) - ((np.sum(dz * (_x - _mu)) / _m / np.power(_sigma, 3)) * (_x - _mu))

        return [dx]
