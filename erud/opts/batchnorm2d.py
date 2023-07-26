from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np
from erud._utils import epsilon as eps

class batchnorm2d(payload) :

    __x : any = None
    __mu : np.ndarray = None
    __sigma : np.ndarray = None

    __s : int = None
    __m : int = None
    __n : int = None

    def fprop(self, x) -> any :
        (s, m, n, c) = x.shape

        _mu = 1. / (s * m * n) * np.sum(x, axis = (0, 1, 2), keepdims = True)
        _sigma = np.sqrt(1. / (s * m * n) * np.sum(np.power(x - _mu, 2), axis = (0, 1, 2), keepdims = True) + eps)

        self.__x = x
        self.__mu = _mu
        self.__sigma = _sigma
        self.__s = s
        self.__m = m
        self.__n = n

        return (x - _mu) / _sigma
    
    def bprop(self, dz) -> list[any] :

        _x = self.__x
        _mu = self.__mu
        _sigma = self.__sigma
        _s = self.__s
        _m = self.__m
        _n = self.__n

        dx = (dz / _sigma) - (np.sum(dz, axis = (0, 1, 2), keepdims = True) / (_s * _m * _n) / _sigma) - ((np.sum(dz * (_x - _mu), axis = (0, 1, 2), keepdims = True) / (_s * _m * _n) / np.power(_sigma, 3)) * (_x - _mu))

        return [dx]