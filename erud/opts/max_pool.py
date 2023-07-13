from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

class max_pool(payload) :

    __stride : int
    __p : int
    __q : int

    __x : np.ndarray

    def __init__(self, stride : int = 1, p : int = 1, q : int = 1) :
        self.__stride = stride
        self.__p = p
        self.__q = q
    
    def fprop(self, x : np.ndarray) -> np.ndarray :
        self.__x = x

        _stride = self.__stride
        _p = self.__p
        _q = self.__q
        (s, m1, n1, c1) = x.shape


        m2 = int(np.floor((m1 - _p) / _stride + 1))
        n2 = int(np.floor((n1 - _q) / _stride + 1))

        z = np.zeros((s, m2, n2, c1))

        for si in range(s) :
            for m2i in range(m2) :
                for n2i in range(n2) :
                    for c1i in range(c1) :
                        z[si, m2i, n2i, c1i] = np.max(x[si, (_stride * m2i):(_stride * m2i + _p), (_stride * n2i):(_stride * n2i + _q), c1i])
        
        return z

    def bprop(self, dz : np.ndarray) -> list[np.ndarray] :
        _x = self.__x
        _stride = self.__stride
        _p = self.__p
        _q = self.__q

        (s, m2, n2, c1) = dz.shape

        dx = np.zeros_like(_x)

        for si in range(s) :
            for m2i in range(m2) :
                for n2i in range(n2) :
                    for c1i in range(c1) :
                        _slice = _x[si, (m2i * _stride):(m2i * _stride + _p), (n2i * _stride):(n2i * _stride + _q), c1i]
                        dx[si, (m2i * _stride):(m2i * _stride + _p), (n2i * _stride):(n2i * _stride + _q), c1i] += (dz[si, m2i, n2i, c1i] * (np.max(_slice) == _slice))
        
        return [dx]


