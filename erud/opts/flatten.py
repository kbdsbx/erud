from erud.cg.payload import payload
import numpy as np

class flatten (payload) :
    __x : np.ndarray = None

    def fprop(self, x :np.ndarray) -> np.ndarray :
        self.__x = x

        (s, m1, n1, c1) = x.shape

        return x.reshape((s, m1 * n1 * c1))
    
    def bprop(self, dz : np.ndarray) -> list[np.ndarray] :
        (s, m1, n1, c1) = self.__x.shape

        dx = dz.reshape(s, m1, n1, c1)

        return [dx]