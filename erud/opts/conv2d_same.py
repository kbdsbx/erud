from erud.opts.conv2d_v2 import conv2d_v2
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

class conv2d_same (conv2d_v2) :

    def __init__(self, stride = 1) :
        super(conv2d_same, self).__init__(stride, 0)
    
    def fprop(self, x: np.ndarray, w: np.ndarray) -> np.ndarray :
        """
        m1 = floor((m1 + 2 * padding - p) / stride + 1)
        padding = ceil(((m1 - 1) * s - m1 + p) / 2)
        """
        (s, m1, n1, c1) = x.shape
        (p, q, c1, c2) = w.shape
        _stride = self.__stride
        self.__padding = int(np.ceil(((m1 - 1) * _stride - m1 + p) / 2))

        return super(conv2d_same, self).fprop(x, w)
    

    def bprop(self, dz: np.ndarray) -> list[np.ndarray] :

        return super(conv2d_same, self).bprop(dz)