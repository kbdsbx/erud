from erud.cg.payload import payload
from erud.c_extend.max_pool import max_pool_fprop, max_pool_bprop
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

class max_pool_v3(payload) :

    # 步长
    __stride : int = None

    # 存储池化时最大值的位置
    __cx : np.ndarray = None

    s : int = 0
    m1 : int = 0
    n1 : int = 0
    c : int = 0
    m2 : int = 0
    n2 : int = 0
    p : int = 0
    q : int = 0


    def __init__(self, stride: int = 1, p : int = 1, q : int = 1) :
        self.__stride = stride
        self.__p = p
        self.__q = q
    

    def fprop(self, x : np.ndarray) -> np.ndarray :
        _stride = self.__stride
        _p = self.__p
        _q = self.__q
        (s, m1, n1, c) = x.shape


        m2 = int(np.floor((m1 - _p) / _stride + 1))
        n2 = int(np.floor((n1 - _q) / _stride + 1))

        z = np.zeros((s, m2, n2, c), dtype=np.float32)

        x = x.reshape((s * m1 * n1 * c))
        z = z.reshape((s * m2 * n2 * c))
        cx = np.zeros((s * m1 * n1 * c), dtype=np.int32)

        # C++扩展调用
        max_pool_fprop(x, z, cx, s, m1, n1, m2, n2, _p, _q, c, _stride)

        z = z.reshape((s, m2, n2, c))

        # 缓存反向传播用得到的内容
        self.__cx = cx
        self.s = s
        self.m1 = m1
        self.n1 = n1
        self.m2 = m2
        self.n2 = n2
        self.c = c
        self.p = _p
        self.q = _q

        return z

    def bprop(self, dz : np.ndarray) -> list[np.ndarray] :
        _stride = self.__stride

        s = self.s
        m1 = self.m1
        n1 = self.n1
        m2 = self.m2
        n2 = self.n2
        c = self.c
        p = self.p
        q = self.q

        # 参数
        _cx = self.__cx
        dz = dz.reshape((s * m2 * n2 * c))

        # 结果缓存
        dx = np.zeros((s * m1 * n1 * c), dtype=np.float32)

        # C++ 扩展调用
        max_pool_bprop(_cx, dz, dx, s, m1, n1, m2, n2, p, q, c, _stride)

        # 处理返回的结果
        dx = dx.reshape((s, m1, n1, c))

        return [dx]
