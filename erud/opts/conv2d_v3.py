from erud.cg.payload import payload
from erud.c_extend.conv2d import conv2d_fprop, conv2d_bprop
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np

class conv2d_v3(payload) :

    # 步长
    __stride : int = None
    # 扩展边距，通常用来填充零
    __padding: int = None

    # 卷积核
    __w : np.ndarray = None
    # padding后形变后的X（cache X）
    __cx : np.ndarray = None

    s : int = 0
    m1 : int = 0
    n1 : int = 0
    c1 : int = 0
    m2 : int = 0
    n2 : int = 0
    c2 : int = 0
    p : int = 0
    q : int = 0

    @property
    def stride (self) -> any:
        return self.__stride
    
    @stride.setter
    def stride(self, s) -> any :
        self.__stride = s
        return s
    
    @property
    def padding(self) -> any:
        return self.__padding
    
    @padding.setter
    def padding(self, p) -> any:
        self.__padding = p
        return p


    def __init__(self, stride = 1, padding = 0) :
        self.__stride = stride
        self.__padding = padding
    
    
    def fprop(self, x : np.ndarray, w : np.ndarray) -> np.ndarray :
        (s, m1, n1, c1) = x.shape
        (p, q, c1, c2) = w.shape
        _padding = self.__padding
        _stride = self.__stride

        m2 = int(np.floor((m1 + (2 * _padding) - p) / _stride + 1))
        n2 = int(np.floor((n1 + (2 * _padding) - q) / _stride + 1))

        (self.s, self.m1, self.n1, self.c1) = x.shape
        (self.p, self.q, self.c1, self.c2) = w.shape
        self.m2 = m2
        self.n2 = n2

        # padding
        px = np.pad(x, ((0, 0), (_padding, _padding), (_padding, _padding), (0, 0)), "constant")


        # 将整个张量拉成向量
        px = px.reshape((s * (m1 + _padding * 2) * (n1 + _padding * 2) * c1))
        w = w.reshape((p * q * c1 * c2))
        z = np.zeros((s * m2 * n2 * c2), dtype=np.float32)
        cx = np.zeros((s * m2 * n2 * p * q * c1), dtype=np.float32)

        # c++扩展调用
        conv2d_fprop(px, w, z, cx, s, (m1 + _padding * 2), (n1 + _padding * 2), c1, m2, n2, c2, p, q, _stride)

        z = z.reshape((s, m2, n2, c2))

        # 存储反向传播用得到的内容
        self.__cx = cx
        self.__w = w

        return z
    
    def bprop(self, dz : np.ndarray) -> list[np.ndarray] :
        _stride = self.__stride
        _padding = self.__padding

        s = self.s
        m1 = self.m1
        n1 = self.n1
        c1 = self.c1
        m2 = self.m2
        n2 = self.n2
        c2 = self.c2
        p = self.p
        q = self.q

        # 参数
        _cx = self.__cx
        _w = self.__w
        dz = dz.reshape((s * m2 * n2 * c2))
       

        # 存放结果的缓存
        dpx = np.zeros((s * (m1 + _padding * 2) * (n1 + _padding * 2) * c1), dtype=np.float32)
        dw = np.zeros((p * q * c1 * c2), dtype=np.float32)

        # c++扩展调用
        conv2d_bprop(_cx, _w, dz, dpx, dw, s, (m1 + _padding * 2), (n1 + _padding * 2), c1, m2, n2, c2, p, q, _stride)

        # 处理结果
        if _padding :
            dx = dpx.reshape((s, m1 + _padding * 2, n1 + _padding * 2, c1))[:, _padding : -_padding, _padding : -_padding, :]
        else :
            dx = dpx.reshape((s, m1, n1, c1))
        
        dw = dw.reshape((p, q, c1, c2))
        

        return [dx, dw]

