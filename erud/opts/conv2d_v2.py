from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

class conv2d_v2(payload) :

    # 步长
    __stride : int = None
    # 扩展边距，通常用来填充零
    __padding: int = None

    # padding后的输入
    __px : np.ndarray = None
    # 拉平的输入
    __tx : np.ndarray = None
    # 拉平的卷积核
    __tw : np.ndarray = None
    # 结果
    __w : np.ndarray = None

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
        """  
        输入样本需固定为shape=(s, m1, n1, c1)

        s 为样本数
        m1 为样本宽度
        n1 为样本高度
        c1 为样本通道数
        
        卷积核参数需固定为shape=(p, q, c1, c2)

        p 为卷积核宽度
        q 为卷积核高度，通常与宽度相等
        c1 为输入（样本）通道数
        c2 为输出通道数

        
        卷积计算返回值固定为shape=(s, m2, n2, c2)

        s 为样本数，不变

        m2 为输出宽度，m2 = floor((m1 + 2 * padding - p) / stride + 1)
            m1 为输入宽度
            padding 为增加的边距，通常根据valid、same、full取不同的值
            p 为卷积核宽度
            stride 为步长
            
        n2 为输出高度，n2 = floor((n1 + 2 * padding - q) / stride + 1)
            n1 为输入高度
            padding 为增加的边距，通常根据valid、same、full取不同的值
            q 为卷积核高度
            stride 为步长

        c2 为输出通道数，或理解为三维卷积核的个数
        """
        self.__w = w
        self.__x = x

        (s, m1, n1, c1) = x.shape
        (p, q, c1, c2) = w.shape
        _padding = self.__padding
        _stride = self.__stride

        m2 = int(np.floor((m1 + (2 * _padding) - p) / _stride + 1))
        n2 = int(np.floor((n1 + (2 * _padding) - q) / _stride + 1))

        # padding
        px = np.pad(x, ((0, 0), (_padding, _padding), (_padding, _padding), (0, 0)), "constant")

        # 将滑动窗口值拉成向量
        # (s, m1, n1, c1) -> (s, m2, n2, p * q * c1)
        tx = np.zeros((s, m2, n2, p * q * c1))

        for si in range(s) :
            for m2i in range(m2) :
                for n2i in range(n2) :
                    tx[si, m2i, n2i, :] = (px[si, (_stride * m2i):(_stride * m2i + p), (_stride * n2i):(_stride * n2i + q), :]).reshape((p * q * c1))
        
        # (p * q * c1, c2)
        tw = w.reshape((p * q * c1, c2))

        # 卷积转换成矩阵乘法 (s, m2, n2, c2)
        z = np.matmul(tx, tw)
        # gpu 加速
        # z = cp.matmul(cp.array(tx), cp.array(tw)).get()

        self.__tw = tw
        self.__tx = tx
        self.__px = px

        return z
    
    def bprop(self, dz : np.ndarray) -> list[np.ndarray] :
        _tw = self.__tw
        _tx = self.__tx
        _stride = self.__stride
        _padding = self.__padding
        (s, m1, n1, c1) = self.__px.shape
        (p, q, c1, c2) = self.__w.shape
        (s, m2, n2, c2) = dz.shape

        _tw_temp = _tw.transpose((1, 0))
        # (s, m2, n2, c2) * (c2, p * q * c1) = (s, m2, n2, p * q * c1)
        dtx = np.matmul(dz, _tw_temp).reshape(_tx.shape)
        dpx = np.zeros_like(self.__px)

        for si in range(s) :
            for m2i in range(m2) :
                for n2i in range(n2) :
                    dpx[si, (m2i * _stride):(m2i * _stride + p), (n2i * _stride):(n2i * _stride + q), :] += (dtx[si, m2i, n2i, :]).reshape((p, q, c1))

        if _padding :
            dx = dpx[:, _padding : -_padding, _padding : -_padding, :]
        else :
            dx = dpx


        _tx_temp = _tx.transpose((0, 1, 3, 2))
        # (s, m2, p * q * c1, n2) * (s, m2, n2, c2) = (s, m2, p * q * c1, c2)
        dtw = np.matmul(_tx_temp, dz)
        dw = np.sum(dtw, axis = (0, 1)).reshape((p, q, c1, c2))


        return [dx, dw]

