from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np

# 多样本二维图片带通道（四维数据）的卷积
class conv2d(payload) :

    # 步长
    __stride : int
    # 扩展边距，通常用来填充零
    __padding: int

    # padding后的样本
    __px : np.ndarray = None
    # 卷积核
    __w : np.ndarray = None
    # 结果
    __z : np.ndarray = None

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

        (s, m1, n1, c1) = x.shape
        (p, q, c1, c2) = w.shape
        _padding = self.__padding
        _stride = self.__stride

        m2 = int(np.floor((m1 + (2 * _padding) - p) / _stride + 1))
        n2 = int(np.floor((n1 + (2 * _padding) - q) / _stride + 1))

        z = np.zeros((s, m2, n2, c2))

        # padding
        px = np.pad(x, ((0, 0), (_padding, _padding), (_padding, _padding), (0, 0)), "constant")

        for si in range(s) :
            for m2i in range(m2) :
                for n2i in range(n2) :
                    for c2i in range(c2) :
                        # 滑动窗口中的矩阵乘法
                        # 维度: (1, p, q, c1) * (p, q, c1, 1) -> sum -> (1, 1, 1, 1)
                        z[si, m2i, n2i, c2i] = np.sum( px[si, (_stride * m2i):(_stride * m2i + p), (_stride * n2i):(_stride * n2i + q), :] * w[:, :, :, c2i] )
        
        self.__z = z
        self.__px = px
        
        return z
    
    # 我很难跟你说清反向传播逻辑，因为我也写不出数学公式，只能写出工程代码
    def bprop(self, dz : np.ndarray) -> list[np.ndarray] :
        _px = self.__px
        _z = self.__z
        _w = self.__w

        _stride = self.__stride
        _padding = self.__padding

        (s, m2, n2, c2) = _z.shape
        (p, q, c1, c2) = _w.shape


        # dx = np.zeros_like(_x)
        # for si in range (s) :
        #     for m2i in range(m2) :
        #         for n2i in range(n2) :
        #             # 维度: (p, q, c1, c2) * (1, 1, 1, c2) -> sum(-1) -> (p, q, c1, 1)
        #             dx[si, m2i:(m2i + p), n2i:(n2i + q), :] += np.sum( _w[:, :, :, :] * dz[si, m2i, n2i, :], axis = -1)
        

        # dw = np.zeros_like(_w)
        # for c2i in range (c2) :
        #     for m2i in range(m2) :
        #         for n2i in range(n2) :
        #             # 维度: (s, p, q, c1) * (s, 1, 1, 1) -> sum(0) -> (1, p, q, c1)
        #             dw[:, :, :,  c2i] += np.sum( _x[:, m2i:(m2i + p), n2i:(n2i + q), :] * dz[:, m2i, n2i, c2i], axis = 0 )


        dpx = np.zeros_like(_px)
        dw = np.zeros_like(_w)
        for si in range (s) :
            for m2i in range(m2) :
                for n2i in range(n2) :
                    for c2i in range(c2) :
                        dpx[si, (m2i * _stride):(m2i * _stride + p), (n2i * _stride):(n2i * _stride + q), :] += (_w[:, :, :, c2i] * dz[si, m2i, n2i, c2i])
                        dw[:, :, :, c2i] += (_px[si, (m2i * _stride):(m2i * _stride + p), (n2i * _stride):(n2i * _stride + q), :] * dz[si, m2i, n2i, c2i])
        
        dx = dpx[:, _padding : -_padding, _padding : -_padding, :]

        return [dx, dw]





