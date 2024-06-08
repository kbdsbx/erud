from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as cp
import numpy as np
from erud._utils import epsilon as eps


class yolo1_loss(payload) :
    __lcoord : any
    __lnoobj : any
    __mask : any
    __a : any
    __ysqrts : any
    __yhatsqrts : any

    def iou(self, i, j, x, y, w, h, xhat, yhat, what, hhat) :
        # 将比例还原成真实大小
        x1 = x * 64 + i * 64
        y1 = y * 64 + j * 64
        w1 = w * 448
        h1 = h * 448
        x2 = xhat * 64 + i * 64
        y2 = yhat * 64 + j * 64
        w2 = what * 448
        h2 = hhat * 448

        # 计算四角坐标
        l1 = x1 - w1 / 2
        t1 = y1 - h1 / 2
        r1 = x1 + w1 / 2
        b1 = y1 + h1 / 2
        l2 = x2 - w2 / 2
        t2 = y2 - h2 / 2
        r2 = x2 + w2 / 2
        b2 = y2 + h2 / 2

        # 计算交叉框大小
        w = min(r1, r2) - max(l1, l2)
        h = min(b1, b2) - max(t1, t2)

        # 计算交集
        # 负的相交框宽/高表示两个框相离
        if w < 0 or h < 0 :
            i = 0
        else :
            i = w * h
        
        # 计算原始框大小
        s1 = (r1 - l1) * (b1 - t1)
        s2 = (r2 - l2) * (b2 - t2)

        # 计算并集
        u = s1 + s2 - i

        # 计算IoU
        return max(0, i / u)

    def __init__(self, lcoord = 5., lnoobj = 0.5) :
        self.__lcoord = lcoord
        self.__lnoobj = lnoobj
    
    def fprop(self, yhat, y) -> any :
        '''
        yhat: s * 7 * 7 * [p1, x1, y1, w1, h1, p2, x2, y2, w2, h2, c1, c2, ..., c20]
        y: s * 7 * 7 * [p, x, y, w, h, p, x, y, w, h, c1, c2, ..., c20]
        p为1/0时可以使用此代码，若使用标签平滑等技术则无法使用
        '''
        _lcoord = self.__lcoord
        _lnoobj = self.__lnoobj
        mask = np.zeros_like(y)

        # 计算掩码
        # 掩码执行Loss中的1函数功能及权重参数功能
        # 此处可能需要C++来优化
        (sc, mc, nc, _) = y.shape
        for s in range(sc) :
            for m in range(mc) :
                for n in range(nc) :
                    bbox1 = yhat[s, m, n, 1:5]
                    bbox2 = yhat[s, m, n, 6:10]
                    truebox = y[s, m, n, 1:5]
                    s1 = self.iou(m, n, *truebox, *bbox1)
                    s2 = self.iou(m, n, *truebox, *bbox2)
                    # 若bbox1为最佳匹配，则将bbox2的坐标相关掩码置零，使其不提供Loss，反之亦然
                    if y[s,m,n,0] :
                        if s1 > s2 :
                            mask[s, m, n, 0] = 1
                            mask[s, m, n, 1:5] = _lcoord
                            mask[s, m, n, 6:10] = 0
                        else :
                            mask[s, m, n, 0:5] = 0
                            mask[s, m, n, 5] = 1
                            mask[s, m, n, 6:10] = _lcoord

                        # 类别预测掩码只考虑是否存在truth box，不考虑bbox的IoU
                        mask[s, m, n, 10:] = 1
                    # 而置信度掩码当不含truth box时提供一个较小的损失
                    else :
                        mask[s,m,n,:] = 0
                        mask[s,m,n,np.r_[0, 5]] = _lnoobj


        self.__mask = mask

        # 对宽度和高度取开平方来降低对不同大小图片的影响
        yhat[:, :, :, np.r_[1:5, 6:10]] = np.sqrt( yhat[:, :, :, np.r_[1:5, 6:10]])
        y[:, :, :, np.r_[1:5, 6:10]] = np.sqrt(y[:, :, :, np.r_[1:5, 6:10]])

        # 存储开方后的宽度和高度
        self.__yhatsqrts = np.ones_like(yhat)
        self.__yhatsqrts[:,:,:,np.r_[1:5, 6:10]] = yhat[:, :, :, np.r_[1:5, 6:10]]
        self.__ysqrts = np.ones_like(y)
        self.__ysqrts[:, :, :, np.r_[1:5, 6:10]] = y[:, :, :, np.r_[1:5, 6:10]]

        _a = y - yhat
        self.__a = _a

        z = _a * _a * mask

        return z
    
    def bprop(self, dz) -> list[any] :
        _a = self.__a
        _mask = self.__mask
        _yhatsqrts = self.__yhatsqrts
        _ysqrts = self.__ysqrts

        dyhat = -2 * _a * dz * _mask * (0.5 / (_yhatsqrts + eps))
        dy = 2 * _a * dz * _mask * (0.5 / (_ysqrts + eps))

        return [dyhat, dy]


