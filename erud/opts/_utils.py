import numpy as np

# 获得广播的轴，如果在计算在z = x [opt] y时发生了广播，则broadcast_axis(z, x)返回计算时相对z，x广播的轴
# z: 计算结果
# x: 计算变量
# return: array 发生广播的轴
def broadcast_axis(z : np.ndarray , x : np.ndarray) :
    zshape = z.shape
    xshape = x.shape
    cast = len(zshape) - len(xshape)

    if cast < 0 :
        raise ValueError('Can not boradcast shape %s to %s' % (xshape, zshape))
    
    nxshape = tuple([1 for _ in range(cast)]) + xshape

    axis = []
    for i in range(len(zshape)) :
        if zshape[i] == nxshape[i] :
            continue
        elif nxshape[i] == 1 :
            axis.append(i)
        else :
            raise ValueError('Can not boradcast shape %s to %s' % (xshape, zshape))
    
    return axis

# 通过对张量的偏导数相加计算其导数
# 缩小dx的维度使其与x的维度相同
# dx: dz/dx，即dz对dx的偏导数
# x: 原操作数
# return: scalar or np.ndarry 与x维度相同的dx
def partial_sum(dx : any, x : any) -> any:
    if isinstance(x, np.ndarray) :
        # 如果x的维度和dx不一样，说明x在z = f(x,...)的运算中发生了广播
        # 如果两者一样，那么casts=[]，np.sum函数将不做任何事
        # 找到运算中广播的轴——意味着dx沿着这些轴上的每一个值都是x的偏导
        casts = broadcast_axis(dx, x)
        # 轴上偏导相加，成为导数
        dx = np.sum(dx, axis = tuple(casts))
        dx = np.reshape(dx, x.shape)
    else :
        # 如果x是标量，那么dx中每一个值都是x的偏导，直接沿着所有轴相加即可
        dx = np.sum(dx)

    return dx