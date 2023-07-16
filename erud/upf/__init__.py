"""
Update parameters Fucntions

封装更新参数所需的方法
"""

from erud.upf.norm import norm as norm_class
from erud.upf.momentum import momentum as momentum_class
from erud.upf.adam import adam as adam_class

def norm (*args) :
    """
    普通更新函数
    (z, dz) : z - rate * dz

    ### 参数
    rate : 更新率
    """
    return norm_class(*args).updateFunc

def momentum (*args) :
    """
    稀有更新函数

    使用移动加权平均修正偏差，能够增加梯度下降的速度
    （参考momentum相关文献）

    ### 参数
    rate : 更新率
    beta : 移动加权平均的权重
    """
    return momentum_class(*args).updateFunc

def adam (*args) :
    """
    史诗更新函数

    不仅使用加权平均进行偏差修正，还使用平方平均根对不同方向上的参数更新量进行调整，能大幅增加梯度下降速度
    （参考adam相关文献）

    ### 参数
    rate : 更新率
    beta_momentum : 移动加权平均的权重
    beta_rms: 平方根调整动量的权重
    """
    return adam_class(*args).updateFunc