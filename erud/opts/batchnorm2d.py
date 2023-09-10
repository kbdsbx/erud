from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np
from erud._utils import epsilon as eps

class batchnorm2d(payload) :

    __x : any = None
    __mean : np.ndarray = None
    __var : np.ndarray = None

    # 训练时记录移动平均
    # 测试时直接使用
    __moving_mean : np.ndarray = None
    __moving_var : np.ndarray = None

    # 是否处于训练模式
    __is_training : int = None

    # 移动平均动量
    __momentum : float = None

    __s : int = None
    __m : int = None
    __n : int = None

    def __init__ (self, is_training = 1, momentum = 0.999) :
        self.__is_training = is_training
        self.__momentum = momentum
        return 

    def fprop(self, x) -> any :
        if not self.__is_training :
            return (x - self.__moving_mean) / self.__moving_var

        (s, m, n, c) = x.shape

        _mean = 1. / (s * m * n) * np.sum(x, axis = (0, 1, 2), keepdims = True)
        _var = 1. / (s * m * n) * np.sum(np.power(x - _mean, 2), axis = (0, 1, 2), keepdims = True)

        self.__x = x
        self.__mean = _mean
        self.__var = _var
        self.__s = s
        self.__m = m
        self.__n = n

        if self.__moving_mean is None :
            self.__moving_mean = np.zeros((1, 1, 1, c))
        if self.__moving_var is None :
            self.__moving_var = np.ones((1, 1, 1, c))
        
        # 记录均值和方差的移动平均
        self.__moving_mean = self.__momentum * self.__moving_mean + (1. - self.__momentum) * self.__mean
        self.__moving_var = self.__momentum * self.__moving_var + (1. - self.__momentum) * self.__var

        assert _mean.shape == (1, 1, 1, c)

        return (x - _mean) / np.sqrt(_var + eps)
    
    def bprop(self, dz) -> list[any] :

        _x = self.__x
        _mean = self.__mean
        _qvar = np.sqrt(self.__var + eps)
        _s = self.__s
        _m = self.__m
        _n = self.__n

        dx = (dz / _qvar) - (np.sum(dz, axis = (0, 1, 2), keepdims = True) / (_s * _m * _n) / _qvar) - ((np.sum(dz * (_x - _mean), axis = (0, 1, 2), keepdims = True) / (_s * _m * _n) / np.power(_qvar, 3)) * (_x - _mean))

        return [dx]
    
    # 导出
    def exports(self) -> object :
        name = (super(batchnorm2d, self).exports())["name"]

        exp = {
            'type' : 'batchnorm2d',
            'name' : name,
            'mean' : self.__moving_mean.tolist(),
            'var' : self.__moving_var.tolist(),
        }

        return exp
    
    # 导入
    def imports (self, value) : 
        super(batchnorm2d, self).imports(value)

        self.__moving_mean = np.array(value['mean'])
        self.__moving_var = np.array(value['var'])

        