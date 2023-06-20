from erud.cg.payload import payload
import numpy as np

class matmul (payload) :
    # 缓存
    __x : np.ndarray
    __y : np.ndarray

    # 前向传播
    def fprop(self, x : np.ndarray, y : np.ndarray) -> np.ndarray:
        self.__x = x
        self.__y = y

        return np.matmul(self.__x, self.__y)
    
    # 反向传播
    def bprop(self, dz) -> list[np.ndarray] :
        _x = self.__x
        _y = self.__y

        _y_temp_axis = [i for i in range(len(_y.shape))]
        if len(_y.shape) == 1 :
            _y_temp_axis = tuple([1, 0])
            _y_temp = _y.reshape((_y.shape[0], 1))
        else :
            t = _y_temp_axis[-1]
            _y_temp_axis[-1] = _y_temp_axis[-2]
            _y_temp_axis[-2] = t
            _y_temp = _y
        
        # 按照矩阵后两个维度转置
        _y_temp = _y_temp.transpose(*_y_temp_axis)

        dx = np.matmul(dz, _y_temp)
        dx = dx.reshape(_x.shape)

        _x_temp_axis = [i for i in range(len(_x.shape))]
        if len(_x.shape) == 1 :
            _x_temp_axis = tuple([1, 0])
            _x_temp = _x.reshape((_x.shape[0], 1))
        else :
            t = _x_temp_axis[-1]
            _x_temp_axis[-1] = _x_temp_axis[-2]
            _x_temp_axis[-2] = t
            _x_temp = _x
        
        _x_temp = _x_temp.transpose(*_x_temp_axis)

        dy = np.matmul(_x_temp, dz)
        dy = dy.reshape(_y.shape)

        return [dx, dy]
