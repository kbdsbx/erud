from erud.cg.payload import payload
from erud._utils import useGPU
if useGPU :
    import cupy as np
else :
    import numpy as np
from erud.upf.updateable import updateable
from erud.upf.norm import norm
from erud.upf.momentum import momentum 
from erud.upf.adam import adam
from inspect import isfunction

class var (payload) :
    __data : any = None
    __update_func : updateable = None
        
    __update_func_list = {
        'norm' : norm,
        'adam' : adam,
        'momentum' : momentum
    }

    @property
    def data (self) -> any :
        return self.__data
    
    
    # 变量赋值
    @data.setter
    def data(self, d) :
        self.__data = d
    
    # 更新函数
    # 是否拥有更新函数通常被用来判断节点是否是常量
    @property
    def update_func(self) -> any:
        return self.__update_func
    
    @update_func.setter
    def update_func(self, d) :
        self.__update_func = d

    # d: 负载
    # update_fun(z, dz) : 参数的更新方法
    def __init__ (self, d, update_func = None) :
        self.__data = d
        self.__update_func = update_func
    
    # 前向传播时，变量提供值
    # 变量值会向后分发（沿着出度）给所有使用此变量的表达式
    def fprop(self) -> any:
        return self.__data

    # 反向传播时，dz/dc = 0
    def bprop(self, dz = None) -> list[any]:
        # 反向传播更新参数
        if dz is not None and self.__update_func is not None :
            if isinstance(self.__update_func, updateable) :
                res = self.__update_func.updateFunc(self.__data, dz)
            if isfunction(self.__update_func) :
                res = self.__update_func(self.__data, dz)
            if res is not None :
                self.__data = res

        if isinstance(self.__data, np.ndarray) :
            return [np.zeros_like(self.__data)]
        else :
            return [0]
    
    def __str__ (self) :
        if self.__data is not None :
            return str(self.__data)
        else :
            return ""
    
    # 导出
    def exports (self) -> object:
        name = (super(var, self).exports())["name"]

        exp = {
            'class' : 'var',
            'name' : name,
        }

        # 只导出具有更新方法的学习参数，没有更新方法的固定参数和样本不导出
        if self.__update_func is not None :
            if isinstance(self.__data, np.ndarray) :
                exp['data'] = self.__data.tolist()
            else :
                exp['data'] = self.__data
            
            exp['updateable'] = self.__update_func.exports()
        
        return exp
    
    # 导入
    def imports (self, value) :
        super(var, self).imports(value)

        if 'data' in value :
            v = value['data']
            if isinstance(v, list) :
                self.__data = np.array(v)
            else :
                self.__data = v
        
        # 更新方法导入
        if 'updateable' in value :
            ob = value['updateable']
            self.__update_func = self.__update_func_list[ob['class']](ob['rate'])
            self.__update_func.imports(ob)
        
    

