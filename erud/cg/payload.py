
# 节点的负载类，所有计算图中参与计算的模块的父类
class payload : 
    # 运载类节点的名称，可以给任何节点命名以便于在计算图中寻找此节点
    __name : any

    @property
    def name (self) -> any:
        return self.__name
    
    @name.setter
    def name(self, n) -> any:
        self.__name = n
        return n
    
    # 亟待子类实现的前向传播方法
    # 提供多个变量，提供变量的个数取决于指向此节点的路径的个数
    # 返回一个变量，通常为提供给下一层节点的计算值
    # 变量类型可以使标量、矢量、矩阵、张量等
    def fprop () -> any : ...

    # 亟待子类实现的反向传播方法
    # 提供一个变量，通常为上一层传下来的导数值
    # 返回多个变量，返回变量的数量等同于前向传播参数数量，不同参数会沿着不同路径一一对应向后传递
    # 变量类型可以使标量、矢量、矩阵、张量等
    def bprop () -> list[any] : ...

