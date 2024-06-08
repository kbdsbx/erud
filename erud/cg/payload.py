
# 节点的负载类，所有计算图中参与计算的模块的父类
class payload : 
    # 运载类节点的名称，可以给任何节点命名以便于在计算图中寻找此节点
    __name : any = None

    # 负载的固有属性，用来表示此负载将使用什么传播类型进行前向/反向传播的
    # 传播类型
    # 0 为分发类型，为默认值
    #   在此类型下，变量沿前向传播路径传递时如果遇到多条出边，则使用分发规则，即将自身复制多份分别分发
    #   反向传播则视所有入边为原变量的偏导，计算原变量的导数时只需要将所有偏导累加
    # 1 为拆分类型
    #   在此类型下，变量将在此处被拆分成n份自变量，各不相同，不同的出边接受不同的子变量。具体拆分规则由操作符处理
    #   反向传播则将每个入边的导数都视为子变量的导数，计算原变量的导数时需要将子变量的导数合并，而非求和
    __prop_type : int = 0

    @property
    def name (self) -> any:
        return self.__name
    
    @property
    def prop_type (self) -> any:
        return self.__prop_type
    
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

    # 导出方法
    # 任何需要导出值的子类需要实现的方法，用户计算图将运行中的临时值以JSON的方式保存在文件中
    # 不需要导出数值的子类不需要实现此方法
    # 导出方法除了能够用在导出文件上，也可以用在迁移上
    def exports(self) -> any : 
        obj = {
            'class' : str(self.__class__.__name__)
        }
        if self.__name is not None :
            obj['name'] = self.__name
        return obj
    
    # 导入方法
    # 任何需要导入值的子类需要实现的方法，从用户文件缓存中读入值来还原计算图
    # 不需要导入数值的子类不需要实现此方法
    def imports(self, value) :
        if 'name' in value :
            self.__name = value["name"]

    def __str__ (self) -> str:
        return self.name

