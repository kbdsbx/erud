

class updateable :

    # 初始化更新方法
    def __init__ (self) : ...

    # 待继承的更新方法，通过学习率和梯度更新参数
    def updateFunc(self) : ...

    # 导出更新方法的必要数据
    def exports(self) -> object : ...

    # 从缓存中导入更新方法的必要数据
    def imports(self, value : object) : ...