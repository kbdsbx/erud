# Erud
This is a tiny machine learning framework, but I don't know for sure as what it will become.

# 构建计算图

### 术语辨析
1. 用户：使用此框架的开发者
1. 客户：使用框架的一般人
1. 数据：用户或客户传入的样本数据，可能为数量、向量、矩阵或张量等
1. 参数：用户传入的数据，可能为数量、向量、矩阵或张量等
1. 变量：用户看不见的数据，隐藏在实现细节里

### 计算图的特征

1. 叶节点：存储张量的节点
    * 固定参数，包括数据和超参数
    * 学习参数

    > 所以需要一个变量区分参数类型，或者说是否允许更新

2. 非页节点：存储操作和缓存
    
    组图阶段只存储“操作”，留一个或者两个缓存单位空着，正向传播的时候顺便计算要反向传播的变量，然后存在单位里，等反向传播的时候取出来用

    * 基本二元运算
        * 加
        * 减
        * 乘
        * 除
        * 矩阵乘
        * ...
    * 复合二元操作
        * 卷积
        * 交叉熵
        * ...
    * 基本一元运算
        * 次方
        * ...
    * 复合一元操作
        * relu
        * tanh
        * 修改变量的维度
        * ...
    * 复合三元操作
        * 也许有...
    
    > 一开始可以搞点基本操作组计算图，组完了再加复合操作做后期优化

    > 复合操作可以由基本操作组合，或者用户自己往里加，加的时候注意传入参数和计算反向传播就行

3. 边：双向边

    虽然计算图对前向传播来说是一个有向无环图，但毕竟反向传播还是要用边来找前向节点的，所以往图里塞节点的时候需要给后面的节点加前节点的指向

    |节点类型|前向入度|前向出度|
    |-|-|-|
    |叶节点|0|n，看参与的计算数量|
    |一元操作节点|1|1|
    |二元操作节点|2|1|
    |三元操作节点|3|1|
    |根节点，一般是损失函数|n，看参与的参数量|0|

    反向出入度和前向相反

3. 组图

    搭建图：

    手动搭建“图”类型的数据结构，以图插入节点的方式构建图，接口的话可以设计成图插入，可以是普通的方式，也可以是运算符重载，前期可以只用图插入的方式测试，等前后向传播写完了再扩充接口

    如果以前向传播的有向无环图作为搭建基准，那么
    1. 准备叶结点（数据）
    2. 根据出度插入多个操作节点
    3. 操作节点中会带上叶结点（参数）
    4. 操作节点后续插入多个操作节点，直到根节点


4. 前向传播

    前向传播使用有向无环图的拓扑排序作为遍历算法，算法如下

    1. 找到图中入度为0的节点，如果节点为叶节点则向后传递数据，如果节点为操作节点则计算操作、存储缓存数据、向后传递数据
    2. 将此节点从图中标记删除（不是真删），删除后的图中会产生新的入度为0的节点
    3. 重复1，直到标记删除根节点或图里节点全部删除（两个条件通常都一样）

    前向传播算法通常是能够访问整个图的算法，可以是图类的方法
    
    前向传播的调用条件是所有叶结点都有数据填充，包括表示样本的叶结点和表示参数的叶结点，后者通常为固定值（超参数）或随机值（参数）

5. 验证

    图在搭建完成后可以有验证方法进行验证，主要验证目标是针对各个张量的计算是否合法，验证算法模拟前向传播的流程，但只做计算的断言，而不传递真实值

6. 反向传播

    反向传播是全自动计算算法，通常在前向传播输出代价值后由用户显示调用，反向传播遵从导数计算法则：
    
    * 单路全导
    * 多路偏导
    * 分叉相加
    * 分段相乘

    其中需要参数会像前向传播一样，沿着边传递给上一个计算节点，计算节点拿到上一个参数以后会先计算本节点的导数，然后和之前的导数相乘（单路）或相乘相加（多路），继续传递给下一个节点

7. 环

    <b style="color: red">循环神经网络是否存在环</b>，如果存在环的话还能不能正常工作，这个需要验证，但至少从基础结构来说，目前实现的图是允许环的存在的


> 第一阶段就是构建整个计算图，增加简单四则运算作为操作节点，然后完成前向和反向传播。测试用例要包括变量、向量和张量，其中验证功能可以放在最后加


计算图节点node含有属性data，类型为payload，所有张量类和操作符类都继承自payload，并实现fprop方法和bprop方法，其中fprop会在计算图执行前向传播时陆续调用，bprop会在执行反向传播时调用。

计算图边edge含有属性carry，类型为标量、向量、矩阵或张量，正在计算的节点会从入度边中一次性取得所有运载并进行计算，已计算完成的节点node会将值分发给出度中所有边的运载

### 命令和接口

单元测试
```bush
> .\test.bat
```

计算图的使用接口（目前的接口）

引入
```python
from erud.cg.graph import ComputationGraph as graph
from erud.cg.node import ComputationNode as node
```

新建节点
```python
n1 = node()
n2 = node()
```

插入/删除节点
```python
g = graph()
g.insertNode(n1)
g.insertNode(n2)

g.removeNode(n2)
```

添加/删除边
```python
g.addEdge(n1, n2)

g.removeEdge(n1, n2)
```

### 目前完成的功能

1. 计算图的添加删除节点
2. 计算图的添加删除边
3. 张量负载：常量、变量
4. 二元操作负载：加、减、乘、除