# Erud

为什么叫Erud？因为此框架受博识尊（Erudition）指引，践行智识（Nous）命途。

# 构建计算图

计算图（Computation Graph）是神经网络的核心，用于构建神经网络计算的脚手架

## 术语辨析
1. 用户：使用此框架的开发者
1. 客户：使用框架的一般人
1. 数据：用户或客户传入的样本数据，可能为数量、向量、矩阵或张量等
1. 参数：用户传入的数据，可能为数量、向量、矩阵或张量等
1. 变量：用户看不见的数据，隐藏在实现细节里
1. 控制参数：不参与计算图运算，只是传递给操作符或者初始化函数来改变运算规则

## 计算图的特征

1. 叶节点：存储张量的节点
    * 固定参数，包括数据和超参数
    * 学习参数

    > 所以需要一个变量区分参数类型，或者说是否允许更新

2. 非页节点：存储操作和缓存
    
    组图阶段只存储“操作”，留一个或者两个缓存单位空着，正向传播的时候顺便计算要反向传播的变量，然后存在单位里，等反向传播的时候取出来用

    * 基本二元运算
        * 加add（完成）
        * 减sum（完成）
        * 乘mul（完成）
        * 除div（完成）
        * 矩阵乘matmul（完成）
        * ...
    * 复合二元操作
        * 卷积（完成）
        * 交叉熵cross_entropy（完成）
        * L1（完成）
        * L2（完成）
        * ...
    * 基本一元运算
        * 次方（用其他方式代替）
        * 张量求和sum（完成）
        * 总代价cost（完成）
        * ...
    * 复合一元操作
        * relu（完成）
        * sigmoid（完成）
        * tanh（完成）
        * softmax（完成）：softmax需要另一个操作数轴，所以虽然是一元操作，但在nous中的调用方法为``-> softmax 1 ->``或``-> softmax [0,1] ->``
        * <del>dropout（完成）: dropout需要一个概率值表示单元失效概率，在0~1之间`-> dropout 0.5 ->`</del>
        * batchnorm（完成）
        * 最大池化max_pool（完成）
        * 修改变量的维度flatten（完成）
        * ...
    * 复合三元操作
        * 也许有...
    * 带控制参数的一元或多元操作
        * softmax（完成）：softmax需要一个元组来区分样本集合`X`中，每个样本延哪个轴存放，并根据存放的轴号进行概率计算，调用方法为`-> softmax((1)) ->` 或`-> softmax((0, 1)) ->`
        * dropout（完成）: dropout需要一个概率值表示单元失效概率，在0~1之间，调用方法改为`-> dropout(0.5) ->`
        * softmax_cross_entropy（完成） : `softmax`及其交叉熵的复合运算符，调用方法为`-> softmax_cross_entropy(1) Y ->`
    
    > 一开始可以搞点基本操作组计算图，组完了再加复合操作做后期优化

    > 复合操作可以由基本操作组合，或者用户自己往里加，加的时候注意传入参数和计算反向传播就行

3. 边：双向边

    <del>虽然计算图对前向传播来说是一个有向无环图</del>（见条目8），但毕竟反向传播还是要用边来找前向节点的，所以往图里塞节点的时候需要给后面的节点加前节点的指向

    |节点类型|前向入度|前向出度|
    |-|-|-|
    |叶节点|0|n，看参与的计算数量|
    |一元操作节点|1|1|
    |二元操作节点|2|1|
    |三元操作节点|3|1|
    |根节点，一般是损失函数|n，看参与的参数量|0|

    反向出入度和前向相反

4. 组图

    搭建图：

    手动搭建“图”类型的数据结构，以图插入节点的方式构建图，接口的话可以设计成图插入，可以是普通的方式，也可以是运算符重载，前期可以只用图插入的方式测试，等前后向传播写完了再扩充接口

    如果以前向传播的有向无环图作为搭建基准，那么
    1. 准备叶结点（数据）
    2. 根据出度插入多个操作节点
    3. 操作节点中会带上叶结点（参数）
    4. 操作节点后续插入多个操作节点，直到根节点


5. 前向传播

    前向传播使用有向无环图的拓扑排序作为遍历算法，算法如下

    1. 找到图中入度为0的节点，如果节点为叶节点则向后传递数据，如果节点为操作节点则计算操作、存储缓存数据、向后传递数据
    2. 将此节点从图中标记删除（不是真删），删除后的图中会产生新的入度为0的节点
    3. 重复1，直到标记删除根节点或图里节点全部删除（两个条件通常都一样）

    前向传播算法通常是能够访问整个图的算法，可以是图类的方法
    
    前向传播的调用条件是所有叶结点都有数据填充，包括表示样本的叶结点和表示参数的叶结点，后者通常为固定值（超参数）或随机值（参数）

6. 验证

    图在搭建完成后可以有验证方法进行验证，主要验证目标是针对各个张量的计算是否合法，验证算法模拟前向传播的流程，但只做计算的断言，而不传递真实值

7. 反向传播

    反向传播是全自动计算算法，通常在前向传播输出代价值后由用户显示调用，反向传播遵从导数计算法则：
    
    * 单路全导
    * 多路偏导
    * 分叉相加
    * 分段相乘

    其中需要参数会像前向传播一样，沿着边传递给上一个计算节点，计算节点拿到上一个参数以后会先计算本节点的导数，然后和之前的导数相乘（单路）或相乘相加（多路），继续传递给下一个节点

8. 环

    <b style="color: red">循环神经网络是否存在环</b>，如果存在环的话还能不能正常工作，这个需要验证，但至少从基础结构来说，目前实现的图是允许环的存在的


    > 第一阶段就是构建整个计算图，增加简单四则运算作为操作节点，然后完成前向和反向传播。测试用例要包括变量、向量和张量，其中验证功能可以放在最后加

9. 一些细节

    计算图节点node含有属性data，类型为payload，所有张量类和操作符类都继承自payload，并实现fprop方法和bprop方法，其中fprop会在计算图执行前向传播时陆续调用，bprop会在执行反向传播时调用。

    计算图边edge含有属性carry，类型为标量、向量、矩阵或张量，正在计算的节点会从入度边中一次性取得所有运载并进行计算，已计算完成的节点node会将值分发给出度中所有边的运载


    目前版本为两个节点增加重复边会抛出异常，即对x和y来说，x -> y只可以存在一条边（但可以同时存在x -> y和y -> x）。但考虑到如果有一个节点x，执行x * x运算，那么数据节点x指向操作节点*的边可以有多条。所以边的限制将在后续版本删除

10. 复杂图

    最开始设想计算图是有向无环图简单图，后来觉得无环的限制可能不适用于循环网络或者递归网络，现在觉得简单图可能不适用于节点的灵活性，所以计算图的类型应该是……额……图。

    对于复杂图，两个之间的节点可以有不止一条边，比如对于表达式``X + X``来说，节点``X``可以指向加法节点``+``两次，存在两条边且不影响目前的计算图计算逻辑，只需要开放一下限制即可

    环是一个例外，环可以满足类似于``X = X + 1``的效果，也就是节点``X``与节点``1``相加后又指向了节点``X``，这对计算图是一个挑战，因为无环图可以通过拓扑排序的遍历算法一次性处理完所有节点，但一旦计算图中存在环，现有的拓扑排序将是死循环。

    一种方法是将环独立于计算图之外，让用户去处理循环网络的事情，保持计算图的可计算性，另一种方法就是规定循环的次数，但这样无疑是将整个图的处理变得复杂，在没有证据证明图中带有环能够使计算图的表达更加高效，更让人接受之前，还是先这样吧。



> 现在这个接口用起来真是难受死了，得像个办法出一套简单点的接口

### 调试

目前计算图还没有调试相关的内容，一个可能的想法是添加调试状态标记，并在调试状态下缓存/打印前向传播和反向传播的值

调试需要提上日程，softmax分类测试用例实在是太慢了，需要添加额外的功能来记录计算中数据流经过各个节点的总时间，找到计算图中最花费时间的关键运算，然后才能尝试优化

# 计算图的构建接口

> 既然是图，那就用图的方式构建好了

> 需要一个类似于SQL的结构化语言帮助我快速创建计算图，如果像`test\test_graph.py`里的测试用例那样创建计算图我会疯的


## nous

nous 是一套格式化语言，可以帮助各位走上智识的命途，领略星神博识尊践行的命运真谛（计算图快速搭建语言）

示例：

```python
graph = '''
X matmul W1 then relu then matmul W2 then cross_entropy Y as J
W2 then sqr then sum as C
W1 then sqr then sum then plus C then times LAMBDA then plus J then REST
'''

# corss_entropy(sigmoid(relu(relu(relu(relu(X * W1 + B1) * W2 + B2) * W3 + B3) * W4 + B4)))
graph = '''
X:(1000, 10) matmul W1:r(10, 50) -> add B1:z(40) -> relu -> matmul W2:z(40, 20) -> add B2:z(20) -> relu -> matmul W3:r(20, 10) -> add B3:z(10) -> matmul W4:r(10, 1) -> add B4:z(1) -> sigmoid -> cross_entropy y:(1000) -> J:$$
'''

# rest = [(5 + 10) * (6 - 19) / 3 ] / (-65) - 1
graph = '''
5 add 10 as u1
6 sub 19 as u2
u1 mul u2 then div 3 then div -65 then sub 1 then REST
'''

graph = '''
# 这里是注释

# 下面两个各是一行
5 add 10 as u
6 sub 19 as v

    # 下面两个是一行
    u mul v ->
    div 3 as t

# 结束
t -> y:$$
'''
```

### 语法规则

1. 声明变量
    * ``X``: 声明一个变量，命名为'X'，不赋值
    * ``:5`` 或 ``5``: 声明一个匿名变量，赋值``5``。如果没有冒号``':'``，当声明不满足于变量命名规范时视为值，否则视为变量
    * ``X:5``: 声明一个变量，命名为'X'，并赋值``5``
    * ``X:randn(10, 15)``: 声明一个张量变量，维度为(10,15)，命名为'X'，并随机赋值
    * ``X:zero(10, 15)`` 或 ``X:(10,15)``: 声明一个张量变量，维度为(10,15)，命名为'X'，并赋值为0
    * ``X:ones(10, 15)``: 声明一个张量变量，维度为(10,15)，命名为'X'，并赋值为1
    * ``X:he((10, 15), 10)``: 声明一个张量变量，维度为(10,15)，命名为'X'，使用`he`进行初始化
    * ``X:[[1,2], [3,4]]``: 声明一个张量变量，维度为(2,2)，命名为'X'，并赋值为`[[1,2],[3,4]]`

    目前已有的初始化函数
    1. `zero(<tuple>)`或`(<tuple>)`: 执行零初始化
    2. `ones(<tuple>)`: 执行1初始化
    3. `randn(<tuple>)`: 执行正态随机初始化
    4. `he(<tuple>, layer_num)`: 执行he初始化

2. 运算符操作

    操作有默认操作和扩展操作两种，其中默认操作为框架内置操作，包括加减乘除、矩阵乘、relu、sigmoid、softplus、softmax、maxout、tanh、rbf、cross_entropy等

    > 其中部分随后实现

    > 未来已来，实现已现

    扩展操作为用户添加操作，高级用户可在默认操作的基础上扩展自己的操作，只需要继承``erud.cg.payload``类，声明关键词，实现``fprop``和``bprop``方法即可

    扩展操作可以覆盖默认操作，如果用户想要替换默认操作，比如想要替换cross_entropy操作，可以自行声明一个类``class cross_entropy (payload)``并将关键词改为``cross_entropy``，框架在装载操作符时就会将用户声明的``cross_entropy``操作替换掉内部默认的``cross_entropy``操作

    具体步骤参考“高级用户的高级玩法”

    * ``5 add 10``: 包含左右操作数的计算声明，操作数为任意类型变量，但用户需要自己检查变量是否满足操作运算的规则（例如矩阵乘法）
    * ``add W1``: 只包含一个操作数的计算声明，如果操作符需要两个操作数，则左操作数为上一层的结果
    * ``relu``: 没有操作数的计算声明，如果操作符需要一个操作数，则左操作数为上一层的结果
    * ``dropout(0.5)``: 带参数的操作符

3. 层操作

    层操作并不严格定义层，只定义计算图中节点与节点之间的前后关系

    * ``then`` 或 ``->``: 层分隔符，运算时将左边式子的计算结果传递给右边式子的左操作数

4. 赋值操作

    赋值操作并不产生新的节点，只修改原本节点的命名，以便可以在计算图中按名称找到此节点
    
    * ``a add b as res``: 将加法操作``add``重命名为`res`

5. 声明结果变量

    计算图中可以有一个或多个结果变量，结果变量存放结果值，计算图的前向传播`fprop`会以数组的方式返回所有结果，结果变量为计算图或子图的汇点

    匿名变量只能由`fprop`获得，命名变量也可以由查找节点的方法`getData`在前向传播完成后取得对应的值

    * ``... -> rest``或``... -> $$``: 声明匿名结果变量节点
    * ``... -> J:rest``或``... -> J:$$``: 声明结果变量节点，并命名为``J``

6. 声明子块

    * ``(a add b) then mul c``: 小括号用于声明子块
    * 使用换行来声明子块，每一行为一个子块
        ```
        5 add 10 as u1
        6 sub 19 as u2
        u1 mul u2 then REST
        ```

7. 换行

    * 当尾部为层连接符时，此行与下一行属于同一个子块
        ```
        # 前五行为同一子块
        X ->
            matmul W1 add B1 -> relu ->
            matmul W2 add B2 -> relu ->
            matmul W3 add B3 -> relu ->
        sigmoid as u
        # 最后一行为一子块
        u cross_entropy y -> J:$$
        ```

8. 注释
    * 以`#`开始的行是注释

9. 引用
    
    代码会从上到下从左到右依次扫描，其中第一次声明的变量或操作符将会被新建，而后续找到的所有同名变量和操作符会被视为第一个变量的引用，指向同一个节点
    
    根据这个特性，可以创建`X mul X`的代码，使用乘法代替平方

    目前为止，代码没有作用域和子域的功能，所有变量和操作符在同一计算图中平等存在，任何引用都会全局搜索计算图节点

### 写代码的时候注意空格！
符号不能作为区别代码元素的字符，空格才能，以下代码是错误的

* `X->Y` 或 `X addY` : 中间没有空格分隔会把一整块代码解释成一个变量
* `X: [[1, 2], [3, 4]]` 或 `X: (1, 2)` : 冒号后面的空格会把左右两个都解释成变量，而且是中间没有操作符的错误格式

### 操作符没有优先级！
当`add`和`mul`同时出现时，并不会优先计算`mul`而是根据出现先后计算，这是因为任何操作符，包括框架内置的操作符都可以被覆写，所有操作符平级，想要计算优先级请使用子块
* `X add Y mul Z`: 优先计算`add`，然后计算`mul`
* `X add (Y mul Z)` : 优先计算`mul`，然后计算`add`

### 存储与还原

计算中的计算图可以使用`g.exports(<filename>)`方法将计算图的参数和一些信息保存在json格式的文件中，并使用`g.imports(<filename>)`方法读取。

此方法仅顺序保存各个节点中的样本或权重数据，不保存运算节点间的关系、参数节点采用的初始化方法和更新方法及其他结构的数据，这些内容请直接保存`nous`代码并使用此代码还原计算图结构。

此功能为长期深度学习程序所准备，但是由于本框架垃圾的运行效率（尤其是卷积和池化），为了保证您训练神经网络时的身心愉悦，请使用TensorFlow、PyTorch等其他框架进行长期训练和业务实现。


## 高级用户的高级玩法

### 自定义操作符

目前计算图中包括的二元运算符有`add`，`sub`，`mul`，`div`，`matmul`等，如果你是高级用户，想要在计算图中添加自定义运算符来创建更丰富的计算图或者优化计算流程，那么你可以这样做：

1. 实现运算符类，并实现fprop和bprop方法

    ```python
    from erud.cg.payload import payload

    class you_opt (payload) :
        
        def fprop (self, <you want args>) :
            # some computations.
            return <value>
        
        def bprop (self, dz) -> list[any]:
            # some computations.
            return [<values>]
    ```

    注意事项：
    1. 前向传播`fprop`可以接受多个参数，通常取决于你的操作符是几元运算符。返回一个结果，通常为计算结果。
    2. 反向传播`bprop`接受一个参数，也就是整个计算子图最终结果`J`对此操作的结果`z`的导数`dJ/dz`，返回一个或多个结果。如果你的操作是二元操作，前向传播有`x`和`y`两个参数，那么确保按顺序返回偏导数`[dx, dy]`，并保证`dx`和`x`具有相同的维度，这些偏导数会沿着前向传播的路径返回，保证反向传播的正确性。
    3. 保证你前向传播和反向传播的参数兼容你使用的所有格式参数，比如向量、矩阵或张量，保证每一种参数或者不同参数传过来后依旧能够计算出正确的结果。
    
    操作符的反向传播编写是一个颇有挑战性的工作，尤其是涉及到矩阵或张量参数的梯度运算。谨慎计算，随时准备好单元测试，或者尝试使用已有的运算符进行搭建。

2. 将运算符注册到计算图（暂时不支持全局注册）

    ```python
    g = group()
    g.registerOperator('you_opt', you_opt)
    ```

3. 在计算图中使用你的运算符

    ```python
    g = nous(
        '''
        X you_opt Y add Z -> $$
        '''
    ).parse()
    ```

4. 覆盖掉已有的运算符

    什么？你觉得我写的不好用？你他……山之石可以攻玉，你觉得对。

    注册一个同名运算符会覆盖掉默认或者之前注册的运算符，覆盖会重复发生，只保留最新的一个，比如
    ```python
    g.registerOperator('add', my_add_class)
    ```
    将会把默认运算符`add`替换为自定义的类，并在计算图计算时调用自定义的传播方法

### 自定义初始化函数

目前变量的初始化函数只有`''`，`zeros`，`ones`，`randn`四种，你可以添加你自己的初始化函数

1. 编写自己的初始化方法，例如想要从数据库中取出数据

    ```python
    def my_init_func( args ) :
        ar = get_data_from_database( '<SQL>' )
        return np.array(ar)
    ```

2. 将初始化方法注册到图

    ```python
    g.registerInitFunc('my_init_func', my_init_func)
    ```

3. 使用初始化方法初始化变量值，比如样本数据

    ```python
    g = nous(
        '''
        X:my_init_func(<some args>) add Y -> Z:$$
        '''
    ).parse()
    ```
    
4. 自定义方法可以覆盖已有的方法或者内置方法

    注意事项：
    1. 确保初始化方法的返回值能够正确参与运算
    2. 参数格式要满足一般表达式格式，比如不能出现半个括号，这样就会解析出错
    3. 命名为空的字符串`''`也是一个初始化函数，在执行`X:(5, 3, 4)`时调用，并可以被`g.registerInitFunc('', my_init_func)`覆盖
    4. 函数参数只允许类型为标量、数组、元组类型的参数，不允许字符串及其他类型的参数，例如`X:my_init_func(abc)`是非法的，而`X:my_init_func(5.5, [1, 2], (1,2, 3))`是合法的，将会把参数`[5.5, [1,2], (1,2,3)]`传入自定义方法`my_init_func`



# 命令和接口

### 安装

还没有打包发布，暂时没有安装命令

### 单元测试

```bush
> .\test.bat
```

### 计算图的使用接口

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

> 忘掉上面的接口吧，新建节点再连线是大猩猩做的事

### 使用nous格式化语言新建计算图

```python
from erud.nous import nous

# 使用nous类解析nous脚本，一键创建复杂计算图，妈妈再也不用担心我的神经网络搭建
n = nous(
    '''
    X matmul W1 add B1 -> relu ->
      matmul W2 add B2 -> relu ->
      cross_entropy Y as J_mle

    W2 -> sqr -> sum as u7
    W1 -> sqr -> sum -> add u7 -> mul LAMBDA as u8

    J_mle add u8 -> J:$$
    '''
)

g = n.parse()
```

根据名称赋值
```python
g.setData('X', np.random.randn(4,3,2))
```

根据名称获取值
```python
# 获取参数W1
w1 = g.getData('W1')
# 获取运算结果J
j = g.getData('J')
```

### 设置学习参数的自动更新函数

```python
rate = 0.01
g.setUpdateFunc('W1', lambda z, dz : z - rate * dz)
```
或者使用框架内置更新函数
```python
import erud.upf as upf
rate = 0.01
g.setUpdateFunc('W1', upf.norm(rate))
```

框架目前内置了三种更新函数
1. `upf.norm(rate)`: 普通的更新函数
2. `upf.momentum(rate, beta)`: 史诗的更新函数，使用加权平均进行偏差修正，能够增加梯度下降速度
3. `upf.adam(rate, beta_momentum, beta_rms)`: 传说的更新函数，不仅使用加权平均进行偏差修正，还使用平方平均根对不同方向上的参数更新量进行调整，能大幅增加梯度下降速度（但也可能让梯度下降跳入局部最小值出不来）

### 目前完成的功能

1. 计算图的添加删除节点
2. 计算图的添加删除边
3. 张量负载：常量、变量
4. 二元操作负载：加、减、乘、除、矩阵（张量）乘法等
5. 完成计算图前向传播与反向传播
6. 实现简化版本的深度学习测试用例`\test\test_graph.py::test_bprop`
7. 规范化nous语言格式
8. 实现nous语言解释器，能够将nous代码解释为计算图
9. 自定义操作符
10. 自定义初始化类
11. 更新函数
12. 深度学习之逻辑回归（浅层神经网络）`\test\test_logistic_regression.py`
12. 深度学习之深层神经网络`\test\test_deep_neural_network.py`
12. 深度学习之L2/dropout正则化`\test\test_regularization.py`
12. 深度学习之mini-batch`\test\test_mini_batch.py`

# 示例

### 浅层神经网络

感谢吴恩达深度学习课程课后实现提供的数据集，全部代码见 `\test\test_logistic_regression.py`

浅层神经网络是一种简单的神经网络，只有一层参数层以及一层激活层，使用交叉熵函数计算代价

训练阶段：

```python
# 迭代次数
num_iterations = 1000
# 学习率
rate = 0.005

# 每个图片样本有为64 * 64 * 3 = 12288个参数，所以W被初始化为(1, 12288)
# 一共有209个样本
# 偏置参数b被初始化为0，所有样本共有一个偏置
# 使用交叉熵函数与标签集Y计算各个样本代价，并由cost函数计算所有样本总代价
# 最终代价存储在终止符J中
g = nous('''
W:zeros(1, 12288) matmul X:(12288, 209) add b:0 -> sigmoid as temp -> cross_entropy Y -> cost -> J:$$
''').parse()

# 给样本X赋值
g.setData('X', train_set_x)
# 给标签Y赋值
g.setData('Y', train_set_y_orig)
# 更新参数的方法，w := w- rate * dj/dw；b := b - rate * dj/db
g.setUpdateFunc('W', lambda w, dw : w - rate * dw)
g.setUpdateFunc('b', lambda b, db : b - rate * db)

# 循环计算n次
for i in range(num_iterations) :

    # 先执行前向传播
    g.fprop()
    # 再执行反向传播
    g.bprop()
    # 反复多次

    if i % 100 == 0 :
        print('Cost after iteration %i : %f' % (i, g.getData('J') ) )
```

验证阶段：

验证阶段可以新建一个计算图，并把之前计算图中的参数传递给新的计算图进行计算，也可以在老的计算图上进行计算

```python
# 新建一个计算图
# 此计算图只为了计算神经网络准确率，故而使用了非正规操作threshold和accuracy，这两个操作都不实现反向传播
# threshold阈值操作一般跟在sigmoid后边，大于阈值为1，小于阈值为0
# accuracy计算yhat与y之间数据相同的比例，目前只能计算逻辑回归函数的值，有待进一步完善
g1 = nous('''
W matmul X add b -> sigmoid -> threshold(0.5) -> accuracy Y -> J:$$
''').parse()

# 将图g中学习好的参数传入新的计算图g1
g1.setData('W', g.getData('W'))
g1.setData('b', g.getData('b'))

# 计算训练精准率
g1.setData('X', train_set_x)
g1.setData('Y', train_set_y_orig)
g1.fprop()
print('train accuracy: %s' %(g1.getData('J')))

# 计算测试精准率
g1.setData('X', test_set_x)
g1.setData('Y', test_set_y_orig)
g1.fprop()
print('test accuracy: %s' %(g1.getData('J')))

```

### 深层神经网络

样本没变，全部代码见 `\test\test_deep_neural_network.py`

与浅层神经网络步骤一致，不同的是搭建计算图以及赋值的数量不同
```python
g = nous('''
X:(209, 12288) ->

    matmul W1:(12288, 20) add b1:(20) -> relu ->
    matmul W2:(20, 7) add b2:(7) -> relu ->
    matmul W3:(7, 5) add b3:(5) -> relu ->
    matmul W4:(5, 1) add b4:(1) -> sigmoid ->

cross_entropy Y:(209, 1) -> cost -> J:$$
''').parse()

```

209个样本经过四层神经网络，每层分别有12288、20、7、5个神经元，前三层使用`relu`激活函数，最后一层使用`sigmoid`激活函数，得到标签Yhat为`(209, 1)`向量，最后计算总代价J

初始化参数改为正态分布初始化

```python
g.setData('W1', np.random.randn(12288, 20) / np.sqrt(12288))
g.setData('W2', np.random.randn(20, 7) / np.sqrt(20))
g.setData('W3', np.random.randn(7, 5) / np.sqrt(7))
g.setData('W4', np.random.randn(5, 1) / np.sqrt(5))
```

### mini-batch 梯度下降
mini-batch 会将所有集合中的样本随机分成n个小集合，依次进入计算图进行学习计算，使原本需要计算m次的迭代增长为`m * n`次，但大幅降低每次的计算量

全部代码详见`test\test_mini_batch.py`

```python
# 将训练集分成多个子集
# 比如原本有300个样本，如果每64个位一组，则分成[64, 64, 64, 64, 44]五组
batches = set_mini_batch(train_X, train_Y)

rate = 0.0007
num_iterations = 10000

# 不声明X和Y的维度也可以搭建计算图，但无法进行计算，只有给两个节点正确赋值以后才能够运算
g = nous(
    '''
    X ->

        matmul W1:he((2, 5), 4) add b1:(5) -> relu ->
        matmul W2:he((5, 2), 10) add b2:(2) -> relu ->
        matmul W3:he((2, 1), 4) add b3:(1) -> sigmoid ->

    cross_entropy Y -> cost -> J:$$
    '''
).parse()

g.setUpdateFunc('W1', upf.norm(rate))
g.setUpdateFunc('W2', upf.norm(rate))
g.setUpdateFunc('W3', upf.norm(rate))
g.setUpdateFunc('b1', upf.norm(rate))
g.setUpdateFunc('b2', upf.norm(rate))
g.setUpdateFunc('b3', upf.norm(rate))

for i in range(num_iterations) :
    for b in batches :
        # 依次将每个样本子集送入计算图进行计算
        g.setData('X', b[0].T)
        g.setData('Y', b[1].T)

        g.fprop()
        g.bprop()

    if i % 1000 == 0 :
        print("Cost after iteration {}: {}".format(i, g.getData('J')))
print("Cost after iteration {}: {}".format(num_iterations, g.getData('J')))


```

### 多分类回归

多分类回归使用one-hot向量、将原本的sigmoid二分类器更改为softmax多分类器并辅佐对应的交叉熵函数。其中应用了mini-batch

测试时使用非标准运算符`max_index`，是`ont-hot`的逆运算，获取`one-hot`向量中最大的值的下标


全部代码见`test\test_softmax_multiple_classification.py`

```python

g = erud.nous(
    '''
    X:(1080, 12288) ->
    
        matmul W1:xavier((12288, 25), 12288) add b1:(25) -> relu ->
        matmul W2:xavier((25, 12), 25) add b2:(12) -> relu ->
        matmul W3:xavier((12, 6), 12) add b3:(6) ->
    
    softmax_cross_entropy(1) Y:(1080, 6) -> cost -> J:$$
    '''
).parse()

...

for i in range(num_iterations) :
    for b in batches :
        g.setData('X', b[0].T)
        g.setData('Y', b[1].T)

        g.fprop()
        g.bprop()

    if i % 100 == 0 :
        print("Cost after iteration {}: {}".format(i, g.getData('J')))
print("Cost after iteration {}: {}".format(num_iterations, g.getData('J')))


# 测试
gtest = erud.nous(
    '''
    X ->
    
        matmul W1:xavier((12288, 25), 12288) add b1:(25) -> relu ->
        matmul W2:xavier((25, 12), 25) add b2:(12) -> relu ->
        matmul W3:xavier((12, 6), 12) add b3:(6) ->
    
    max_index(1) -> accuracy Y -> J:$$
    '''
).parse()


```


### 卷积神经网络

`conv2d`执行卷积操作，`max_pool`执行最大池化，`flatten`拉平网络，并链接卷积层与全连接层

代码见`./test/test_conv.py`

```python
num_iterations = 200
rate = 0.04

g = erud.nous(
    '''
    X:(1080, 64, 64, 3) ->

        ##### 一层卷积
        ##### (1080, 64, 64, 3) -> (1080, 64, 64, 8) -> (1080, 8, 8, 8)

        conv2d(1, 2) W1:xavier((4, 4, 3, 8), 16) -> relu -> max_pool(8, 8, 8) ->

        ##### 二层卷积
        ##### (1080, 8, 8, 8) -> (1080, 8, 8, 16) -> (1080, 2, 2, 16)

        conv2d(1, 1) W2:xavier((2, 2, 8, 16), 4) -> relu -> max_pool(4, 4, 4) ->

        ##### 全连接
        flatten -> matmul W3:xavier((64, 6), 64) add b3:(6) ->

    softmax_cross_entropy(1) Y:(1080, 6) -> cost -> J:$$
    '''
).parse()

g.setUpdateFunc('W1', erud.upf.norm(rate))
g.setUpdateFunc('W2', erud.upf.norm(rate))
g.setUpdateFunc('W3', erud.upf.norm(rate))
g.setUpdateFunc('b3', erud.upf.norm(rate))

for i in range (num_iterations) :
    for b in batches :
        g.setData('X', b[0])
        g.setData('Y', b[1])

        g.fprop()
        g.bprop()

    if i % 1 == 0 :
        print("Cost after iteration {}: {}".format(i, g.getData('J')))
print("Cost after iteration {}: {}".format(num_iterations, g.getData('J')))

```

