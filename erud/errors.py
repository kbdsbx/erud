
####### 计算图异常 #######

# 找不到节点
class NodeNotFindError ( Exception ) : ...

# 找不到边
class EdgeNotFindError ( Exception ) : ...

# 重复添加边
class EdgeRepeatError (Exception ) : ...

# 重复添加节点
class NodeRepeatError (Exception) : ...


####### nous语言表达式异常 ######

# 表达式错误
class ParseError (Exception) : ...


####### 其他异常 ######

# 不支持的功能
class UnsupportedError (Exception) : ...