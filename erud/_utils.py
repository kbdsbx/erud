
# 一个很小值
epsilon = 1e-16

# 是否使用gpu
useGPU = False 


def iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2) :
    '''
    计算交并比

    (xmin1, ymin1) (xmax1, ymax1) 第一个点的左上右下坐标
    (xmin2, ymin2) (xmax2, ymax2) 第二个点的左上右下坐标
    '''

    # 计算交叉框大小
    w = min(xmax1, xmax2) - max(xmin1, xmin2)
    h = min(ymax1, ymax2) - max(ymin1, ymin2)

    # 计算交集
    # 负的相交框宽/高表示两个框相离
    if w < 0 or h < 0 :
        i = 0
    else :
        i = w * h
    
    # 计算原始框大小
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    # 计算并集
    u = s1 + s2 - i

    # 计算IoU
    return max(0, i / u)