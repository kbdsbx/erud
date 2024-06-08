import erud
import math
from erud.upf.adam import adam
from erud._utils import useGPU
if useGPU :
	import cupy as cp
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import xml.dom.minidom as dom

# 标号与类型的映射
htt = {
        0 : 'person',
        1 : 'bird',
        2 : 'cat',
        3 : 'cow',
        4 : 'dog',
        5 : 'horse',
        6 : 'sheep',
        7 : 'aeroplane',
        8 : 'bicycle',
        9 : 'boat',
        10 : 'bus',
        11 : 'car',
        12 : 'motorbike',
        13 : 'train',
        14 : 'bottle',
        15 : 'chair',
        16 : 'diningtable',
        17 : 'pottedplant',
        18 : 'sofa',
        19 : 'tvmonitor',
    }

# 类型与标号的映射
tth = {
        'person' : 0,
        'bird' : 1,
        'cat' : 2,
        'cow' : 3,
        'dog' : 4,
        'horse' : 5,
        'sheep' : 6,
        'aeroplane' : 7,
        'bicycle' : 8,
        'boat' : 9,
        'bus' : 10,
        'car' : 11,
        'motorbike' : 12,
        'train' : 13,
        'bottle' : 14,
        'chair' : 15,
        'diningtable' : 16,
        'pottedplant' : 17,
        'sofa' : 18,
        'tvmonitor' : 19,
    }

# 在图片上绘制矩形框
# values = [[p, xmin, ymin, xmax, ymax, c]]
def drawbox(img, values) :
    if isinstance(img, str) :
        img = cv2.imread(img)

    for v in values :
        p = v[0]
        lt = (v[1], v[2])
        rb = (v[3], v[4])
        c = v[5]
        fpos = (v[1], v[2] - 10)
        fcol = (0, 255, 255)
        color = (0, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(img, lt, rb, color, 2)
        cv2.putText(img, '{} {:.3f}'.format(c, p), fpos, font, .5, fcol, 2)

    p = plt.subplot()
    p.invert_yaxis()

    plt.imshow(img)
    plt.show()

# 在图片上绘制yolo格式信息
# y : 7 * 7 * [p, x, y, w, h, p, x, y, w, h, c1, c2, ..., c20]
def drawboxyolo(img, Y, show = False, save = False) :
    img = np.uint8(img)

    fcolor = (0, 255, 255)
    color = (0, 255, 255)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for m in range(7) :
        for n in range(7) :
            vec = Y[m][n]
            ## 第一个盒子
            p = vec[0]
            # 不渲染置信度为0的盒子
            if p == 0 :
                continue
            cx = 64 * vec[1]
            cy = 64 * vec[2]
            w = 448 * vec[3]
            h = 448 * vec[4]
            xmin = math.floor(m * 64 + cx - w / 2)
            ymin = math.floor(n * 64 + cy - h / 2)
            xmax = math.floor(m * 64 + cx + w / 2)
            ymax = math.floor(n * 64 + cy + h / 2)
            ctype = htt[np.argmax(vec[10:])]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(img, '{} {:.3f}'.format(ctype, p), (xmin, ymin - 10), font, .5, fcolor, 1)

            # 第二个盒子
            p = vec[5]
            # 不渲染置信度为0的盒子
            if p == 0 :
                continue
            cx = 64 * vec[6]
            cy = 64 * vec[7]
            w = 448 * vec[8]
            h = 448 * vec[9]
            xmin = math.floor(m * 64 + cx - w / 2)
            ymin = math.floor(n * 64 + cy - h / 2)
            xmax = math.floor(m * 64 + cx + w / 2)
            ymax = math.floor(n * 64 + cy + h / 2)
            ctype = htt[np.argmax(vec[10:])]
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), color, 1)
            cv2.putText(img, '{} {:.3f}'.format(ctype, p), (xmin, ymin - 10), font, .5, fcolor, 1)

    p = plt.subplot()
    p.invert_yaxis()

    plt.imshow(img)
    if show :
        plt.show() 
    if save :
        plt.imsave(save)

# 将数据文件中的信息转换成yolo可以使用的格式(Y标签)
def img_normalization (w, h, objinfos) :
    
    infoblock = []
    for m in range(7) :
        # 计算grid的范围
        lt= w / 7. * m
        rt = w / 7. * (m + 1)
        inforow= []
        for n in range(7) :
            # 计算grid的范围
            lb = h / 7. * n
            rb = h / 7. * (n + 1)

            for info in objinfos :
                # 计算物体在原始图片中的宽高
                ow = info[3] - info[1]
                oh = info[4] - info[2]
                # 计算物体在原始图片中的中心点
                cx = info[1] + ow / 2
                cy = info[2] + oh / 2
                
                # 如果中心点落在了此grid
                if (cx > lt and cx <= rt and cy > lb and cy <= rb) :
                    infocol = [
                        info[0],
                        (cx % (w / 7)) / (w / 7),
                        (cy % (h / 7)) / (h / 7),
                        ow / w,
                        oh / h,
                        info[0],
                        (cx % (w / 7)) / (w / 7),
                        (cy % (h / 7)) / (h / 7),
                        ow / w,
                        oh / h,
                        *np.eye(20)[tth[info[5]]]
                    ]
                    break
            else :
                infocol = [0 for i in range(30)]

            inforow.append(infocol)
        infoblock.append(inforow)
    
    return infoblock

allfile = []

# 获取VOC2007训练集信息，获取mini-batch
# 训练集共5011张图片
# 测试集共4952张图片
# batch_index 索引，从0开始
# batch_size 大小
def getVOC2007 (batch_index, batch_size=64, test = False) :
    '''
    batch_index mini-batch索引，从0开始
    batch_size mini-batch 大小，默认64
    test 是否使用测试集图片，默认False
    '''
    # 测试集
    if test : 
        imgpath = __file__[:__file__.rfind('\\')] + '/datasets/voc2007-test/JPEGImages-test/'
        xmlpath = __file__[:__file__.rfind('\\')] + '/datasets/voc2007-test/Annotations-test/'
    else :
        imgpath = __file__[:__file__.rfind('\\')] + '/datasets/voc2007/JPEGImages/'
        xmlpath = __file__[:__file__.rfind('\\')] + '/datasets/voc2007/Annotations/'

    global allfile
    
    if len(allfile) == 0 :
        allfile = os.listdir(imgpath)
    
    if (batch_index + 1) * batch_size > len(allfile) :
        batch = allfile[batch_index * batch_size :]
    else :
        batch = allfile[batch_index * batch_size : (batch_index + 1) * batch_size]

    yoloinfolist = []
    yoloimglist = []

    for f in batch :
        fname = f[:f.rfind('.')]
        fpath = os.path.join(imgpath, f)
        apath = os.path.join(xmlpath, fname + '.xml')

        # 获取图片的宽度高度和通道数
        anno = dom.parse(apath)
        w = int(anno.getElementsByTagName('width')[0].firstChild.data)
        h = int(anno.getElementsByTagName('height')[0].firstChild.data)
        c = int(anno.getElementsByTagName('depth')[0].firstChild.data)

        objs = anno.getElementsByTagName('object')

        objinfos = []

        for obj in objs :
            # 当前对象类别
            tp = obj.getElementsByTagName('name')[0].firstChild.data
            xmin = int(obj.getElementsByTagName('xmin')[0].firstChild.data)
            ymin = int(obj.getElementsByTagName('ymin')[0].firstChild.data)
            xmax = int(obj.getElementsByTagName('xmax')[0].firstChild.data)
            ymax = int(obj.getElementsByTagName('ymax')[0].firstChild.data)

            info = [1, xmin, ymin, xmax, ymax, tp]
            objinfos.append(info)

        # 计算图片样本标签
        yoloinfo = img_normalization(w, h, objinfos)
        yoloinfolist.append(yoloinfo)

        # 计算图片样本数据
        img = cv2.imread(fpath)
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))
        img = cv2.resize(img, (448, 448))
        yoloimglist.append(img.tolist())
    
    return yoloimglist, yoloinfolist

    # drawbox(fpath, objinfos)
    # img = cv2.imread(fpath)
    # b,g,r = cv2.split(img)
    # img = cv2.merge((r,g,b))
    # img = cv2.resize(img, (448, 448))
    # drawboxyolo(img, yoloinfo)

def no_test_yolov1() :

    X, Y = getVOC2007(0, 1)
    X = np.array(X)
    Y = np.array(Y)
    print(X.shape)
    print(Y.shape)

    # 查看某张图片和对象绘制
    # drawboxyolo(X[0], Y[0])

    # 归一化
    X = X / 255.

    print('Dataset loaded')

    code = '''
X:(1, 448, 448, 3) ->
    # 1 -> 224 -> 112
    conv2d_v3(2, 4) W1:xavier_cnn(7, 7, 3, 64) ->
        leaky_relu ->
        max_pool_v3(2, 2, 2) ->
    # 2 -> 56
    conv2d_v3_same W2:xavier_cnn(3, 3, 64, 192) ->
        leaky_relu ->
        max_pool_v3(2, 2, 2) ->
    # 3
    conv2d_v3_same W3:xavier_cnn(1, 1, 192, 128) ->
        leaky_relu ->
    # 4
    conv2d_v3_same W4:xavier_cnn(3, 3, 128, 256) ->
        leaky_relu ->
    # 5    
    conv2d_v3_same W5:xavier_cnn(1, 1, 256, 256) ->
        leaky_relu ->
    # 6 -> 28
    conv2d_v3_same W6:xavier_cnn(3, 3, 256, 512) ->
        leaky_relu ->
        max_pool_v3(2, 2, 2) ->
    # 7    
    conv2d_v3_same W7:xavier_cnn(1, 1, 512, 256) ->
        leaky_relu ->
    # 8    
    conv2d_v3_same W8:xavier_cnn(3, 3, 256, 512) ->
        leaky_relu ->
    # 9
    conv2d_v3_same W9:xavier_cnn(1, 1, 512, 256) ->
        leaky_relu ->
    # 10 
    conv2d_v3_same W10:xavier_cnn(3, 3, 256, 512) ->
        leaky_relu ->
    # 11
    conv2d_v3_same W11:xavier_cnn(1, 1, 512, 256) ->
        leaky_relu ->
    # 12
    conv2d_v3_same W12:xavier_cnn(3, 3, 256, 512) ->
        leaky_relu ->
    # 13
    conv2d_v3_same W13:xavier_cnn(1, 1, 512, 256) ->
        leaky_relu ->
    # 14
    conv2d_v3_same W14:xavier_cnn(3, 3, 256, 512) ->
        leaky_relu ->
    # 15
    conv2d_v3_same W15:xavier_cnn(1, 1, 512, 512) ->
        leaky_relu ->
    # 16 -> 14
    conv2d_v3_same W16:xavier_cnn(3, 3, 512, 1024) ->
        leaky_relu ->
        max_pool_v3(2, 2, 2) ->
    # 17
    conv2d_v3_same W17:xavier_cnn(1, 1, 1024, 512) ->
        leaky_relu ->
    # 18
    conv2d_v3_same W18:xavier_cnn(3, 3, 512, 1024) ->
        leaky_relu ->
    # 19
    conv2d_v3_same W19:xavier_cnn(1, 1, 1024, 512) ->
        leaky_relu ->
    # 20 
    conv2d_v3_same W20:xavier_cnn(3, 3, 512, 1024) ->
        leaky_relu ->
    # 21 
    conv2d_v3_same W21:xavier_cnn(3, 3, 1024, 1024) ->
        leaky_relu ->
    # 22 -> 7
    conv2d_v3(2, 1) W22:xavier_cnn(3, 3, 1024, 1024) ->
        leaky_relu ->
    # 23 
    conv2d_v3_same W23:xavier_cnn(3, 3, 1024, 1024) ->
        leaky_relu ->
    # 24 
    conv2d_v3_same W24:xavier_cnn(3, 3, 1024, 1024) ->
        leaky_relu ->
    
    # batch * 7 * 7 * 1024
    
flatten ->

    matmul WL1:xavier(50176, 4096) add bl1:(4096) -> 
        leaky_relu ->

    matmul WL2:xavier(4096, 1470) add bl2:(1470) -> 
        sigmoid ->
    
    reshape((7, 7, 30)) => Yhat ->
    
    yolo1_loss Y:ones(1, 7, 7, 30) -> cost -> J:$$
'''

    path = __file__[:__file__.rfind('\\')] + '/datasets/'
    cachename = path + '/yolo1_cache.json'
    rate = 0.001

    n = erud.nous()

    # 如果存在缓存文件则使用缓存文件创建网络
    if os.path.exists(cachename) :
        n, obj = erud.nous_imports(cachename)
        g = n.g
        num_over_iterations = obj['num_over_iterations']
        num_iterations= obj['num_iterations']
    else :
        g = n.parse(code)
        num_over_iterations = 0
        num_iterations = 100

        for i in range(24) :
            g.setUpdateFunc('W%s' % (i+1), erud.upf.adam(rate))

        g.setUpdateFunc('WL1', erud.upf.adam(rate))
        g.setUpdateFunc('bl1', erud.upf.adam(rate))
        g.setUpdateFunc('WL2', erud.upf.adam(rate))
        g.setUpdateFunc('bl2', erud.upf.adam(rate))


    # g.show()

    print('Network already.')

    Xtest, Ytest = getVOC2007(0, 4952, True)
    trainLoss = []
    testLoss = []

    for i in range(num_iterations - num_over_iterations) :

        # mini-batch 训练
        for bindex in range(math.floor(5011 / 64)) :

            X,Y = getVOC2007(bindex)

            g.setData('X', X)
            g.setData('Y', Y)

            g.fprop()
            g.bprop()
        
        costs = g.getData('J')
        trainLoss.append(costs)

        if (i + num_over_iterations + 1) % 1 == 0 :
            print("Cost after iteration {}: {}".format(1 + i + num_over_iterations, g.getData('J')))
            # erud.exports(n, cachename, {
            #     'costs' : costs,
            #     'num_over_iterations' : 1 + i + num_over_iterations,
            #     'num_iterations' : num_iterations,
            #     'rate' : rate
            # })
        if (i + num_over_iterations + 1) % 1 == 0 :
            # 多次备份缓存
            erud.exports(n, path + '/yolo-bk-' + str(i + num_over_iterations + 1) + '.json', {
                'costs' : costs,
                'num_over_iterations' : 1 + i + num_over_iterations,
                'num_iterations' : num_iterations,
                'rate' : rate,
            })
        
        # 每经历n次学习，计算网络在测试集上的损失
        if (i + num_over_iterations + 1) % 4 == 0 :
            g.setData('X', Xtest)
            g.setData('Y', Ytest)
            g.fprop()
            testLoss.append(g.getData('J'))


    # 查看预测结果和对象绘制
    # drawboxyolo(X[0], g.getData('J')[0])

    # 输出训练集loss
    print('trainLoss : ')
    print(trainLoss)

    # 输出测试集loss
    print('testLoss : ')
    print(testLoss)

    # 输出计算统计时间(s)
    print(g.tableTimespend())

def prediction(g) :
    '''
    预测

    g: 训练好的网络
    '''

    # 预测图片集
    X = []
    imgpath = __file__[:__file__.rfind('\\')] + '/datasets/voc2007-test/JPEGImages-pred/'
    imglist = os.listdir(imgpath)
    for f in imglist :
        # 计算图片样本数据
        img = cv2.imread(f)
        b,g,r = cv2.split(img)
        img = cv2.merge((r,g,b))
        img = cv2.resize(img, (448, 448))
        X.append(img.tolist())

    Xpred = np.array(X) / 255.
    # 不计算损失，所以Y随意
    Ypred = np.zeros((len(X), 7, 7, 30))

    g.setData('X', Xpred)
    g.setData('Y', Ypred)
    g.fprop()

    # 获取预测结果
    Yres = g.getData('Yhat')

    respath = __file__[:__file__.rfind('\\')] + '/datasets/voc2007-test/JPEGImages-res/'

    i = 0
    # 查看计算预测结果
    # 存储预测结果
    for f in imglist :
        name = f[f.rfind('\\'):]
        drawboxyolo(X[i], Yres[i], False, respath + name)
        i+=1

    # 非极大抑制

    # 查看最终预测结果
    