#!coding=utf-8
''''' 
created by wh
'''
import numpy as np
import func
from sklearn import linear_model as lm
from sklearn import model_selection as ms
import matplotlib.pyplot as plt
import time

# 1.读取数据并且打乱数据保存在data/distrubData中 2.选择50%为训练样本25%为验证样本25%为测试样本（根据打乱后的数据
# 进行选择）3.根据scale随机选择gamma 4.根据kernal做特征映射得到featureMap 5.对参数alpha进行求解

#############################
test_a = []
test_b = []


##############################



def kernal(x1, x2, gamma):
    # 计算kernal
    nrow = x1 - x2
    K = np.exp(-(np.linalg.norm(nrow, axis=1, ord=2)).reshape(nrow.shape[0], 1) ** 2 * float(gamma))
    return K


def featureMap(x, gamma, xBase):
    # 做特征映射,得到的结果返回到featureMap中
    Map = np.zeros([x.shape[0], xBase.shape[0]])  # init
    for i in range(xBase.shape[0]):
        # 按列生成Map
        Map[:, [i]] = kernal(x, xBase[i], gamma=gamma[i])
    return Map


def plot_ppt(coef, train, gamma):
    x = np.arange(start=-1, stop=1, step=0.00001)
    x = x.reshape(x.shape[0], 1)
    # print np.linalg.norm(x-ppt_i[1],axis=1,ord=2).reshape(x.shape[0], 1)
    y = coef * np.exp(-(np.linalg.norm(x - train, axis=1, ord=2).reshape(x.shape[0], 1)) ** 2 * gamma)
    plt.plot(x, y)


def scale2fitting(scale, x_train, y_train, x_test, y_test):
    global test_a, test_b
    # 选取gamma，其中scale为尺度上界一般取(2^n)，具体scale值需要crossValiation确定，gamma服从U[0,scale]
    parameters = {}
    # parameters里第一列放的是系数coef_，第二列放的是x_train的位置，第三列放的是gamma
    parameters['x_train'] = x_train
    gamma = np.random.uniform(0, scale, size=x_train.shape[0])
    parameters['gamma'] = gamma
    trainMap = featureMap(x_train, gamma, x_train)
    try:
        F1 = lm.LassoLarsCV(cv=5, normalize=False)
        F1.fit(trainMap, y_train)
        parameters['coef'] = F1.coef_
        y_train_fit = F1.predict(trainMap)
        mseTrain = (float(1) / len(y_train)) * np.linalg.norm((y_train_fit - y_train), ord=2) ** 2
        rmseTrain = mseTrain ** 0.5

        testMap = featureMap(x_test, gamma, x_train)
        y_test_fit = F1.predict(testMap)
        mseTest = (float(1) / len(y_test)) * np.linalg.norm((y_test_fit - y_test), ord=2) ** 2
        rmseTest = mseTest ** 0.5

        return {'mseTrain': mseTrain, 'mseTest': mseTest, 'rmseTrain': rmseTrain, 'rmseTest': rmseTest,
                'parameters': parameters, 'scale': scale, 'x_train': x_train, 'gamma': gamma, 'model': F1}

    except:
        print 'lasso/lars error'
        return {'mseTest': 999999999}


def calculateData(dataPath, disturbNum, trainRate, scaleRange=20, fold=10):
    ultScale = {}
    time1 = time.time()
    global x_test, y_test, y_test_fit, result
    path = func.readData(dataPath, skipRows=1, disturbNum=disturbNum)
    data = np.loadtxt(path, delimiter=',', skiprows=1)
    train = data[:int(data.shape[0] * trainRate), :]
    test = data[int(data.shape[0] * 0.75):, :]

    x_test = test[:, range(data.shape[1] - 1)]
    y_test = test[:, data.shape[1] - 1]

    a = np.arange(start=-scaleRange, stop=scaleRange, step=1)
    for i in range(len(a)):
        print 'now,calculate %sth ,total:%s ' % (i, len(a))
        fit = {}
        kf = ms.KFold(n_splits=fold)
        sumMseTest = 0
        j = 0
        count = 0
        for id_cvTrain, id_cvTest in kf.split(train):
            cvTrain = train[id_cvTrain]
            cvTest = train[id_cvTest]
            temp = scale2fitting(2 ** (a[i]), x_train=cvTrain[:, range(cvTrain.shape[1] - 1)],
                                 y_train=cvTrain[:, cvTrain.shape[1] - 1],
                                 x_test=cvTest[:, range(cvTest.shape[1] - 1)],
                                 y_test=cvTest[:, cvTest.shape[1] - 1])
            fit['%s' % j] = temp
            j += 1
            if temp == 0:
                print 'Warning.both lasso/lars error'
                pass
            else:
                sumMseTest += temp['mseTest']
                count += 1
        meanMseTest = sumMseTest / float(count)
        if ultScale == {} or meanMseTest < ultScale['meanMseTest']:
            ultScale = fit
            ultScale['meanMseTest'] = meanMseTest
        else:
            pass

    result = {}
    for i in range(fold):
        if result == {} or ultScale['%s' % i]['mseTest'] < result['mseTest']:
            result = ultScale['%s' % i]
            print '第%s次更好' % i

    F2 = result['model']
    testMap = featureMap(x_test, result['gamma'], result['x_train'])
    y_test_fit = F2.predict(testMap)
    mseUlt = (float(1) / len(y_test)) * np.linalg.norm((y_test_fit - y_test), ord=2) ** 2
    rmseUlt = mseUlt ** 0.5
    result['mseUlt'] = mseUlt
    result['rmseUlt'] = rmseUlt

    time2 = time.time()

    return {'trainSample': train.shape[0], 'rmseTrain': result['rmseTrain'], 'rmseUlt': result['rmseUlt'],
            'scale': result['scale'],
            'time': time2 - time1}


# 画图
def paint():
    plt.figure(1)
    for i in range(result['parameters']['x_train'].shape[0]):
        if result['parameters']['coef'][i] != 0:
            plot_ppt(result['parameters']['coef'][i], result['parameters']['x_train'][i],
                     result['parameters']['gamma'][i])
    plt.scatter(x_test, y_test, marker='x', c='b')
    plt.scatter(x_test, y_test_fit, marker='+', c='r')
    plt.axis([-1, 1, 0, 1])
    plt.show()


def mlt_plt():
    plt.figure(2)
    x = np.arange(1, 1.1, 0.01)
    y = x
    plt.plot(x, y)
    plt.scatter(y_test, y_test_fit, marker='x', c='b')
    plt.show()
