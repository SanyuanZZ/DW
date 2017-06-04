#! coding=utf-8
import singleL1
import json
import warnings

warnings.filterwarnings("ignore")


def gg(name):
    with open('singleResult/stock_fold10/%s.json' % name, 'a+') as f:
        for i in range(296, 537, 40):
            print '正在进行,i=', i
            tempDic = singleL1.calculateData(dataPath='data/originalData/STOCK_scale_536.csv', disturbNum=0,
                                                      trainRate=i * 0.75 / 296.0, fold=10)
            f.write(json.dumps(tempDic) + '\n')


a = ['test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9']
for n in a:
    gg(n)
