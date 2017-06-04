#! coding=utf-8
import libSvm
import json
import warnings

warnings.filterwarnings("ignore")


def gg(name):
    with open('result/svmResult/yacht/%s.json' % name, 'a+') as f:
        for i in range(68, 309, 30):
            print '正在进行,i=', i
            tempDic = libSvm.calculateData(dataPath='data/libData/yacht.txt', trainRate=i * 0.75 / 308.0)
            f.write(json.dumps(tempDic) + '\n')


a = ['test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9']
for n in a:
    gg(n)
