#! coding=utf-8
import os
import json
import numpy as np

resultFile_1 = os.listdir('result')
for fileName_1 in resultFile_1:
    resultFile_2 = os.listdir('result/%s' % fileName_1)
    for fileName_2 in resultFile_2:
        resultFile_3 = os.listdir('result/%s/%s' % (fileName_1, fileName_2))
        nowPath = 'result/%s/%s/' % (fileName_1, fileName_2)
        print nowPath
        #####################################################
        # create scale list
        with open(nowPath + resultFile_3[0], 'r') as f:
            numberOfScale = 0
            for row in f:
                numberOfScale += 1
            print numberOfScale
            scaleAll = {}
            for scaleNumber in range(numberOfScale):
                exec ('scale_{num}=[]'.format(num=scaleNumber))
                exec ('scaleAll[\'scale_{num}\']=[]'.format(num=scaleNumber))
        ######################################################
        for j in range(len(resultFile_3)):
            # j=0...9 实验重复次数
            with open(nowPath + resultFile_3[j], 'r') as f:
                m = 0  # m为尺度个数
                for row in f:
                    scaleAll[scaleAll.keys()[m]].append(json.loads(row))
                    m += 1

        tempRmseTrain = []
        tempRmseTest = []
        meanRmseTrain = {}
        meanRmseTest = {}
        stdRmseTrain = {}
        stdRmseTest = {}

        for u in range(numberOfScale):
            tempRmseTrain = []
            tempRmseTest = []
            meanRmseTrain = {}
            meanRmseTest = {}
            stdRmseTrain = {}
            stdRmseTest = {}

            for k in range(10):
                tempRmseTrain.append(scaleAll[scaleAll.keys()[u]][k]['rmseTrain'])
                tempRmseTest.append(scaleAll[scaleAll.keys()[u]][k]['rmseUlt'])
            print tempRmseTrain
            meanRmseTrain[scaleAll.keys()[u]] = np.mean(tempRmseTrain)
            meanRmseTest[scaleAll.keys()[u]] = np.mean(tempRmseTest)
            stdRmseTrain[scaleAll.keys()[u]] = np.std(tempRmseTrain)
            stdRmseTest[scaleAll.keys()[u]] = np.std(tempRmseTest)
            with open(nowPath + 'meanAndVar.json', 'a+') as f:
                f.write('meanRmseTrain')
                f.write(json.dumps(meanRmseTrain))
                f.write('meanRmseTest')
                f.write(json.dumps(meanRmseTest))
                f.write('stdRmseTrain')
                f.write(json.dumps(stdRmseTrain))
                f.write('stdRmseTest')
                f.write(json.dumps(stdRmseTest) + '\n')
