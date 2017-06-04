#!coding=utf-8
''''' 
created by wh
'''
import numpy as np
import random
import os


def readData(dataPath, delimiter=',', skipRows=0, disturbNum=0):
    ###########################################################################
    ### Read data,according to row,delimiter is ','default skiprows is Null ###
    ### disturbNum=1 means disturb data.                                    ###
    ###########################################################################
    disturbName = 'd_' + dataPath[dataPath.rfind('/') + 1:]
    try:
        os.makedirs(dataPath[:dataPath.rfind('/') + 1] + 'disturbData')
    except:
        pass
    if disturbNum == 1:
        data = np.loadtxt(dataPath, delimiter=delimiter, skiprows=skipRows)
        rangeOfData = range(data.shape[0])
        random.shuffle(rangeOfData)
        disturb = data[rangeOfData,]
        disturbPath = dataPath[:dataPath.rfind('/') + 1] + 'disturbData/' + disturbName
        np.savetxt(fname=disturbPath, X=disturb, fmt='%.14e',
                   delimiter=',', header='disturb%s' % disturbName)

    else:
        print 'onlyRead'
        disturbPath = dataPath[:dataPath.rfind('/') + 1] + 'disturbData/' + disturbName
        print disturbPath
    return disturbPath
