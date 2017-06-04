import time
import numpy as np
from libsvm.python.svmutil import *
from libsvm.python.svm import *
import func
import matplotlib.pyplot as plt


def fitLibsvm(x_train, y_train, x_test, y_test):
    model = svm_problem(y_train, x_train)
    param = svm_parameter('-s 3 -t 0')
    F1 = svm_train(model, param)
    y_train_fit, p__train_acc, p_train_val = svm_predict(y_train, x_train, F1)
    mseTrain = (float(1) / len(y_train)) * np.linalg.norm((np.array(y_train_fit) - np.array(y_train)), ord=2) ** 2
    rmseTrain = mseTrain ** 0.5

    y_test_fit, p__test_acc, p_test_val = svm_predict(y_test, x_test, F1)
    mseTest = (float(1) / len(y_test)) * np.linalg.norm((np.array(y_test_fit) - np.array(y_test)), ord=2) ** 2
    rmseTest = mseTest ** 0.5

    return {'mseTrain': mseTrain, 'mseUlt': mseTest, 'rmseTrain': rmseTrain, 'rmseUlt': rmseTest}


def calculateData(dataPath, trainRate):
    time1 = time.time()
    global x_test, y_test, y_test_fit, result
    y, x = svm_read_problem(dataPath)
    y_train = y[:int(len(y) * trainRate)]
    x_train = x[:int(len(x) * trainRate)]
    y_test = y[int(len(y) * 0.75):]
    x_test = x[int(len(x) * 0.75):]

    result = fitLibsvm(x_train, y_train, x_test, y_test)
    time2 = time.time()
    result['time'] = time2 - time1
    result['trainRate']=trainRate
    return result

