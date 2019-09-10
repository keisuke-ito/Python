import os
import sys
import numpy as np
import pandas as pd
from NN import NineLayerNN
from matplotlib import pyplot as plt

# The residual sum of squares(残差平方和)
def rss(y, t):
    out_rss = np.sum(abs(y - t))
    return out_rss


def DNN9(k, div_pre, x_train, x_test, y_train, y_test, MS1, MS2, MLP1, MLP2, MLP3, SD2, SD3, PARAMETER, Weight, Bias):
    # os.chdir('/Users/itoukeisuke/Python/pycharm/5layer_AEMS/program/')
    collect = True
    check = True

    # Make Data
    div = div_pre

    ## Parameter Setting
    train_num, Input = x_train.shape
    test_num, Output = y_test.shape

    NN9 = NineLayerNN(Input, MS1, MS2, MLP1, MLP2, MLP3, SD2, SD3, Output, Weight, Bias, PARAMETER)

    errmin = np.inf
    ERROR = []
    CO = []
    if collect:
        Error_decay = []
    else:
        Error_decay = 0

    for epoch in range(PARAMETER['EPOCH']):

        # Model Fit
        for i in range(train_num):
            idata = x_train.iloc[i].values
            tdata = y_train.iloc[i].values
            NN9.fit(idata, tdata, Input, Output)

        if k == (div - 1) and collect:
            # Error collect
            Error = []
            for i in range(train_num):
                idata = x_train.iloc[i].values
                tdata = np.reshape(y_train.iloc[i].values, (Output, 1))
                out = NN9.predict(idata, Input)

                Error.append(rss(tdata, out))

            # 各epochでの平均の誤差を格納する
            Error_decay.append(np.mean(Error))
            print(Error_decay)

        # Model Test
        if (epoch % PARAMETER['MOD_EPOCH']) == 0:
            e = 0
            for i in range(test_num):
                idata = x_test.iloc[i].values
                tdata = np.reshape(y_test.iloc[i].values, (Output, 1))
                predict = NN9.predict(idata, Input)  # predict: 144×1

                e += rss(tdata, predict)

            if errmin > e:
                errmin = e
                w0 = NN9.w0
                w1 = NN9.w1
                w2 = NN9.w2
                w3 = NN9.w3
                w4 = NN9.w4
                w5 = NN9.w5
                w6 = NN9.w6
                w7 = NN9.w7
                b0 = NN9.b0
                b1 = NN9.b1
                b2 = NN9.b2
                b3 = NN9.b3
                b4 = NN9.b4
                b5 = NN9.b5
                b6 = NN9.b6
                b7 = NN9.b7

            else:
                break

            print('|[DNN9] Epoch: {} / {} | Error: {:.2f} |'.format(epoch + 1, PARAMETER['EPOCH'], e))

        print('|        NN9 Last Error: {:.2f}       |'.format(errmin))


    # 9 Layer Check
    if check:
        E_test = []
        Predict = []
        for i in range(test_num):
            tdata = np.reshape(y_test.iloc[i].values, (Output, 1))
            res = NN9.predict(x_test.iloc[i].values, Input)

            Predict.append(np.reshape(res, (Output,)))
            E_test.append(rss(tdata, res))

        E_per = np.mean(E_test)
        Predict = pd.DataFrame(Predict)
        CO_test = []
        for i in range(len(Predict)):
            R = np.corrcoef(y_test.iloc[i], Predict.iloc[i])
            CO_test.append(R[1, 0])
        CO_test = np.mean(CO_test)
        print('Performance for 9-Layer DNN')
        print('Corrcoef: {:.3f}, Error: {:.2f}'.format(CO_test, E_per))

        ERROR.append(E_per)
        CO.append(CO_test)



    return w0, w1, w2, w3, w4, w5, w6, w7, b0, b1, b2, b3, b4, b5, b6, b7, ERROR, CO, Error_decay




