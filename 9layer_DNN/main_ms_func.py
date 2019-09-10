import os
import numpy as np
import pandas as pd
from make_data import MakeDataMS
from NN import ThreeLayerNN, FiveLayerNN
from matplotlib import pyplot as plt

# The residual sum of squares(残差平方和)
def rss(y, t):
    out_rss = np.sum(abs(y - t))
    return out_rss


def AE5_ms(div_pre, E, train_data, test_data, dim, H1, H2):
    # os.chdir('/Users/itoukeisuke/Python/pycharm/5layer_AEMS/program/')
    collect = False
    check = False
    if div_pre != "":
        div = 1
    else:
        div = 6

    ## Parameter Setting
    EPOCH   = E
    # EPOCH = 300
    Input   = dim
    hidden1 = H1
    hidden2 = H2
    hidden3 = hidden1
    Output  = dim
    lr      = 0.4
    REGULARIZATION = {'REGRESSION': 'RIDGE', 'LAMBDA': 0.0000004}
    ETA            = {'RATE': lr, 'ADJUSTMENT': 0.99}
    PARAMETER = {'REGULARIZATION': REGULARIZATION,
                 'ETA': ETA,
                 'ALPHA': 0.15,
                 'NOISE': 0.000001,
                 'DISTANCE': 'MSE',
                 'EPOCH': EPOCH,
                 'MOD_EPOCH': 5}

    # Make instance of Neural Network
    AE1 = ThreeLayerNN(Input, hidden1, Input, PARAMETER)
    AE2 = ThreeLayerNN(hidden1, hidden2, hidden3, PARAMETER)

    # Pre Setting
    errmin1 = np.inf
    errmin2 = np.inf
    errmin5 = np.inf
    ERROR = []
    CO = []

    # k-fold Cross Validation
    for k in range(div):
        # print('| k : {} / {}                       |'.format(k+1, div))
        x_train = train_data
        x_test = test_data
        train_num = len(x_train)
        test_num = len(x_test)

        # --- Training --- #
        # 1,3層目(Input -> hidden1, hidden3 -> Output)
        print('|          MSAE 1st Training         |')
        for epoch in range(EPOCH):

            # Model Fit #
            for i in range(train_num):
                idata = x_train.iloc[i].values
                tdata = idata
                AE1.fit(idata, tdata, Input, Input)

            # Model Test #
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                e_1 = 0
                for i in range(test_num):
                    idata = x_test.iloc[i].values
                    tdata = np.reshape(x_test.iloc[i].values, (Input, 1))
                    predict = AE1.predict(idata, Input)  # predict: 144×1

                    e_1 += rss(tdata, predict)

                if errmin1 > e_1:
                    errmin1 = e_1
                    w0 = AE1.w_ih
                    w3 = AE1.w_ho
                    b0 = AE1.b_ih
                    b3 = AE1.b_ho
                else:
                    break

                print('|[MSAE1] Epoch: {} / {} | Error: {:.2f} |'.format(epoch+1, EPOCH, e_1))

        print('|        MSAE1 Last Error: {:.2f}       |'.format(errmin1))


        # 2層目(hidden1 -> hidden2)
        print('|          MSAE 2nd Training         |')

        # make data
        Featured_train = []
        for i in range(train_num):
            Featured_train.append(np.reshape(AE1.feed_makedata(x_train.iloc[i].values, Input), (hidden1,)))

        Featured_test = []
        for i in range(test_num):
            Featured_test.append(np.reshape(AE1.feed_makedata(x_test.iloc[i].values, Input), (hidden1,))) # 出力: hidden1 × 1

        # Featured_train/test : sample * feature
        Featured_train = pd.DataFrame(Featured_train)
        Featured_test  = pd.DataFrame(Featured_test)

        for epoch in range(EPOCH):

            # Model Fit #
            for i in range(train_num):
                idata = Featured_train.iloc[i].values
                tdata = idata
                AE2.fit(idata, tdata, hidden1, hidden1)

            # Model Test #
            e_2 = 0
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                for i in range(test_num):
                    idata = Featured_test.iloc[i].values
                    tdata = np.reshape(Featured_test.iloc[i].values, (hidden1, 1))
                    predict = AE2.predict(idata, hidden1)

                    e_2 += rss(tdata, predict)

                if errmin2 > e_2:
                    errmin2 = e_2
                    w1 = AE2.w_ih
                    w2 = AE2.w_ho
                    b1 = AE2.b_ih
                    b2 = AE2.b_ho

                else:
                    break

                print('|[MSAE2] Epoch: {} / {}, Error: {:.2f} |'.format(epoch+1, EPOCH, e_2))

        print('|       MSAE2 Last Error: {:.2f}        |'.format(errmin2))


        # Make 5 Layer AE
        Weight = {'w0': w0, 'w1': w1, 'w2': w2, 'w3': w3}
        Bias = {'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3}

        print('|         MS 5 Layer Tuning          |')
        NN5 = FiveLayerNN(Input, hidden1, hidden2, hidden3, Output, Weight, Bias, PARAMETER)

        if collect:
            Error_decay = []
        else:
            Error_decay = 0
        for epoch in range(EPOCH):

            for i in range(train_num):
                idata = x_train.iloc[i].values
                tdata = idata
                NN5.fit(idata, tdata, Input, Output)

            if k == div-1 and collect:
                # Error collect
                Error = []
                for i in range(train_num):
                    idata = x_train.iloc[i].values
                    tdata = np.reshape(x_train.iloc[i].values, (Input, 1))
                    out = NN5.predict(idata, Input)

                    Error.append(rss(tdata, out))

                # 各epochでの平均の誤差を格納する
                Error_decay.append(np.mean(Error))

            # Model Test
            e_5 = 0
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                for i in range(test_num):
                    idata = x_test.iloc[i].values
                    tdata = np.reshape(x_test.iloc[i].values, (Input, 1))
                    predict = NN5.predict(idata, Input)  # predict: 212×1

                    e_5 += rss(tdata, predict)

                if errmin5 > e_5:
                    errmin5 = e_5
                    w0 = NN5.w0
                    w1 = NN5.w1
                    w2 = NN5.w2
                    w3 = NN5.w3
                    b0 = NN5.b0
                    b1 = NN5.b1
                    b2 = NN5.b2
                    b3 = NN5.b3

                print('|[MSNN5] Epoch: {} / {} | Error: {:.2f} |'.format(epoch+1, EPOCH, e_5))

        print('|       MSNN5 Last Error: {:.2f}        |'.format(errmin5))
        print("")
        print("### --- ALL MS TRAINING FINISH --- ###")
        print("")

        # --- 5 Layer AE CHECK --- #
        if check:
            E_test = []
            Predict = []
            for i in range(test_num):
                tdata = np.reshape(x_test.iloc[i].values, (Input, 1))
                res = NN5.predict(x_test.iloc[i].values, Input)

                Predict.append(np.reshape(res, (Input,)))
                E_test.append(rss(tdata, res))

            E_per = np.mean(E_test)
            Predict = pd.DataFrame(Predict)
            CO_test = []
            for i in range(len(Predict)):
                R = np.corrcoef(x_test.iloc[i], Predict.iloc[i])
                CO_test.append(R[1, 0])
            CO_test = np.mean(CO_test)
            print('Performance for 5 Layer AE for MS')
            print('Corrcoef: {:.3f}, Error: {:.2f}'.format(CO_test, E_per))

            ERROR.append(E_per)
            CO.append(CO_test)

    print('=== AE5 MS Finish ! ===')

    return w0, w1, w2, w3, b0, b1, b2, b3


    # 変数の保存
    # Val = {'w0': w0,
    #        'w1': w1,
    #        'w2': w2,
    #        'w3': w3,
    #        'b0': b0,
    #        'b1': b1,
    #        'b2': b2,
    #        'b3': b3,
    #        'ERROR': ERROR,
    #        'CO': CO,
    #        'Error_decay': Error_decay}
    # np.save('AE_5NN_MS', Val)
    #
    # ### MAKE FIGURE ###
    # epochs = np.arange(1, EPOCH + 1)
    # plt.plot(epochs, Error_decay, label='Km: 60, Dm: 30')
    # plt.title('Error Reduction')
    # plt.xlabel('Epoch')
    # plt.ylabel('Reconstruction Error')
    # plt.legend()
    # # plt.show()
    # plt.savefig('Error_Reduction.png')

