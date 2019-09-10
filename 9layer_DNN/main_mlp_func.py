import pandas as pd
import numpy as np
from NN import FiveLayerNN, ThreeLayerNN

# The residual sum of squares(残差平方和)
def rss(y, t):
    out_rss = np.sum(abs(y - t))
    return out_rss

def MLP5(div_pre, E, train_ms, test_ms, train_sd, test_sd, dim1, dim2, H1, H2, H3):
    # os.chdir('/Users/itoukeisuke/Python/pycharm/5layer_AEMS/program/')
    collect = False
    check = False

    # Make Data
    if div_pre != '':
        div = 1
    else:
        div = 6


    ## Parameter Setting
    EPOCH   = E
    # EPOCH = 300
    Input   = dim1
    hidden1 = H1
    hidden2 = H2
    hidden3 = H3
    Output  = dim2
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
    MLP1 = ThreeLayerNN(Input, hidden1, Input, PARAMETER)
    MLP2 = ThreeLayerNN(hidden1, hidden2, hidden1, PARAMETER)
    MLP3 = ThreeLayerNN(Output, hidden3, Output, PARAMETER)

    # Pre Setting
    errmin1 = np.inf
    errmin2 = np.inf
    errmin3 = np.inf
    errmin5 = np.inf
    ERROR = []
    CO = []

    # k-fold Cross Validation
    for k in range(div):
        # print('| k : {} / {}                       |'.format(k+1, div))
        x_train = train_ms
        y_train = train_sd
        x_test = test_ms
        y_test = test_sd

        train_num = len(x_train)
        test_num = len(x_test)

        # --- Training --- #
        # 1層目(Input -> hidden1)
        print('|          MLP 1st Training         |')
        for epoch in range(EPOCH):
             # Model Fit #
            for i in range(train_num):
                idata = x_train.iloc[i].values
                tdata = y_train.iloc[i].values
                MLP1.fit(idata, tdata, Input, Input)

            # Model Test #
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                e_1 = 0
                for i in range(test_num):
                    idata = x_test.iloc[i].values
                    tdata = np.reshape(y_test.iloc[i].values, (Input, 1))
                    predict = MLP1.predict(idata, Input)  # predict: 144×1

                    e_1 += rss(tdata, predict)

                if errmin1 > e_1:
                    errmin1 = e_1
                    w0 = MLP1.w_ih
                    b0 = MLP1.b_ih
                else:
                    break

                print('|[MLP1] Epoch: {} / {} | Error: {:.2f} |'.format(epoch+1, EPOCH, e_1))

        print('|        MLP1 Last Error: {:.2f}       |'.format(errmin1))


        # 3層目(Output -> hidden3)
        print('|          MLP 3rd Training         |')
        for epoch in range(EPOCH):
             # Model Fit #
            for i in range(train_num):
                idata = x_train.iloc[i].values
                tdata = y_train.iloc[i].values
                MLP3.fit(idata, tdata, Output, Output)

            # Model Test #
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                e_3 = 0
                for i in range(test_num):
                    idata = x_test.iloc[i].values
                    tdata = np.reshape(y_test.iloc[i].values, (Output, 1))
                    predict = MLP3.predict(idata, Output)

                    e_3 += rss(tdata, predict)

                if errmin3 > e_3:
                    errmin3 = e_3
                    w3 = MLP3.w_ho
                    b3 = MLP3.b_ho
                else:
                    break

                print('|[MLP3] Epoch: {} / {} | Error: {:.2f} |'.format(epoch+1, EPOCH, e_3))

        print('|        MLP3 Last Error: {:.2f}       |'.format(errmin3))


        # 2層目(hidden1 -> hidden2)
        print('|          MLP 2nd Training         |')

        # make data
        Featured_train = []
        for i in range(train_num):
            Featured_train.append(np.reshape(MLP1.feed_makedata(x_train.iloc[i].values, Input), (hidden1,)))

        Featured_test = []
        for i in range(test_num):
            Featured_test.append(np.reshape(MLP3.feed_makedata(y_test.iloc[i].values, Input), (hidden1,))) # 出力: hidden1 × 1

        # Featured_train/test : sample * feature
        Featured_train = pd.DataFrame(Featured_train)
        Featured_test  = pd.DataFrame(Featured_test)

        for epoch in range(EPOCH):

            # Model Fit #
            for i in range(train_num):
                idata = Featured_train.iloc[i].values
                tdata = idata
                MLP2.fit(idata, tdata, hidden1, hidden1)

            # Model Test #
            e_2 = 0
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                for i in range(test_num):
                    idata = Featured_test.iloc[i].values
                    tdata = np.reshape(Featured_test.iloc[i].values, (hidden1, 1))
                    predict = MLP2.predict(idata, hidden1)

                    e_2 += rss(tdata, predict)

                if errmin2 > e_2:
                    errmin2 = e_2
                    w1 = MLP2.w_ih
                    w2 = MLP2.w_ho
                    b1 = MLP2.b_ih
                    b2 = MLP2.b_ho

                else:
                    break

                print('|[MLP2] Epoch: {} / {}, Error: {:.2f} |'.format(epoch+1, EPOCH, e_2))

        print('|       MLP2 Last Error: {:.2f}        |'.format(errmin2))


        # Make 5 Layer AE
        Weight = {'w0': w0, 'w1': w1, 'w2': w2, 'w3': w3}
        Bias = {'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3}

        print('|          5 Layer Tuning          |')
        MLP5 = FiveLayerNN(Input, hidden1, hidden2, hidden3, Output, Weight, Bias, PARAMETER)

        Error_decay = []
        for epoch in range(EPOCH):

            for i in range(train_num):
                idata = x_train.iloc[i].values
                tdata = y_train.iloc[i].values
                MLP5.fit(idata, tdata, Input, Output)

            if k == (div-1) and collect:
                # Error collect
                Error = []
                for i in range(train_num):
                    idata = x_train.iloc[i].values
                    tdata = np.reshape(y_train.iloc[i].values, (Output, 1))
                    out = MLP5.predict(idata, Input)

                    Error.append(rss(tdata, out))

                # 各epochでの平均の誤差を格納する
                Error_decay.append(np.mean(Error))

            # Model Test
            e_5 = 0
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                for i in range(test_num):
                    idata = x_test.iloc[i].values
                    tdata = np.reshape(y_test.iloc[i].values, (Output, 1))
                    predict = MLP5.predict(idata, Input)  # predict: 212×1

                    e_5 += rss(tdata, predict)

                if errmin5 > e_5:
                    errmin5 = e_5
                    w0 = MLP5.w0
                    w1 = MLP5.w1
                    w2 = MLP5.w2
                    w3 = MLP5.w3
                    b0 = MLP5.b0
                    b1 = MLP5.b1
                    b2 = MLP5.b2
                    b3 = MLP5.b3

                print('|[MLP5] Epoch: {} / {} | Error: {:.2f} |'.format(epoch+1, EPOCH, e_5))

        print('|       MLP5 Last Error: {:.2f}        |'.format(errmin5))
        print("")
        print("### --- ALL TRAINING FINISH --- ###")
        print("")

        # --- 5 Layer AE CHECK --- #
        if check:
            E_test = []
            Predict = []
            for i in range(test_num):
                tdata = np.reshape(y_test.iloc[i].values, (Output, 1))
                res = MLP5.predict(x_test.iloc[i].values, Input)

                Predict.append(np.reshape(res, (Output,)))
                E_test.append(rss(tdata, res))

            E_per = np.mean(E_test)
            Predict = pd.DataFrame(Predict)
            CO_test = []
            for i in range(len(Predict)):
                R = np.corrcoef(y_test.iloc[i], Predict.iloc[i])
                CO_test.append(R[1, 0])
            CO_test = np.mean(CO_test)
            print('Performance for 5 Layer AE for MS')
            print('Corrcoef: {:.3f}, Error: {:.2f}'.format(CO_test, E_per))

            ERROR.append(E_per)
            CO.append(CO_test)

    print('=== All MLP Finish ! ===')

    return w0, w1, w2, w3, b0, b1, b2, b3


    # # 変数の保存
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
    # np.save('MLP_5NN', Val)
    #
    # ### MAKE FIGURE ###
    # epochs = np.arange(1, EPOCH + 1)
    # plt.plot(epochs, Error_decay, label='Km: 60, Dm: 30')
    # plt.title('Error Reduction')
    # plt.xlabel('Epoch')
    # plt.ylabel('Reconstruction Error')
    # plt.legend()
    # # plt.show()
    # plt.savefig('Error_Reduction_MLP.png')
    #
