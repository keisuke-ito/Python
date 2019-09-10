import os
import numpy as np
import pandas as pd
from make_data import MakeDataSD
from NN import ThreeLayerNN, FiveLayerNN
from matplotlib import pyplot as plt

# The residual sum of squares(残差平方和)
def rss(y, t):
    out_rss = np.sum(abs(y - t))
    return out_rss


if __name__ == '__main__':
    os.chdir('/Users/itoukeisuke/Python/pycharm/program_sd/')
    collect = True
    check = True

    # Make Data
    div = 6
    X_train, Y_train, dim = MakeDataSD(div)

    ## Parameter Setting
    EPOCH   = 100
    # EPOCH = 300
    Input   = dim
    hidden1 = 60
    hidden2 = 30
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
        print('| k : {} / {}                       |'.format(k+1, div))
        x_train = X_train[k]
        x_test = Y_train[k]
        train_num = len(x_train)
        test_num = len(x_test)

        # --- Training --- #
        # 1,3層目(Input -> hidden1, hidden3 -> Output)
        print('|          AE 1st Training         |')
        for epoch in range(EPOCH):
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                print('|[AE1]   --- EPOCH: {} / {} ---   |'.format(epoch+1, EPOCH))

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

                print('|[AE1] Epoch: {} / {} | Error: {:.2f} |'.format(epoch+1, EPOCH, e_1))

        print('|        AE1 Last Error: {:.2f}       |'.format(errmin1))


        # 2層目(hidden1 -> hidden2)
        print('|          AE 2nd Training         |')

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
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                print('|[AE2]   --- EPOCH: {} / {} ---   |'.format(epoch+1, EPOCH))

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

                print('|[AE2] Epoch: {} / {}, Error: {:.2f} |'.format(epoch+1, EPOCH, e_2))

        print('|       AE2 Last Error: {:.2f}        |'.format(errmin2))


        # Make 5 Layer AE
        Weight = {'w0': w0, 'w1': w1, 'w2': w2, 'w3': w3}
        Bias = {'b0': b0, 'b1': b1, 'b2': b2, 'b3': b3}

        print('|          5 Layer Tuning          |')
        NN5 = FiveLayerNN(Input, hidden1, hidden2, hidden3, Output, Weight, Bias, PARAMETER)

        Error_decay = []
        for epoch in range(EPOCH):
            if (epoch % PARAMETER['MOD_EPOCH']) == 0:
                print('|[NN5]   --- EPOCH: {} / {} ---   |'.format(epoch+1, EPOCH))

            for i in range(train_num):
                idata = x_train.iloc[i].values
                tdata = idata
                NN5.fit(idata, tdata, Input, Output)

            if k == (div-1) and collect:
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

                print('|[NN5] Epoch: {} / {} | Error: {:.2f} |'.format(epoch+1, EPOCH, e_5))

        print('|       NN5 Last Error: {:.2f}        |'.format(errmin5))
        print("")
        print("### --- ALL TRAINING FINISH --- ###")
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
            print('Performance for 5 Layer AE for SD')
            print('Corrcoef: {:.3f}, Error: {:.2f}'.format(CO_test, E_per))

            ERROR.append(E_per)
            CO.append(CO_test)

    print('=== All Finish ! ===')


    # 変数の保存
    Val = {'w0': w0,
           'w1': w1,
           'w2': w2,
           'w3': w3,
           'b0': b0,
           'b1': b1,
           'b2': b2,
           'b3': b3,
           'ERROR': ERROR,
           'CO': CO,
           'Error_decay': Error_decay}
    np.save('AE_5NN_SD', Val)

    ### MAKE FIGURE ###
    epochs = np.arange(1, EPOCH + 1)
    plt.plot(epochs, Error_decay, label='$K_{S}$: 60, $D_{S}$: 30')
    plt.title('Error Reduction')
    plt.xlabel('Epoch')
    plt.ylabel('Reconstruction Error')
    plt.legend()
    plt.show()
    plt.savefig('Error_Reduction_SD.png')

