import numpy as np
import os, sys
from main_ms_func import AE5_ms
from main_sd_func import AE5_sd
from main_mlp_func import MLP5
from main_9l_func import DNN9
from data_converter import Convert
from make_data import MakeDataMS, MakeDataSD
from send_mail import SEND_MAIL
from matplotlib import pyplot as plt

def loopfunc(E, F, h1_ms, h2_ms, h1_sd, h2_sd, h1_mlp, h2_mlp, h3_mlp):
    # os.chdir('/Users/itoukeisuke/Python/pycharm/9layer_NN/program/')
    # os.chdir('C:\\Users\\keisuke\\Dropbox\\Python\\9layer\\9layer_NN\\')
    # os.chdir('C:\\Users\\keisuke\\Python3\\Pycharm_Projects\\py37\\9layer_NN\\')
    os.chdir('/Users/itoukeisuke/Python/pycharm/9layer_NN/')

    # Make Data
    div = 6
    X_ms, Y_ms, dim1 = MakeDataMS(div) # divで分割されたデータセットが格納
    X_sd, Y_sd, dim2 = MakeDataSD(div)


    ## Parameter Setting

    EPOCH = E
    # EPOCH = 300
    Fine_Tuning = F

    # MS
    hidden1_ms = h1_ms
    hidden2_ms = h2_ms

    # SD
    hidden1_sd = h1_sd
    hidden2_sd = h2_sd

    # MLP
    hidden1_mlp = h1_mlp
    hidden2_mlp = h2_mlp
    hidden3_mlp = h3_mlp

    # 9-Layer DNN Parameter
    lr      = 0.4
    REGULARIZATION = {'REGRESSION': 'RIDGE', 'LAMBDA': 0.0000004}
    ETA            = {'RATE': lr, 'ADJUSTMENT': 0.99}
    PARAMETER = {'REGULARIZATION': REGULARIZATION,
                 'ETA': ETA,
                 'ALPHA': 0.15,
                 'NOISE': 0.000001,
                 'DISTANCE': 'MSE',
                 'EPOCH': Fine_Tuning,
                 'MOD_EPOCH': 5}

    for k in range(div):
        print("k = {} / {}".format(k, div))
        train_ms = X_ms[k]
        test_ms = Y_ms[k]
        train_sd = X_sd[k]
        test_sd = Y_sd[k]

        # MS and SD Autoencoder
        w0_ms, w1_ms, _, _, b0_ms, b1_ms, _, _ = AE5_ms(div, EPOCH, train_ms, test_ms, dim1, hidden1_ms, hidden2_ms)
        w0_sd, w1_sd, w2_sd, w3_sd, b0_sd, b1_sd, b2_sd, b3_sd = AE5_sd(div, EPOCH, train_sd, test_sd, dim2, hidden1_sd, hidden2_sd)

        Weight = {'w0': w0_ms, 'w1': w1_ms, 'w6': w1_sd, 'w7': w0_sd}
        Bias = {'b0': b0_ms, 'b1': b1_ms, 'b6': b1_sd,'b7': b0_sd}

        # MLP
        F_train_ms, F_test_ms, F_train_sd, F_test_sd = Convert(train_ms, test_ms, train_sd, test_sd, dim1, dim2, Weight, Bias)
        w0_mlp, w1_mlp, w2_mlp, w3_mlp, b0_mlp, b1_mlp, b2_mlp, b3_mlp = MLP5(div, EPOCH, F_train_ms, F_test_ms, F_train_sd, F_test_sd, hidden2_ms, hidden2_sd, hidden1_mlp, hidden2_mlp, hidden3_mlp)

        # 9-Layer DNN

        # Parameter Setting
        Weight['w2'] = w0_mlp
        Weight['w3'] = w1_mlp
        Weight['w4'] = w2_mlp
        Weight['w5'] = w3_mlp
        Weight['w6'] = w2_sd
        Weight['w7'] = w3_sd

        Bias['b2'] = b0_mlp
        Bias['b3'] = b1_mlp
        Bias['b4'] = b2_mlp
        Bias['b5'] = b3_mlp
        Bias['b6'] = b2_sd
        Bias['b7'] = b3_sd

        # Model Train
        w0, w1, w2, w3, w4, w5, w6, w7, b0, b1, b2, b3, b4, b5, b6, b7, ERROR, CO, Error_decay\
            = DNN9(k, div, train_ms, test_ms, train_sd, test_sd, hidden1_ms, hidden2_ms,\
                   hidden1_mlp, hidden2_mlp, hidden3_mlp, hidden2_sd, hidden1_sd,\
                   PARAMETER, Weight, Bias)


        print("")
        print("### --- ALL FINISH --- ###")
        print("")


    # 変数の保存
    Val = {'w0': w0,'w1': w1,'w2': w2,'w3': w3,
           'w4': w4,'w5': w5,'w6': w6,'w7': w7,
           'b0': b0,'b1': b1,'b2': b2,'b3': b3,
           'b4': b4,'b5': b5,'b6': b6,'b7': b7,
           'ERROR': ERROR,
           'CO': CO,
           'Error_decay': Error_decay}
    np.save('9L_model', Val)

    # ### MAKE FIGURE ###
    # epochs = np.arange(1, Fine_Tuning + 1)
    # plt.plot(epochs, Error_decay)
    # plt.title('Error Reduction')
    # plt.xlabel('Epoch')
    # plt.ylabel('Reconstruction Error')
    # # plt.legend()
    # plt.show()
    # plt.savefig('Error_Reduction.png')

    # send mail
    # SEND_MAIL('FINISH', 'About main_9l.py')





