import numpy as np
import pandas as pd
import ActiveFunc as AF

def Convert(TRAIN_MS, TEST_MS, TRAIN_SD, TEST_SD, dim1, dim2, Weight, Bias):
    train_num = len(TRAIN_MS)
    test_num = len(TEST_MS)

    # TRAIN DATA MS
    F_train_ms = []
    for i in range(train_num):
        o_i = np.reshape(TRAIN_MS.iloc[i].values, (dim1, 1))
        x_h1 = np.dot(Weight['w0'], o_i) + Bias['b0']
        o_h1 = AF.sigmoid(x_h1)
        x_h2 = np.dot(Weight['w1'], o_h1) + Bias['b1']
        o_h2 = AF.sigmoid(x_h2)

        F_train_ms.append(np.reshape(o_h2, (len(o_h2), )))


    # TEST DATA MS
    F_test_ms = []
    for i in range(test_num):
        o_i = np.reshape(TEST_MS.iloc[i].values, (dim1, 1))
        x_h1 = np.dot(Weight['w0'], o_i) + Bias['b0']
        o_h1 = AF.sigmoid(x_h1)
        x_h2 = np.dot(Weight['w1'], o_h1) + Bias['b1']
        o_h2 = AF.sigmoid(x_h2)

        F_test_ms.append(np.reshape(o_h2, (len(o_h2),)))


    # TRAIN DATA SD
    F_train_sd = []
    for i in range(train_num):
        o_i = np.reshape(TRAIN_SD.iloc[i].values, (dim2, 1))
        x_h1 = np.dot(Weight['w7'], o_i) + Bias['b7']
        o_h1 = AF.sigmoid(x_h1)
        x_h2 = np.dot(Weight['w6'], o_h1) + Bias['b6']
        o_h2 = AF.sigmoid(x_h2)

        F_train_sd.append(np.reshape(o_h2, (len(o_h2),)))


    # TEST DATA SD
    F_test_sd = []
    for i in range(test_num):
        o_i = np.reshape(TEST_SD.iloc[i].values, (dim2, 1))
        x_h1 = np.dot(Weight['w7'], o_i) + Bias['b7']
        o_h1 = AF.sigmoid(x_h1)
        x_h2 = np.dot(Weight['w6'], o_h1) + Bias['b6']
        o_h2 = AF.sigmoid(x_h2)

        F_test_sd.append(np.reshape(o_h2, (len(o_h2),)))


    F_train_ms = pd.DataFrame(F_train_ms)
    F_test_ms = pd.DataFrame(F_test_ms)
    F_train_sd = pd.DataFrame(F_train_sd)
    F_test_sd = pd.DataFrame(F_test_sd)

    return F_train_ms, F_test_ms, F_train_sd, F_test_sd

