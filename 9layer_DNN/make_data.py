import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

def MakeDataSD(div):
    sd_data = pd.read_csv('SD_DATA.csv', header=None) # SD : 121×144
    # 最大値の抽出
    max_data = sd_data.max().max()
    # 正則化
    norm_data = sd_data / max_data
    _, feature_num = norm_data.shape
    kf = KFold(n_splits=div, shuffle=True)
    X_train = []
    Y_train = []
    for train_index, test_index in kf.split(norm_data):
        X_train.append(norm_data.iloc[train_index])
        Y_train.append(norm_data.iloc[test_index])

    return X_train, Y_train, feature_num


def MakeDataMS(div):
    ms_data = pd.read_csv('MS_DATA.csv', header=None) # MS : 121 × 212
    ms_data = ms_data.values
    # 最大値の抽出
    max_data = ms_data.max()
    # 正則化
    norm_data = ms_data / max_data
    # 各スペクトルにピークが出るように正則化
    peak_data = norm_data.max(axis=1)
    norm_data = norm_data / peak_data[:,None]
    norm_data = pd.DataFrame(norm_data)
    _, feature_num = norm_data.shape
    kf = KFold(n_splits=div, shuffle=True)
    X_train = []
    Y_train = []
    for train_index, test_index in kf.split(norm_data):
        X_train.append(norm_data.iloc[train_index])
        Y_train.append(norm_data.iloc[test_index])

    return X_train, Y_train, feature_num





