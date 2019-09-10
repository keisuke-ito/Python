## Make Neural Network Class

import numpy as np
import ActiveFunc as AF
import Calculate_distance as cal
from Initialize import Random as R
from chooseRegFunc import Regfunc


## Three Layer NN
class ThreeLayerNN:

    def __init__(self, inodes, hnodes, onodes, PARAMETER):
        # The number of dimensions in each layers
        self.inodes = inodes
        self.hnodes = hnodes
        self.onodes = onodes

        # learning rate
        #        self.lr = lr
        self.ETA = PARAMETER['ETA']['RATE'] * (PARAMETER['ETA']['ADJUSTMENT'] ** PARAMETER['EPOCH'])

        # NOISE(scalar)
        self.NOISE = np.random.normal(0, PARAMETER['NOISE'] * np.sqrt(2 * self.ETA))

        # Initial values of between Weight and Bias(matrix)
        # 一様分布[-min ~ max]
        self.w_ho, self.b_ho, self.w_ih, self.b_ih = R(inodes, hnodes, onodes).uniformly(0.03, -0.03)
        # ガウス分布
        # self.w_ho, self.b_ho, self.w_ih, self.b_ih = R(inodes, hnodes, onodes).Gauss(0.1)

        # Regularization function(vector)
        self.reg = Regfunc(PARAMETER['REGULARIZATION'])
        # self.reg = PARAMETER['REGULARIZATION']

        # Momentum
        self.alpha = PARAMETER['ALPHA']

        self.dw_ih = np.zeros((self.hnodes, self.inodes))
        self.db_ih = np.zeros((self.hnodes, 1))
        self.dw_ho = np.zeros((self.onodes, self.hnodes))
        self.db_ho = np.zeros((self.onodes, 1))

        # Activation function
        self.af = AF.sigmoid
        self.daf = AF.derivative_sigmoid

        # Loss function


    ## Back Propagation ##
    def fit(self, idata, tdata, In_dim, Out_dim):  # idata, tdata: 101×212のうちの 1×212
        # Convert to vertical vector
        o_i = np.reshape(idata, (In_dim, 1))  # input -> hiddenへの出力 212×1
        t   = np.reshape(tdata, (Out_dim, 1))  # 教師データの作成 212×1

        # Hidden Layer
        x_h = np.dot(self.w_ih, o_i) + self.b_ih  # input -> hiddenへの入力
        o_h = self.af(x_h)

        # Output Layer
        x_o = np.dot(self.w_ho, o_h) + self.b_ho
        o_o = x_o

        # Calculate Error
        delta0 = cal.MSE_b(t, o_o)
        delta1 = np.dot(delta0.T, self.w_ho).T * self.daf(o_h)

        # Update of between Weight and Bias
        self.w_ho -=   self.ETA * (np.dot(delta0, o_h.T) + self.reg.select_func(self.w_ho)) + (np.dot(self.alpha, self.dw_ho)) + self.NOISE
        self.dw_ho = -(self.ETA * (np.dot(delta0, o_h.T) + self.reg.select_func(self.w_ho)) + (np.dot(self.alpha, self.dw_ho)) + self.NOISE)
        self.b_ho -=   self.ETA * (delta0 + self.reg.select_func(self.b_ho)) + (np.dot(self.alpha, self.db_ho)) + self.NOISE
        self.db_ho = -(self.ETA * (delta0 + self.reg.select_func(self.b_ho)) + (np.dot(self.alpha, self.db_ho)) + self.NOISE)

        self.w_ih -=   self.ETA * (np.dot(delta1, o_i.T) + self.reg.select_func(self.w_ih)) + (np.dot(self.alpha, self.dw_ih)) + self.NOISE
        self.dw_ih = -(self.ETA * (np.dot(delta1, o_i.T) + self.reg.select_func(self.w_ih)) + (np.dot(self.alpha, self.dw_ih)) + self.NOISE)
        self.b_ih -=   self.ETA * (delta1 + self.reg.select_func(self.b_ih)) + (np.dot(self.alpha, self.db_ih)) + self.NOISE
        self.db_ih = -(self.ETA * (delta1 + self.reg.select_func(self.b_ih)) + (np.dot(self.alpha, self.db_ih)) + self.NOISE)

    ## Feedforward Propagation ##
    def predict(self, idata, dimension):
        # Convert input vector to vertical vector
        o_i = np.reshape(idata, (dimension, 1))

        # Hidden Layer
        x_h = np.dot(self.w_ih, o_i) + self.b_ih
        o_h = self.af(x_h)

        # Output Layer
        x_o = np.dot(self.w_ho, o_h) + self.b_ho
        o_o = x_o

        return o_o

    def feed_makedata(self, idata, dimension):
        o_i = np.reshape(idata, (dimension, 1))
        x_h = np.dot(self.w_ih, o_i) + self.b_ih
        o_h = self.af(x_h)

        return o_h


## Four Layer NN
class FourLayerNN:
    # constracta
    def __init__(self, Input, hidden1, hidden2, Output, Weight, Bias, PARAMETER):
        self.Input = Input
        self.h1 = hidden1
        self.h2 = hidden2
        self.output = Output

        # self.lr = lr
        self.ETA = PARAMETER['ETA']['RATE'] * (PARAMETER['ETA']['ADJUSTMENT'] ** PARAMETER['EPOCH'])

        self.NOISE = np.random.normal(0, PARAMETER['NOISE'] * np.sqrt(2 * self.ETA))

        # input -> hidden1
        self.w0 = Weight['w0']
        self.w1 = Weight['w1']
        self.w2 = Weight['w2']
        self.b0 = Bias['b0']
        self.b1 = Bias['b1']
        self.b2 = Bias['b2']

        self.reg = Regfunc(PARAMETER['REGULARIZATION'])
        #        self.reg = PARAMETER['REGULARIZATION']

        self.alpha = PARAMETER['ALPHA']

        self.dw0 = np.zeros((self.h1, self.Input))
        self.db0 = np.zeros((self.h1, 1))
        self.dw1 = np.zeros((self.h2, self.h1))
        self.db1 = np.zeros((self.h2, 1))
        self.dw2 = np.zeros((self.output, self.h2))
        self.db2 = np.zeros((self.output, 1))

        self.af = AF.sigmoid
        self.daf = AF.derivative_sigmoid

    def fit(self, idata, tdata, dimension):
        o_i = np.reshape(idata, (dimension, 1))
        t = np.reshape(tdata, (dimension, 1))

        # Hidden1
        x_h1 = np.dot(self.w0, o_i) + self.b0  # input from input to hidden
        o_h1 = self.af(x_h1)

        # Hidden2
        x_h2 = np.dot(self.w1, o_h1) + self.b1
        o_h2 = self.af(x_h2)

        # Output
        x_o = np.dot(self.w2, o_h2) + self.b2
        o_o = x_o

        # Calculate Error
        delta0 = cal.MSE_b(t, o_o)
        delta1 = np.dot(delta0.T, self.w2).T * self.daf(o_h2)
        delta2 = np.dot(delta1.T, self.w1).T * self.daf(o_h1)

        # Update of Weight and Bias
        # Momentum sgd
        self.w2 -=   self.ETA * (np.dot(delta0, o_h2.T) + self.reg.select_func(self.w2)) + (np.dot(self.alpha, self.dw2)) + self.NOISE
        self.dw2 = -(self.ETA * (np.dot(delta0, o_h2.T) + self.reg.select_func(self.w2)) + (np.dot(self.alpha, self.dw2)) + self.NOISE)
        self.b2 -=   self.ETA * (delta0 + self.reg.select_func(self.b2)) + (np.dot(self.alpha, self.db2)) + self.NOISE
        self.db2 = -(self.ETA * (delta0 + self.reg.select_func(self.b2)) + (np.dot(self.alpha, self.db2)) + self.NOISE)

        self.w1 -=   self.ETA * (np.dot(delta1, o_h1.T) + self.reg.select_func(self.w1)) + (np.dot(self.alpha, self.dw1)) + self.NOISE
        self.dw1 = -(self.ETA * (np.dot(delta1, o_h1.T) + self.reg.select_func(self.w1)) + (np.dot(self.alpha, self.dw1)) + self.NOISE)
        self.b1 -=   self.ETA * (delta1 + self.reg.select_func(self.b1)) + (np.dot(self.alpha, self.db1)) + self.NOISE
        self.db1 = -(self.ETA * (delta1 + self.reg.select_func(self.b1)) + (np.dot(self.alpha, self.db1)) + self.NOISE)

        self.w0 -=   self.ETA * (np.dot(delta2, o_i.T) + self.reg.select_func(self.w0)) + (np.dot(self.alpha, self.dw0)) + self.NOISE
        self.dw0 = -(self.ETA * (np.dot(delta2, o_i.T) + self.reg.select_func(self.w0)) + (np.dot(self.alpha, self.dw0)) + self.NOISE)
        self.b0 -=   self.ETA * (delta2 + self.reg.select_func(self.b0)) + (np.dot(self.alpha, self.db0)) + self.NOISE
        self.db0 = -(self.ETA * (delta2 + self.reg.select_func(self.b0)) + (np.dot(self.alpha, self.db0)) + self.NOISE)

    def predict(self, idata, dimension):
        o_i = np.reshape(idata, (dimension, 1))

        # Hidden1
        x_h1 = np.dot(self.w0, o_i) + self.b0
        o_h1 = self.af(x_h1)

        # Hidden2
        x_h2 = np.dot(self.w1, o_h1) + self.b1
        o_h2 = self.af(x_h2)

        # Output
        x_ho = np.dot(self.w2, o_h2) + self.b2
        o_o = x_ho

        return o_o


## Five Layer NN
class FiveLayerNN:
    # constracta
    def __init__(self, Input, hidden1, hidden2, hidden3, Output, Weight, Bias, PARAMETER):

        self.Input = Input
        self.h1 = hidden1
        self.h2 = hidden2
        self.h3 = hidden3
        self.output = Output


        # self.lr = lr
        self.ETA = PARAMETER['ETA']['RATE'] * (PARAMETER['ETA']['ADJUSTMENT'] ** PARAMETER['EPOCH'])


        self.NOISE = np.random.normal(0, PARAMETER['NOISE'] * np.sqrt(2 * self.ETA))


        # input -> hidden1
        self.w0 = Weight['w0']
        self.w1 = Weight['w1']
        self.w2 = Weight['w2']
        self.w3 = Weight['w3']
        self.b0 = Bias['b0']
        self.b1 = Bias['b1']
        self.b2 = Bias['b2']
        self.b3 = Bias['b3']


        self.reg = Regfunc(PARAMETER['REGULARIZATION'])
        #        self.reg = PARAMETER['REGULARIZATION']


        self.alpha = PARAMETER['ALPHA']

        self.dw0 = np.zeros((self.h1, self.Input))
        self.db0 = np.zeros((self.h1, 1))
        self.dw1 = np.zeros((self.h2, self.h1))
        self.db1 = np.zeros((self.h2, 1))
        self.dw2 = np.zeros((self.h3, self.h2))
        self.db2 = np.zeros((self.h3, 1))
        self.dw3 = np.zeros((self.output, self.h3))
        self.db3 = np.zeros((self.output, 1))


        self.af = AF.sigmoid
        self.daf = AF.derivative_sigmoid


    def fit(self, idata, tdata, In_dim, Out_dim):

        o_i = np.reshape(idata, (In_dim, 1))
        t = np.reshape(tdata, (Out_dim, 1))

        # Hidden1
        x_h1 = np.dot(self.w0, o_i) + self.b0  # input from input to hidden
        o_h1 = self.af(x_h1)

        # Hidden2
        x_h2 = np.dot(self.w1, o_h1) + self.b1
        o_h2 = self.af(x_h2)

        # Hidden3
        x_h3 = np.dot(self.w2, o_h2) + self.b2
        o_h3 = self.af(x_h3)

        # Output
        x_o = np.dot(self.w3, o_h3) + self.b3
        o_o = x_o

        # Calculate Error
        delta0 = cal.MSE_b(t, o_o)
        delta1 = np.dot(delta0.T, self.w3).T * self.daf(o_h3)
        delta2 = np.dot(delta1.T, self.w2).T * self.daf(o_h2)
        delta3 = np.dot(delta2.T, self.w1).T * self.daf(o_h1)

        # Update of Weight and Bias
        # Momentum sgd
        self.w3 -=   self.ETA * (np.dot(delta0, o_h3.T) + self.reg.select_func(self.w3)) + (np.dot(self.alpha, self.dw3)) + self.NOISE
        self.dw3 = -(self.ETA * (np.dot(delta0, o_h3.T) + self.reg.select_func(self.w3)) + (np.dot(self.alpha, self.dw3)) + self.NOISE)
        self.b3 -=   self.ETA * (delta0 + self.reg.select_func(self.b3)) + (np.dot(self.alpha, self.db3)) + self.NOISE
        self.db3 = -(self.ETA * (delta0 + self.reg.select_func(self.b3)) + (np.dot(self.alpha, self.db3)) + self.NOISE)

        self.w2 -=   self.ETA * (np.dot(delta1, o_h2.T) + self.reg.select_func(self.w2)) + (np.dot(self.alpha, self.dw2)) + self.NOISE
        self.dw2 = -(self.ETA * (np.dot(delta1, o_h2.T) + self.reg.select_func(self.w2)) + (np.dot(self.alpha, self.dw2)) + self.NOISE)
        self.b2 -=   self.ETA * (delta1 + self.reg.select_func(self.b2)) + (np.dot(self.alpha, self.db2)) + self.NOISE
        self.db2 = -(self.ETA * (delta1 + self.reg.select_func(self.b2)) + (np.dot(self.alpha, self.db2)) + self.NOISE)

        self.w1 -=   self.ETA * (np.dot(delta2, o_h1.T) + self.reg.select_func(self.w1)) + (np.dot(self.alpha, self.dw1)) + self.NOISE
        self.dw1 = -(self.ETA * (np.dot(delta2, o_h1.T) + self.reg.select_func(self.w1)) + (np.dot(self.alpha, self.dw1)) + self.NOISE)
        self.b1 -=   self.ETA * (delta2 + self.reg.select_func(self.b1)) + (np.dot(self.alpha, self.db1)) + self.NOISE
        self.db1 = -(self.ETA * (delta2 + self.reg.select_func(self.b1)) + (np.dot(self.alpha, self.db1)) + self.NOISE)

        self.w0 -=   self.ETA * (np.dot(delta3, o_i.T) + self.reg.select_func(self.w0)) + (np.dot(self.alpha, self.dw0)) + self.NOISE
        self.dw0 = -(self.ETA * (np.dot(delta3, o_i.T) + self.reg.select_func(self.w0)) + (np.dot(self.alpha, self.dw0)) + self.NOISE)
        self.b0 -=   self.ETA * (delta3 + self.reg.select_func(self.b0)) + (np.dot(self.alpha, self.db0)) + self.NOISE
        self.db0 = -(self.ETA * (delta3 + self.reg.select_func(self.b0)) + (np.dot(self.alpha, self.db0)) + self.NOISE)


    def predict(self, idata, dimension):

        o_i = np.reshape(idata, (dimension, 1))

        # Hidden1
        x_h1 = np.dot(self.w0, o_i) + self.b0
        o_h1 = self.af(x_h1)

        # Hidden2
        x_h2 = np.dot(self.w1, o_h1) + self.b1
        o_h2 = self.af(x_h2)

        # Hidden3
        x_h3 = np.dot(self.w2, o_h2) + self.b2
        o_h3 = self.af(x_h3)

        # Output
        x_ho = np.dot(self.w3, o_h3) + self.b3
        o_o = x_ho

        return o_o


class SevenLayerNN:

    def __init__(self, inodes, hnodes, onodes, PARAMETER):
        pass



class NineLayerNN:
    # constracta
    def __init__(self, Input, hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7, Output, Weight, Bias, PARAMETER):
        self.Input = Input
        self.h1 = hidden1
        self.h2 = hidden2
        self.h3 = hidden3
        self.h4 = hidden4
        self.h5 = hidden5
        self.h6 = hidden6
        self.h7 = hidden7
        self.output = Output

        # self.lr = lr
        self.ETA = PARAMETER['ETA']['RATE'] * (PARAMETER['ETA']['ADJUSTMENT'] ** PARAMETER['EPOCH'])

        self.NOISE = np.random.normal(0, PARAMETER['NOISE'] * np.sqrt(2 * self.ETA))

        # input -> hidden1
        self.w0 = Weight['w0']
        self.w1 = Weight['w1']
        self.w2 = Weight['w2']
        self.w3 = Weight['w3']
        self.w4 = Weight['w4']
        self.w5 = Weight['w5']
        self.w6 = Weight['w6']
        self.w7 = Weight['w7']
        self.b0 = Bias['b0']
        self.b1 = Bias['b1']
        self.b2 = Bias['b2']
        self.b3 = Bias['b3']
        self.b4 = Bias['b4']
        self.b5 = Bias['b5']
        self.b6 = Bias['b6']
        self.b7 = Bias['b7']

        self.reg = Regfunc(PARAMETER['REGULARIZATION'])
        #        self.reg = PARAMETER['REGULARIZATION']

        self.alpha = PARAMETER['ALPHA']

        self.dw0 = np.zeros((self.h1, self.Input))
        self.db0 = np.zeros((self.h1, 1))
        self.dw1 = np.zeros((self.h2, self.h1))
        self.db1 = np.zeros((self.h2, 1))
        self.dw2 = np.zeros((self.h3, self.h2))
        self.db2 = np.zeros((self.h3, 1))
        self.dw3 = np.zeros((self.h4, self.h3))
        self.db3 = np.zeros((self.h4, 1))
        self.dw4 = np.zeros((self.h5, self.h4))
        self.db4 = np.zeros((self.h5, 1))
        self.dw5 = np.zeros((self.h6, self.h5))
        self.db5 = np.zeros((self.h6, 1))
        self.dw6 = np.zeros((self.h7, self.h6))
        self.db6 = np.zeros((self.h7, 1))
        self.dw7 = np.zeros((self.output, self.h7))
        self.db7 = np.zeros((self.output, 1))

        self.af = AF.sigmoid
        self.daf = AF.derivative_sigmoid

    def fit(self, idata, tdata, In_dim, Out_dim):
        o_i = np.reshape(idata, (In_dim, 1))
        t = np.reshape(tdata, (Out_dim, 1))

        # Hidden1
        x_h1 = np.dot(self.w0, o_i) + self.b0  # input from input to hidden
        o_h1 = self.af(x_h1)

        # Hidden2
        x_h2 = np.dot(self.w1, o_h1) + self.b1
        o_h2 = self.af(x_h2)

        # Hidden3
        x_h3 = np.dot(self.w2, o_h2) + self.b2
        o_h3 = self.af(x_h3)

        # Hidden4
        x_h4 = np.dot(self.w3, o_h3) + self.b3
        o_h4 = self.af(x_h4)

        # Hidden5
        x_h5 = np.dot(self.w4, o_h4) + self.b4
        o_h5 = self.af(x_h5)

        # Hidden6
        x_h6 = np.dot(self.w5, o_h5) + self.b5
        o_h6 = self.af(x_h6)

        # Hidden7
        x_h7 = np.dot(self.w6, o_h6) + self.b6
        o_h7 = self.af(x_h7)

        # Output
        x_o = np.dot(self.w7, o_h7) + self.b7
        o_o = x_o

        # Calculate Error
        delta0 = cal.MSE_b(t, o_o)
        delta1 = np.dot(delta0.T, self.w7).T * self.daf(o_h7)
        delta2 = np.dot(delta1.T, self.w6).T * self.daf(o_h6)
        delta3 = np.dot(delta2.T, self.w5).T * self.daf(o_h5)
        delta4 = np.dot(delta3.T, self.w4).T * self.daf(o_h4)
        delta5 = np.dot(delta4.T, self.w3).T * self.daf(o_h3)
        delta6 = np.dot(delta5.T, self.w2).T * self.daf(o_h2)
        delta7 = np.dot(delta6.T, self.w1).T * self.daf(o_h1)


        # Update of Weight and Bias
        # Momentum sgd
        self.w7 -=   self.ETA * (np.dot(delta0, o_h7.T) + self.reg.select_func(self.w7)) + (np.dot(self.alpha, self.dw7)) + self.NOISE
        self.dw7 = -(self.ETA * (np.dot(delta0, o_h7.T) + self.reg.select_func(self.w7)) + (np.dot(self.alpha, self.dw7)) + self.NOISE)
        self.b7 -=   self.ETA * (delta0 + self.reg.select_func(self.b7)) + (np.dot(self.alpha, self.db7)) + self.NOISE
        self.db7 = -(self.ETA * (delta0 + self.reg.select_func(self.b7)) + (np.dot(self.alpha, self.db7)) + self.NOISE)

        self.w6 -=   self.ETA * (np.dot(delta1, o_h6.T) + self.reg.select_func(self.w6)) + (np.dot(self.alpha, self.dw6)) + self.NOISE
        self.dw6 = -(self.ETA * (np.dot(delta1, o_h6.T) + self.reg.select_func(self.w6)) + (np.dot(self.alpha, self.dw6)) + self.NOISE)
        self.b6 -=   self.ETA * (delta1 + self.reg.select_func(self.b6)) + (np.dot(self.alpha, self.db6)) + self.NOISE
        self.db6 = -(self.ETA * (delta1 + self.reg.select_func(self.b6)) + (np.dot(self.alpha, self.db6)) + self.NOISE)

        self.w5 -=   self.ETA * (np.dot(delta2, o_h5.T) + self.reg.select_func(self.w5)) + (np.dot(self.alpha, self.dw5)) + self.NOISE
        self.dw5 = -(self.ETA * (np.dot(delta2, o_h5.T) + self.reg.select_func(self.w5)) + (np.dot(self.alpha, self.dw5)) + self.NOISE)
        self.b5 -=   self.ETA * (delta2 + self.reg.select_func(self.b5)) + (np.dot(self.alpha, self.db5)) + self.NOISE
        self.db5 = -(self.ETA * (delta2 + self.reg.select_func(self.b5)) + (np.dot(self.alpha, self.db5)) + self.NOISE)

        self.w4 -=   self.ETA * (np.dot(delta3, o_h4.T) + self.reg.select_func(self.w4)) + (np.dot(self.alpha, self.dw4)) + self.NOISE
        self.dw4 = -(self.ETA * (np.dot(delta3, o_h4.T) + self.reg.select_func(self.w4)) + (np.dot(self.alpha, self.dw4)) + self.NOISE)
        self.b4 -=   self.ETA * (delta3 + self.reg.select_func(self.b4)) + (np.dot(self.alpha, self.db4)) + self.NOISE
        self.db4 = -(self.ETA * (delta3 + self.reg.select_func(self.b4)) + (np.dot(self.alpha, self.db4)) + self.NOISE)

        self.w3 -=   self.ETA * (np.dot(delta4, o_h3.T) + self.reg.select_func(self.w3)) + (np.dot(self.alpha, self.dw3)) + self.NOISE
        self.dw3 = -(self.ETA * (np.dot(delta4, o_h3.T) + self.reg.select_func(self.w3)) + (np.dot(self.alpha, self.dw3)) + self.NOISE)
        self.b3 -=   self.ETA * (delta4 + self.reg.select_func(self.b3)) + (np.dot(self.alpha, self.db3)) + self.NOISE
        self.db3 = -(self.ETA * (delta4 + self.reg.select_func(self.b3)) + (np.dot(self.alpha, self.db3)) + self.NOISE)

        self.w2 -=   self.ETA * (np.dot(delta5, o_h2.T) + self.reg.select_func(self.w2)) + (np.dot(self.alpha, self.dw2)) + self.NOISE
        self.dw2 = -(self.ETA * (np.dot(delta5, o_h2.T) + self.reg.select_func(self.w2)) + (np.dot(self.alpha, self.dw2)) + self.NOISE)
        self.b2 -=   self.ETA * (delta5 + self.reg.select_func(self.b2)) + (np.dot(self.alpha, self.db2)) + self.NOISE
        self.db2 = -(self.ETA * (delta5 + self.reg.select_func(self.b2)) + (np.dot(self.alpha, self.db2)) + self.NOISE)

        self.w1 -=   self.ETA * (np.dot(delta6, o_h1.T) + self.reg.select_func(self.w1)) + (np.dot(self.alpha, self.dw1)) + self.NOISE
        self.dw1 = -(self.ETA * (np.dot(delta6, o_h1.T) + self.reg.select_func(self.w1)) + (np.dot(self.alpha, self.dw1)) + self.NOISE)
        self.b1 -=   self.ETA * (delta6 + self.reg.select_func(self.b1)) + (np.dot(self.alpha, self.db1)) + self.NOISE
        self.db1 = -(self.ETA * (delta6 + self.reg.select_func(self.b1)) + (np.dot(self.alpha, self.db1)) + self.NOISE)

        self.w0 -=   self.ETA * (np.dot(delta7, o_i.T) + self.reg.select_func(self.w0)) + (np.dot(self.alpha, self.dw0)) + self.NOISE
        self.dw0 = -(self.ETA * (np.dot(delta7, o_i.T) + self.reg.select_func(self.w0)) + (np.dot(self.alpha, self.dw0)) + self.NOISE)
        self.b0 -=   self.ETA * (delta7 + self.reg.select_func(self.b0)) + (np.dot(self.alpha, self.db0)) + self.NOISE
        self.db0 = -(self.ETA * (delta7 + self.reg.select_func(self.b0)) + (np.dot(self.alpha, self.db0)) + self.NOISE)

    def predict(self, idata, dimension):
        o_i = np.reshape(idata, (dimension, 1))

        # Hidden1
        x_h1 = np.dot(self.w0, o_i) + self.b0
        o_h1 = self.af(x_h1)

        # Hidden2
        x_h2 = np.dot(self.w1, o_h1) + self.b1
        o_h2 = self.af(x_h2)

        # Hidden3
        x_h3 = np.dot(self.w2, o_h2) + self.b2
        o_h3 = self.af(x_h3)

        # Hidden4
        x_h4 = np.dot(self.w3, o_h3) + self.b3
        o_h4 = self.af(x_h4)

        # Hidden5
        x_h5 = np.dot(self.w4, o_h4) + self.b4
        o_h5 = self.af(x_h5)

        # Hidden6
        x_h6 = np.dot(self.w5, o_h5) + self.b5
        o_h6 = self.af(x_h6)

        # Hidden7
        x_h7 = np.dot(self.w6, o_h6) + self.b6
        o_h7 = self.af(x_h7)

        # Output
        x_ho = np.dot(self.w7, o_h7) + self.b7
        o_o = x_ho

        return o_o
