import numpy as np
from main_9l_loopfunc import loopfunc

EPOCH = 1
Fine_Tuning = 1

H1_MS = np.arange(50, 10, 60)
H2_MS = np.arange(50, 10, 60)

H1_SD = np.arange(30, 10, 40)
H2_SD = np.arange(30, 10, 40)

H1_MLP = np.arange(40, 10, 50)
H2_MLP = np.arange(40, 10, 50)
H3_MLP = np.arange(40, 10, 50)

for i in H1_MS:
    loopfunc(EPOCH, Fine_Tuning, H1_MS, H2_MS, H1_SD, H2_SD, H1_MLP, H2_MLP, H3_MLP)