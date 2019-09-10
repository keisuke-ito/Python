#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 16:00:42 2019

@author: itoukeisuke
"""

import numpy as np

class Regfunc:
    
    def __init__(self, REGULARIZATION):
        self.REGRESSION = REGULARIZATION['REGRESSION']
        self.LAMBDA = REGULARIZATION['LAMBDA']
        
        
    def select_func(self, x):
        if self.REGRESSION == 'RIDGE':
            function = self.LAMBDA * x
        
        elif self.REGRESSION == 'LASSO':
            function = self.LAMBDA * np.sign(x)
        
        elif self.REGRESSION == 'NONE':
            function = x
        
        else:
            print('Select REGRESSION (RIDGE, LASSO, NONE)')
        
        return function

    
    