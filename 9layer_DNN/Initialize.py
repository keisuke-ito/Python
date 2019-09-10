#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 22:01:30 2019

@author: itoukeisuke
"""

# Make Random for initialize
import numpy as np
from scipy.stats import norm


class Random:
    
    def __init__(self, InputSize, HiddenSize, OutputSize):
        self.In = InputSize
        self.H = HiddenSize
        self.Out = OutputSize
        self.hi = HiddenSize * InputSize
        self.oh = OutputSize * HiddenSize
        
    def uniformly(self, Max, Min):
        self.w_ho = np.reshape(np.random.uniform(Min, Max, self.oh), (self.Out, self.H))
        self.b_ho = np.reshape(np.random.uniform(Min, Max, self.Out), (self.Out, 1))
        self.w_ih = np.reshape(np.random.uniform(Min, Max, self.hi), (self.H, self.In))
        self.b_ih = np.reshape(np.random.uniform(Min, Max, self.H), (self.H, 1))
        
        return self.w_ho, self.b_ho, self.w_ih, self.b_ih
        
    def Gauss(self, ELEMENT):
        self.w_ho = ELEMENT * norm.pdf((self.Out, self.H))
        self.b_ho = ELEMENT * norm.pdf((self.Out, 1))
        self.w_ih = ELEMENT * norm.pdf((self.H, self.In))
        self.b_ih = ELEMENT * norm.pdf((self.H, 1))
        
        return self.w_ho, self.b_ho, self.w_ih, self.b_ih
        