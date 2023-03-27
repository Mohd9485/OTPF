#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 14:48:34 2023

@author: jarrah
"""
import numpy as np
import time
def SIR(X,Y,X0,A,h,noise=np.sqrt(1e-1)):
    AVG_SIM = X.shape[0]
    N = X.shape[1]
    L = X.shape[2]
    dy = Y.shape[2]
    J = X0.shape[2]
    sigmma = noise*2 # Noise in the hidden state
    sigmma0 = noise # Noise in the initial state distribution
    gamma = noise # Noise in the observation
    x0_amp = 1/noise # Amplifiying the initial state 
    start_time = time.time()
    x_SIR =  np.zeros((AVG_SIM,N,L,J))
    mse_SIR =  np.zeros((N,AVG_SIM))
    rng = np.random.default_rng()
    for k in range(AVG_SIM):
        x_SIR[k,0,] = X0[k,]
        x = X[k,]
        y = Y[k,]
        for i in range(N-1):
            sai_SIR = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J).transpose()
            x_SIR[k,i+1,] = A(x_SIR[k,i,]) + sai_SIR
            W = np.sum((y[i+1,] - h(x_SIR[k,i+1,]).T)*(y[i+1] - h(x_SIR[k,i+1,]).T),axis=1)/(2*gamma*gamma)
            W = W - np.min(W) # we add this step to avoid dividing by zero when h(x) = x^3
            weight = np.exp(-W).T
            #weight = np.exp(-np.sum((y[i+1,] - h(x_SIR[k,i+1,]).T)*(y[i+1] - h(x_SIR[k,i+1,]).T),axis=1)/(2*gamma*gamma)).T
            weight = weight/np.sum(weight)
            #x_SIR[k,i+1,0,] = rng.choice(x_SIR[k,i+1,0,], J, p = W[k,i+1,0,])
            index = rng.choice(np.arange(J), J, p = weight)
            x_SIR[k,i+1,] = x_SIR[k,i+1,:,index].T
        mse_SIR[:,k] = ((x_SIR[k,].mean(axis=2)-x)*(x_SIR[k,].mean(axis=2)-x)).mean(axis=1)
    MSE_SIR = mse_SIR.mean(axis=1)
    print("--- SIR time : %s seconds ---" % (time.time() - start_time))
    return x_SIR,MSE_SIR