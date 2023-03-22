#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 15:10:38 2023

This file is intended to compare our Method from OT vs EnKF in the survey paper 
"Ensemble Kalman Methods: A Mean Field Perspective" vs SIR particle filter.

The 1D system model is
    x(n+1) = A(x(n)) + sai(n) , sai ~ N(0,sigma^2 I)
    y(n+1) = h(x(n+1)) + eta(n+1) , eta ~ N(0,gamma^2 I)

A(x) = (1-alpha)* x    
h(x) = x , |x| , x^2 , x^3

@author: Mohammad Al-Jarrah
"""

import numpy as np
import matplotlib.pyplot as plt
#from scipy.integrate import odeint
import torch, math, time
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR, ExponentialLR
import sys
#%matplotlib auto

#%%

# =============================================================================
# np.random.seed(101)
# torch.manual_seed(101)
# =============================================================================


# Choose h(x) here, the observation rule
def h(x):
    #return x
    #return abs(x)
    return x*x
    #return x*x*x

# Choose A(x) here, the updates for the hidden state    
def A(x):
    return 0.9*np.eye(L) @ (x)

def Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma):
    sai = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),N)
    eta = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),N)

    x = np.zeros((N,L))
    y = np.zeros((N,dy))

    x0 = x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),1)
    x[0,] = x0

    # True system
    for i in range(N-1):
        x[i+1,] = A(x[i,]) + sai[i,]
        y[i+1,] = h(x[i+1,])+ eta[i+1,]
        
    return x,y


#%%    
L = 10 # number of states
tau = 0.1 # timpe step 
T = 5 # final time in seconds
N = int(T/tau) # number of time steps T = 20 s
dy = L # number of states observed
noise = np.sqrt(1e-1) # noise level std
sigmma = noise*2 # Noise in the hidden state
sigmma0 = noise # Noise in the initial state distribution
gamma = noise # Noise in the observation
x0_amp = 1/noise # Amplifiying the initial state 
J = 1000 # Number of ensembles EnKF
AVG_SIM = 1 # Number of Simulations to average over


# OT networks parameters
normalization = 'Mean' # Choose 'None' for nothing , 'Mean' for standard gaussian, 'MinMax' for d[0,1]
INPUT_DIM = [L,dy]
NUM_NEURON = int(32)
SAMPLE_SIZE = int(J) 
BATCH_SIZE =  int(32)
LearningRate = 1e-2
ITERATION = int(1024*10) 
Final_Number_ITERATION = int(1024)
#N_test = int(1e3)
Time_step = N 

t = np.arange(0.0, tau*N, tau)

    
#sys.exit()
#%% EnKF
start_time = time.time()
SAVE_X_EnKF =  np.zeros((AVG_SIM,N,L,J))
SAVE_True_X = np.zeros((AVG_SIM,N,L))
SAVE_True_Y = np.zeros((AVG_SIM,N,dy))
mse_EnKF =  np.zeros((N,AVG_SIM))
for k in range(AVG_SIM):
    
    x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma)
    SAVE_True_X[k,] = x
    SAVE_True_Y[k,] = y
    
    x_EnKF  = np.zeros((N,L,J))
    x_EnKF[0,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))
    x_hatEnKF = np.transpose(np.random.multivariate_normal(np.zeros(L),noise*noise * np.eye(L),J))
    y_hatEnKF = np.zeros((dy,J)) 
    
    m_hatEnKF = np.zeros(L)
    o_hatEnKF = np.zeros((N,dy))
    C_hat_vh = np.zeros((L,dy))
    C_hat_hh = np.zeros((dy,dy))
    
    K_EnKF = np.zeros((L,dy))
    
    SAVE_X_EnKF[k,0,:,:] = x_EnKF[0,]
    # EnKF & 3DVAR
    for i in range(N-1):
# =============================================================================
#         if (i+1)%50 == 0:
#             print("EnKF : Simu#%d/%d ,Time Step:%d/%d" %(k,AVG_SIM,i+1,N-1))
# =============================================================================
        sai_EnKF = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J).transpose()
            
        x_hatEnKF = A(x_EnKF[i,]) + sai_EnKF
            
        eta_EnKF = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),J)    
        #y_hatEnKF = (x_hatEnKF**3).transpose() + eta_EnKF
        y_hatEnKF = (h(x_hatEnKF)).transpose() + eta_EnKF
        
        m_hatEnKF = x_hatEnKF.mean(axis=1)
        #o_hatEnKF = (x_hatEnKF**3).mean(axis=1)
        o_hatEnKF = (h(x_hatEnKF)).mean(axis=1)
        
        a = (x_hatEnKF.transpose() - m_hatEnKF)
        #b = ((x_hatEnKF**3).transpose() - o_hatEnKF)
        b = (h(x_hatEnKF).transpose() - o_hatEnKF)
        C_hat_vh = np.matmul(a.transpose(),b)/J
        C_hat_hh = np.matmul(b.transpose(),b)/J
        K_EnKF = np.matmul(C_hat_vh,np.linalg.inv(C_hat_hh + np.eye(dy)*gamma*gamma))
            
        x_EnKF[i+1,:,:] = x_hatEnKF + np.matmul( K_EnKF, (y[i+1,:] - y_hatEnKF).transpose())
        
        SAVE_X_EnKF[k,i+1,:,:] = x_EnKF[i+1,]
    mse_EnKF[:,k] = ((x_EnKF.mean(axis=2)-x)*(x_EnKF.mean(axis=2)-x)).mean(axis=1)
MSE_EnKF = mse_EnKF.mean(axis=1)

print("--- EnKF time : %s seconds ---" % (time.time() - start_time))


#%% SIR Particle filter 
x_SIR =  np.zeros((AVG_SIM,N,L,J))
#W = np.zeros((AVG_SIM,N,L,J))
rng = np.random.default_rng()
for k in range(AVG_SIM):
    x_SIR[k,0,] = x0_amp*np.transpose(np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),J))
    
    for i in range(N-1):
        sai_SIR = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),J).transpose()
        x_SIR[k,i+1,] = A(x_SIR[k,i,]) + sai_SIR
        weight = np.exp(-np.sum((SAVE_True_Y[k,i+1,] - h(x_SIR[k,i+1,]).T)*(SAVE_True_Y[k,i+1] - h(x_SIR[k,i+1,]).T),axis=1)/(2*gamma*gamma)).T
        #W[k,i+1,] = weight/np.sum(weight)
        weight = weight/np.sum(weight)
        #x_SIR[k,i+1,0,] = rng.choice(x_SIR[k,i+1,0,], J, p = W[k,i+1,0,])
        index = rng.choice(np.arange(J), J, p = weight)
        x_SIR[k,i+1,] = x_SIR[k,i+1,:,index].T
        
# =============================================================================
# for l in range(2):        
#     for j in range(AVG_SIM):
#         #j = 35
#         plt.figure()
#         plt.subplot(2,1,1)
#         for i in range(J):
#             plt.plot(t,SAVE_X_EnKF[j,:,l,i],alpha = 0.2)
#         plt.plot(t,SAVE_True_X[j,:,l],'k--',label = 'dns')
#         plt.xlabel('time')
#         plt.ylabel('EnKF state')
#         plt.legend()
#         plt.show()
#     
#     
#         plt.subplot(2,1,2)
#     
#         for i in range(J):
#             plt.plot(t,x_SIR[j,:,l,i],alpha = 0.5)
#         plt.plot(t,SAVE_True_X[j,:,l ],'k--',label = 'dns')
#         plt.xlabel('time')
#         plt.ylabel('SIR state')
#         plt.legend()
#         plt.show()
# =============================================================================
        
#sys.exit()
#%% OT Setting
# =============================================================================
# # Device configuration, Make sure your device is running faster on 'mps' Check file 'compareMPSvsCPU.py'
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # GPU
# device = torch.device('mps' if torch.has_mps else 'cpu') # M1 Chip
# device = torch.device('cpu')
# =============================================================================

#device = torch.device('mps' if torch.has_mps else 'cpu') # M1 Chip
device = torch.device('cpu')

#%% NN , initialization and training    
class NeuralNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(NeuralNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activationSigmoid = nn.Sigmoid()
        self.activationReLu = nn.ReLU()
        self.layer_x = nn.Linear(self.input_dim[0], self.hidden_dim, bias=True)
        self.layer_y = nn.Linear(self.input_dim[1], self.hidden_dim, bias=True)
        self.layer = nn.Linear(self.hidden_dim, 1, bias=False)
        
    # Input is of size
    def forward(self, x,y):
        xx = self.layer_x(x)
        yy = self.layer_y(y)
        xy = self.activationReLu(xx+yy).square()
        xy = self.layer(xy) 
        return xy

class T_NeuralNet(nn.Module):
    
    def __init__(self, input_dim, hidden_dim):
        super(T_NeuralNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activationSigmoid = nn.Sigmoid()
        self.activationReLu = nn.ReLU()
        self.activationNonLinear = nn.Sigmoid()
        self.layer_x = nn.Linear(self.input_dim[0], self.hidden_dim, bias=False)
        self.layer_y = nn.Linear(self.input_dim[1], self.hidden_dim, bias=False)
        self.layer11 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer12 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer21 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layer22 = nn.Linear(self.hidden_dim, self.hidden_dim, bias=True)
        self.layerout = nn.Linear(self.hidden_dim, input_dim[0], bias=False)
        
# =============================================================================
#         self.m_x = torch.nn.Parameter(torch.randn(self.input_dim[0]))
#         self.m_y = torch.nn.Parameter(torch.randn(self.input_dim[1]))
#         self.A = torch.nn.Parameter(torch.rand(self.input_dim[0],self.input_dim[0]))
#         self.K = torch.nn.Parameter(torch.rand(self.input_dim[0],self.input_dim[1]))
#         self.m_x1 = torch.nn.Parameter(torch.randn(self.input_dim[0]))
#         self.m_y1 = torch.nn.Parameter(torch.randn(self.input_dim[1]))
#         self.A1 = torch.nn.Parameter(torch.rand(self.input_dim[0],self.input_dim[0]))
#         self.K1 = torch.nn.Parameter(torch.rand(self.input_dim[0],self.input_dim[1]))
# =============================================================================
        
    # Input is of size
    def forward(self, x,y):
        
# =============================================================================
#         xy = self.m_x + torch.matmul(x - self.m_x,self.A) + torch.matmul(y - self.m_y,self.K.transpose(0,1))
#         xy = xy + self.activationReLu(self.m_x1 + torch.matmul(x - self.m_x1,self.A1) + torch.matmul(y - self.m_y1,self.K1.transpose(0,1)))
# =============================================================================
        #xy = xy + self.activationSigmoid(self.m_x1 + torch.matmul(x - self.m_x1,self.A1) + torch.matmul(y - self.m_y1,self.K1.transpose(0,1))) 
        #xy =  self.layer(self.activationReLu(self.layer_x(x) + self.layer_y(y))) + xy
        
# =============================================================================
#         xy = self.layer_x(x) +self.layer_y(y)
#         xy = self.layerin(xy)
#         xy = self.activationReLu(xy)
#         xy = self.layer1 (xy)
#         xy = self.layerout(self.activationReLu(xy))
# =============================================================================
        
        X = self.layer_x(x) +self.layer_y(y)
        
        xy = self.layer11(X)
        xy = self.activationReLu(xy)
        xy = self.layer12 (xy)
        
        X = self.activationReLu(xy+X)
        
        xy = self.layer21(X)
        xy = self.activationReLu(xy)
        xy = self.layer22 (xy)
        
        X = self.layerout(self.activationReLu(xy+X))
        xy = X
        return xy  
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        #torch.nn.init.xavier_uniform_(m.weight)
        #torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.kaiming_normal_(m.weight,mode='fan_out', nonlinearity='relu')
        #torch.nn.init.kaiming_uniform_(m.weight,mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.fill_(0.1)

def train(f,T,X_Train,Y_Train, iterations,learning_rate,ts,Ts,batch_size,k,K):
    f.train()
    T.train()
    optimizer_T = torch.optim.Adam(T.parameters(), lr=learning_rate) 
    optimizer_f = torch.optim.Adam(f.parameters(), lr=learning_rate)
# =============================================================================
#     scheduler_f = ExponentialLR(optimizer_f, gamma=0.999) #set LR = 1e-1
#     scheduler_T = ExponentialLR(optimizer_T, gamma=0.999) #set LR = 1e-1
# =============================================================================
# =============================================================================
#     scheduler_f = StepLR(optimizer_f, step_size=50, gamma=0.9) #set LR = 1e-1
#     scheduler_T = StepLR(optimizer_T, step_size=50, gamma=0.9) #set LR = 1e-1
# =============================================================================
   
    inner_iterations = 10
    for i in range(iterations):
        idx = torch.randperm(X1.shape[0])[:batch_size]
        #X_train = torch.tensor(X_Train[idx])
        #Y_train = torch.tensor(Y_Train[idx])
        X_train = X_Train[idx].clone().detach()
        Y_train = Y_Train[idx].clone().detach()
        
        #X_train.requires_grad = True
        Y_shuffled = Y_train[torch.randperm(Y_train.shape[0])].view(Y_train.shape)
        #Y_shuffled.requires_grad = True
        for j in range(inner_iterations):
            map_T = T.forward(X_train,Y_shuffled)
            f_of_map_T= f.forward(map_T,Y_shuffled) 
            #grad_f_of_map_T = torch.autograd.grad(f_of_map_T.sum(),map_T,create_graph=True)[0]
            loss_T = f_of_map_T.mean() - (X_train*map_T).mean()
            optimizer_T.zero_grad()
            loss_T.backward()
            optimizer_T.step()
           
        f_of_xy = f.forward(X_train,Y_train) 
        map_T = T.forward(X_train,Y_shuffled)
        f_of_map_T= f.forward(map_T,Y_shuffled) 
        loss_f =f_of_xy.mean() - f_of_map_T.mean()
        optimizer_f.zero_grad()
        loss_f.backward()
        optimizer_f.step()
        loss = loss_f + (X_train*map_T).mean()
        
        if  (i+1)==iterations or i%50==0:
            #print(g.W.data)
            print("Simu#%d/%d ,Time Step:%d/%d, Iteration: %d/%d, loss = %.4f" %(k+1,K,ts,Ts-1,i+1,iterations,loss.item()))
            
        with torch.no_grad():
            f.layer.weight = torch.nn.parameter.Parameter(nn.functional.relu(f.layer.weight))
         
# =============================================================================
#         scheduler_f.step()
#         scheduler_T.step()
# =============================================================================
        

def Normalization(X,Type = 'None'):
    ''' Normalize Date with type 'MinMax' out data between [0,1] or 'Mean' for mean 0 and std 1 '''
    if Type == 'None':
        return 0,0,X
    elif Type == 'Mean':
        Mean_X_training_data = torch.mean(X)
        Std_X_training_data = torch.std(X)
        return Mean_X_training_data , Std_X_training_data , (X - Mean_X_training_data)/Std_X_training_data
    elif Type == 'MinMax':
        Min = torch.min(X) 
        Max = torch.max(X)
        return Min , Max , (X-Min)/(Max-Min)

        
def Transfer(M,S,X,Type='None'):
    '''Trasfer test Data to normalized data using knowledge of training data
    M = Mean/Min , S = Std/Max , X is data , Type = Mean/Min-Max Normalization '''
    if Type == 'None':
        return X
    elif Type == 'Mean':
        return (X - M)/S
    elif Type == 'MinMax':
        return (X - M)/(S - M)
    
def deTransfer(M,S,X , Type = 'None'):
    ''' Detransfer the normalized data to the origin set
     M = Mean/Min , S = Std/Max , X is data , Type = Mean/Min-Max Normalization'''  
    if Type == 'None':
        return X
    elif Type == 'Mean':
        return X*S + M
    elif Type == 'MinMax':
        return X*(S - M) + M
#%%
start_time = time.time()
SAVE_all_X_OT = np.zeros((AVG_SIM,Time_step,SAMPLE_SIZE,L))
# =============================================================================
# SAVE_True_X_OT = np.zeros((AVG_SIM,N,L))
# SAVE_True_Y_OT = np.zeros((AVG_SIM,N,dy))
# =============================================================================
mse_OT = np.zeros((Time_step,AVG_SIM))
#plt.figure()
for k in range(AVG_SIM):
    
# =============================================================================
#     x,y = Gen_Data(L,dy,N,x0_amp,sigmma0,sigmma,gamma)
# =============================================================================
    x = SAVE_True_X[k,]
    y = SAVE_True_Y[k,]
# =============================================================================
#     SAVE_True_X_OT[k,] = x
#     SAVE_True_Y_OT[k,] = y
# =============================================================================
    
    ITERS = ITERATION
    LR = LearningRate
    
    convex_f = NeuralNet(INPUT_DIM, NUM_NEURON)
    MAP_T = T_NeuralNet(INPUT_DIM, NUM_NEURON)
    MAP_T.apply(init_weights)
    with torch.no_grad():
        convex_f.layer.weight = torch.nn.parameter.Parameter(nn.functional.relu(convex_f.layer.weight))
    
    X0 = x0_amp*np.random.multivariate_normal(np.zeros(L),sigmma0*sigmma0 * np.eye(L),SAMPLE_SIZE)
    X1 = np.zeros((SAMPLE_SIZE,L))
    Y1 = np.zeros((SAMPLE_SIZE,dy))
    x_OT = np.zeros((Time_step,L))
    x_OT[0,:] = X0.mean(axis=0)
    SAVE_all_X_OT[k,0,:,:] = X0
    for time_step in range(1,Time_step):
             
        sai_train = np.random.multivariate_normal(np.zeros(L),sigmma*sigmma * np.eye(L),SAMPLE_SIZE).transpose()
     
        X1 = (A(X0.transpose()) + sai_train).transpose()    
        eta_train = np.random.multivariate_normal(np.zeros(dy),gamma*gamma * np.eye(dy),SAMPLE_SIZE)
        #Y1 = X1**3 + eta_train
        Y1 = h(X1) + eta_train
                
        X1_train = torch.from_numpy(X1)
        X1_train = X1_train.to(torch.float32)
        Y1_train = torch.from_numpy(Y1)
        Y1_train = Y1_train.to(torch.float32)
        X1_train = X1_train.to(device)
        Y1_train = Y1_train.to(device)
        
        #################################################
        MX, SX, X1_train = Normalization(X1_train,Type = normalization)
        MY, SY, Y1_train = Normalization(Y1_train,Type = normalization)
        
        train(convex_f,MAP_T,X1_train,Y1_train,ITERS,LR,time_step,Time_step,BATCH_SIZE,k,AVG_SIM)
        
# =============================================================================
#         if LR > 1e-4:
#             LR = LR/1.1
#         else:
#             LR = 1e-4
# =============================================================================
            
        if ITERS <= Final_Number_ITERATION :
            ITERS = Final_Number_ITERATION
        else:
            ITERS = int(ITERS/2)    
        
        Y1_true = y[time_step,:]
        Y1_true = torch.from_numpy(Y1_true)
        Y1_true = Y1_true.to(torch.float32)
        
        # Update X^(j) for the next time step
        X1_test = torch.from_numpy(X1).to(torch.float32).to(device)
        Y1_true = Y1_true.to(device)
        
        #################################################
        X1_test = Transfer(MX, SX, X1_test,Type = normalization)
        Y1_true = Transfer(MY, SY, Y1_true,Type = normalization)
        
        map_T = MAP_T.forward(X1_test, Y1_true)
        
        #################################################
        map_T = deTransfer(MX, SX, map_T,normalization)
        
        if device.type == 'mps':
            X0 = map_T.cpu().detach().numpy()
        else:
            X0 = map_T.detach().numpy()
        
        x_OT[time_step,:] = (torch.mean(map_T,dim=0)).detach().numpy()
        SAVE_all_X_OT[k,time_step,:,:] = map_T.detach().numpy()
        
        #Y_shuffled = Y1_train[torch.randperm(Y1_train.shape[0])].view(Y1_train.shape)
        #map_T_plot = MAP_T.forward(X1_test, Y_shuffled)
        
        mse_OT[:,k] =  ((x - x_OT)*(x - x_OT)).mean(axis=1)
# =============================================================================
#         plt.subplot(2,int((Time_step-1)/2) , time_step)
#         plt.scatter(X1_test.detach().numpy()[:,0],Y1_train.detach().numpy(), alpha = 0.5, label="$P_{xy}$")
#         plt.scatter(X1_train.detach().numpy()[:,0],Y_shuffled.detach().numpy(), alpha = 0.5,  marker = '*', label="$P_x \otimes P_y$")
#         plt.scatter(map_T_plot.detach().numpy()[:,0],Y_shuffled.detach().numpy(), alpha=0.5, label= "$(T,Id)\#_{P_x\otimes P_y}$")
# 
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.legend()        
# =============================================================================
#%%  
SAVE_all_X_OT = SAVE_all_X_OT.transpose((0,1,3,2))       
MSE_OT =  mse_OT.mean(axis=1)
print("--- %s seconds ---" % (time.time() - start_time))

#sys.exit()
#%%

plt.figure()
plt.plot(t,MSE_EnKF,'g-.',label="EnKF")
plt.plot(t,MSE_OT,'r:',label="OT" )
plt.xlabel('time')
plt.ylabel('mse')
plt.legend()
plt.show()

plt.figure()
plt.semilogy(t,MSE_EnKF,'g-.',label="EnKF")
plt.semilogy(t,MSE_OT,'r:',label="OT" )
plt.xlabel('time')
plt.ylabel('log(mse)')
plt.legend()
plt.show()

#%%
for l in range(L):
    for j in range(AVG_SIM):
        #j = 35
        plt.figure()
        plt.subplot(3,1,1)
        for i in range(J):
            plt.plot(t,SAVE_X_EnKF[j,:,l,i],alpha = 0.2)
        plt.plot(t,SAVE_True_X[j,:,l],'k--',label = 'dns')
        plt.xlabel('time')
        plt.ylabel('EnKF state')
        plt.legend()
        plt.show()
    
    
        plt.subplot(3,1,2)
    
        for i in range(SAMPLE_SIZE):
            plt.plot(t,SAVE_all_X_OT[j,:,l,i],alpha = 0.5)
        plt.plot(t,SAVE_True_X[j,:,l],'k--',label = 'dns')
        plt.xlabel('time')
        plt.ylabel('OT state')
        plt.legend()
        plt.show()
    
        plt.subplot(3,1,3)
    
        for i in range(J):
            plt.plot(t,x_SIR[j,:,l,i],alpha = 0.5)
        plt.plot(t,SAVE_True_X[j,:,l],'k--',label = 'dns')
        plt.xlabel('time')
        plt.ylabel('SIR state')
        plt.legend()
        plt.show()
# =============================================================================
# for j in range(AVG_SIM):
#     plt.figure()
#     plt.subplot(2,1,1)
# # =============================================================================
# #     for i in range(J):
# #         plt.plot(t, [j,:,:,i],alpha = 0.2)
# # =============================================================================
#     plt.plot(t,SAVE_True_Y[j, ],'k--',label = 'dns')
#     plt.xlabel('time')
#     plt.ylabel('EnKF state')
#     plt.legend()
#     plt.show()
# =============================================================================
    
sys.exit()
#%%
np.savez('3.13_DATA_File_Simulations_Linaer_abs_X_Normalized_2_BlocksResidual_100_simulations.npz',\
    time = t, Y_true = SAVE_True_Y,X_true = SAVE_True_X,\
    X_EnKF = SAVE_X_EnKF , X_OT = SAVE_all_X_OT , X_SIR = x_SIR,\
        MSE_EnKF = MSE_EnKF , MSE_OT=MSE_OT)

