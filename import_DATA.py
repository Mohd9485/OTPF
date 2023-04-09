"""
Use this file to import the data and plot all the figure which were used in the 
CDC paper.
    
X(# simulations, # of time steps, # of states, # of samples or particles)
  
@author: Mohammad Al-Jarrah

"""

import numpy as np
import matplotlib.pyplot as plt
import sys

plt.rc('font', size=13)          # controls default text sizes
#plt.rc('axes', labelsize=16)    # fontsize of the x and y labels
def relu(x):
    a = 0
    return np.maximum(x,a)

## This is the data with using normalization in the training and testing
# load = np.load('DATA_file_Linaer_X_100_simulations.npz') # h(x) = x
# load = np.load('DATA_file_NonLinaer_XX_100_simulations.npz') # h(x) = x^2 
# load = np.load('DATA_file_NonLinaer_XXX_100_simulations.npz') # h(x) = x^3
 
load = np.load('DATA_file.npz') # h(x) = x^3


data = {}
for key in load:
    print(key)
    data[key] = load[key]
    
    
time = data['time']
X_true = data['X_true']
Y_true = data['Y_true']
X_EnKF = data['X_EnKF']
X_SIR = data['X_SIR']
X_OT = data['X_OT']
MSE_EnKF = data['MSE_EnKF']
MSE_OT = data['MSE_OT']
MSE_SIR = data['MSE_SIR']


#%%
AVG_SIM = X_OT.shape[0]
J = X_EnKF.shape[3]
SAMPLE_SIZE = X_OT.shape[3]
L = X_true.shape[2]

# =============================================================================
# for l in range(L):
#     for j in range(AVG_SIM):
#         plt.figure()
#         plt.subplot(3,1,1)
#         for i in range(J):
#             plt.plot(time,X_EnKF[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--',label = 'True state')
#         #plt.xlabel('time')
#         plt.ylabel('EnKF')
#         plt.title('j = {}'.format(j)) # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
#         plt.legend()
#         plt.show()
#         
#         #plt.figure()
#         plt.subplot(3,1,2)
#         #for j in range(1):
#         for i in range(SAMPLE_SIZE):
#             plt.plot(time,X_OT[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--',alpha=1)
#         #plt.xlabel('time')
#         plt.ylabel('OT')
#         #plt.legend()
#         plt.show()
#     
#         plt.subplot(3,1,3)
#         #for j in range(1):
#         for i in range(SAMPLE_SIZE):
#             plt.plot(time,X_SIR[j,:,l,i],'C0',alpha = 0.1)
#         plt.plot(time,X_true[j,:,l],'k--',alpha=1)
#         plt.xlabel('time')
#         plt.ylabel('SIR')
#         #plt.legend()
#         plt.show()
# =============================================================================

#%%

Performance_mse_Enkf = ((relu(X_EnKF).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) 
Performance_mse_OT = ((relu(X_OT).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) 
Performance_mse_SIR = ((relu(X_SIR).mean(axis = 3) - relu(X_true))**2).mean(axis=(0,2)) 

# =============================================================================
# plt.figure()
# plt.plot(time,MSE_EnKF,'g-.',label="EnKF")
# plt.plot(time,MSE_OT,'r:',label="OT" )
# plt.plot(time,MSE_SIR,'b:',label="SIR" )
# plt.xlabel('time')
# plt.ylabel('mse')
# plt.title('MSE')
# plt.legend()
# plt.show()
# =============================================================================


# =============================================================================
# plt.figure()
# plt.subplot(1,2,1)
# plt.semilogy(time,MSE_EnKF,'g-.',label="EnKF")
# plt.semilogy(time,MSE_OT,'r:',label="OT" )
# plt.semilogy(time,MSE_SIR,'b:',label="SIR" )
# plt.xlabel('time')
# plt.ylabel('mse')
# plt.title('h(X_t) = $X^2_t$')
# plt.legend()
# plt.show()
# 
# plt.subplot(1,2,2)
# #plt.figure()
# # =============================================================================
# # plt.semilogy(time,Performance_mse_Enkf,'g-.',label="EnKF")
# # plt.semilogy(time,Performance_mse_OT,'r:',label="OT" )
# # =============================================================================
# plt.plot(time,Performance_mse_Enkf,'g-.',label="EnKF")
# plt.plot(time,Performance_mse_OT,'r:')
# plt.plot(time,Performance_mse_SIR,'b:')
# plt.xlabel('time')
# plt.ylabel('mse')
# plt.title('h(X_t) = $X^2_t$') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
# plt.legend()
# plt.show()
# =============================================================================



# =============================================================================
# sys.exit()
# =============================================================================
#%%
l = 0
j=0 # 88
plt.figure()
plt.subplot(3,1,1)
for i in range(J):
    plt.plot(time,X_EnKF[j,:,l,i],'C0',alpha = 0.1)
plt.plot(time,X_true[j,:,l],'k--',label = 'True state')
plt.xlabel('time')
plt.ylabel('EnKF')
#plt.title('State$') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%
plt.legend()
plt.show()

#plt.figure()
plt.subplot(3,1,2)
#for j in range(1):
for i in range(SAMPLE_SIZE):
    plt.plot(time,X_OT[j,:,l,i],'C0',alpha = 0.1)
plt.plot(time,X_true[j,:,l],'k--',alpha=1)
plt.xlabel('time')
plt.ylabel('OT')
plt.legend()
plt.show()

plt.subplot(3,1,3)
#for j in range(1):
for i in range(SAMPLE_SIZE):
    plt.plot(time,X_SIR[j,:,l,i],'C0',alpha = 0.1)
plt.plot(time,X_true[j,:,l],'k--',alpha=1)
plt.xlabel('time')
plt.ylabel('SIR')
plt.legend()
plt.show()

#plt.savefig('NonLinearState_XX.pdf') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%


plt.figure()
#plt.subplot(1,2,1)
plt.semilogy(time,MSE_EnKF,'g--',label="EnKF")
plt.semilogy(time,MSE_OT,'r-.',label="OT" )
plt.semilogy(time,MSE_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('mse')
plt.title('Log-MSE')
plt.legend()
plt.show()
#plt.savefig('NonLinearMSE_XXX.pdf') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%

#sys.exit()

plt.figure()
#plt.figure()
# =============================================================================
# plt.semilogy(time,Performance_mse_Enkf,'g-.',label="EnKF")
# plt.semilogy(time,Performance_mse_OT,'r:',label="OT" )
# =============================================================================
plt.plot(time,Performance_mse_Enkf,'g--',label="EnKF")
plt.plot(time,Performance_mse_OT,'r-.',label="OT" )
plt.plot(time,Performance_mse_SIR,'b:',label="SIR" )
plt.xlabel('time')
plt.ylabel('mse')
plt.title('mse=max(0,X)')
plt.legend()
plt.show()
#plt.savefig('NonLinearPerformance_XX.pdf') # Change this %%%%%%%%%%%%%%%%%%%%%%%%%%




