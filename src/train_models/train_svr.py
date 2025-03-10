# numpy
import numpy as np
#matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# load numpy array from csv file
from numpy import loadtxt
#sklearn
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
#tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.activations import linear,relu
from tensorflow.keras.optimizers import Adam

print('start')
#%% Loading data: HR images
#1#############################################################################
print('reading data from um1_dist150.csv ... ')
#load all data: X train and X test and y trainand y test
data_all_1 = loadtxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Data_3res_normDist/yX_um1_dist150.csv',delimiter=',')

print(f'the shape of the all data_1 (yX_dist150.csv) is : {data_all_1.shape}')

# X data
X_1 = data_all_1[:,1:]
print(f'X_1 shape : {X_1.shape}')

# y data
y_1 = data_all_1[:,0] 
print(f'y_1 shape : {y_1.shape}')

#Get 80% of the dataset as the training set. and 20% as the test set
X_train_1,X_test_1, y_train_1, y_test_1 = train_test_split(X_1,y_1,test_size = 0.2,random_state = 42)

print(f'the shape of the training set (input) is : {X_train_1.shape}')
print(f'the shape of the training set (target) is : {y_train_1.shape}')
print(f'the shape of the test set (input) is : {X_test_1.shape}')
print(f'the shape of the test set (target) is : {y_test_1.shape}')

#%% MR images
#2#############################################################################
print('reading data from yX_um2_dist150.csv ... ')
#load all data: X train and X test and y trainand y test
data_all_2 = loadtxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Data_3res_normDist/yX_um2_dist150.csv',delimiter=',')
print(f'the shape of the all data_2 (yX_um2_dist150.csv) is : {data_all_2.shape}')

# X data
X_2 = data_all_2[:,1:]
print(f'X_2 shape : {X_2.shape}')

# y data
y_2 = data_all_2[:,0] 
print(f'y_2 shape : {y_2.shape}')

#Get 80% of the dataset as the training set. and 20% as the test set
X_train_2,X_test_2, y_train_2, y_test_2 = train_test_split(X_2,y_2,test_size = 0.2,random_state = 42)

#print shape
print(f'the shape of the training set (input) is : {X_train_2.shape}')
print(f'the shape of the training set (target) is : {y_train_2.shape}')
print(f'the shape of the test set (input) is : {X_test_2.shape}')
print(f'the shape of the test set (target) is : {y_test_2.shape}')

#%% LR images
#3#############################################################################
print('reading data from yX_um3_dist150.csv ... ')
#load all data: X train and X test and y trainand y test
data_all_3 = loadtxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Data_3res_normDist/yX_um3_dist150.csv',delimiter=',')
print(f'the shape of the all data_3 (yX_um3_dist150.csv) is : {data_all_3.shape}')

# X data
X_3 = data_all_3[:,1:]
print(f'X_3 shape : {X_3.shape}')

# y data
y_3 = data_all_3[:,0] 
print(f'y_3 shape : {y_3.shape}')

#Get 80% of the dataset as the training set. and 20% as the test set
# X_train_3,X_test_3, y_train_3, y_test_3 = train_test_split(X_3,y_3,test_size = 0.2,random_state = 42)

# kfold = KFold(n_splits = 10, shuffle=True, random_state=42)

# #print shape
# print(f'the shape of the training set (input) is : {X_train_3.shape}')
# print(f'the shape of the training set (target) is : {y_train_3.shape}')
# print(f'the shape of the test set (input) is : {X_test_3.shape}')
# print(f'the shape of the test set (target) is : {y_test_3.shape}')

#%% HR Train SVR

# Instantiate the class
# scaler_svr = StandardScaler()
# Compute the mean and standard deviation of the training set then transform it

# X_train_scaled = scaler_svr.fit_transform(X_train_1)

# Initialize the class
model = SVR(kernel='rbf', degree=6, gamma='scale', tol=1e-3, C=159.27275, epsilon=0.01743 ,verbose=True)

# Train the model
# model.fit(X_train_scaled, y_train_1 )
model.fit(X_train_1, y_train_1 )

# Compute the training MSE
# yhat_1 = model.predict(X_train_scaled)
yhat_train_1 = model.predict(X_train_1)

# Record the training MSEs
train_MSE_1 = mean_squared_error(yhat_train_1,y_train_1) 
print(f"training MSE_1 = {train_MSE_1}")

# Record the training R2
R2_train_1 = r2_score(yhat_train_1,y_train_1)
print(f"training R2_1 = {R2_train_1}")

# X_test_scaled = scaler_svr.fit_transform(X_test_1)
# yhat_1 = model.predict(X_test_scaled)
yhat_test_1 = model.predict(X_test_1)

test_MSE_1 = mean_squared_error(yhat_test_1,y_test_1) 
print(f"test MSE_1 = {test_MSE_1}")

R2_test_1 = r2_score(yhat_test_1,y_test_1)
print(f"test R2_1 = {R2_test_1}")

#%% MR
# Instantiate the class
# scaler_svr = StandardScaler()
# Compute the mean and standard deviation of the training set then transform it

# X_train_scaled_2 = scaler_svr.fit_transform(X_train_2)

# Train the MR model
# model.fit(X_train_scaled_2, y_train_2 )
model.fit(X_train_2, y_train_2 )

# Compute the training MSE
# yhat_2 = model.predict(X_train_scaled_2)
yhat_train_2 = model.predict(X_train_2)

# Record the training MSEs
train_MSE_2 = mean_squared_error(y_train_2, yhat_train_2) 
print(f"training MSE_2 = {train_MSE_2}")

R2_train_2 = r2_score(yhat_train_2,y_train_2)
print(f"training R2_2 = {R2_train_2}")

# X_test_scaled_2 = scaler_svr.fit_transform(X_test_2)

# yhat_2 = model.predict(X_test_scaled_2)
yhat_test_2 = model.predict(X_test_2)

test_MSE_2 = mean_squared_error(yhat_test_2,y_test_2) 
print(f"test MSE_2 = {test_MSE_2}")

R2_test_2 = r2_score(yhat_test_2,y_test_2)
print(f"test R2_2 = {R2_test_2}")

#%% LR
 # Instantiate the class
# scaler_svr = StandardScaler()
    # Compute the mean and standard deviation of the training set then transform it

# X_train_scaled_3 = scaler_svr.fit_transform(X_train_3)

# Train the model
# model.fit(X_train_scaled_3, y_train_3 )
# model.fit(X_train_3, y_train_3 )

#     # Compute the training MSE
# # yhat_3 = model.predict(X_train_scaled_3)
# yhat_3 = model.predict(X_train_3)
#     # Record the training MSEs
# train_MSE_3 = mean_squared_error(y_train_3, yhat_3) 
# print(f"training MSE_3 = {train_MSE_3}")

# R2_train_3 = r2_score(yhat_3,y_train_3)
# print(f"training R2_3 = {R2_train_3}")
#    

# # X_test_scaled_3 = scaler_svr.fit_transform(X_test_3)

# # yhat_3 = model.predict(X_test_scaled_3)
# yhat_3 = model.predict(X_test_3)

# test_MSE_3 = mean_squared_error(yhat_3,y_test_3) 
# print(f"test MSE_3 = {test_MSE_3}")

# R2_test_3 = r2_score(yhat_3,y_test_3)
# print(f"test R2_3 = {R2_test_3}")
#==========================================================================
# KFold
# define the KFold cross validation object
kfold = KFold(n_splits=10, shuffle=True, random_state=42)


# evaluate the model using cross validation with r-squared and MSE metrics
r2_scores = cross_val_score(model, X_3, y_3, cv=kfold, scoring='r2')
mse_scores = cross_val_score(model, X_3, y_3, cv=kfold, scoring='neg_mean_squared_error') 
mse_scores = -mse_scores  # negate the MSE scores to make them positive

# print the mean and standard deviation of the r-squared and MSE scores
print("R-squared scores:", r2_scores)
print("Mean R-squared score:", r2_scores.mean())
print("Standard deviation of R-squared scores:", r2_scores.std())
print("MSE scores:", mse_scores)
print("Mean MSE score:", mse_scores.mean())
print("Standard deviation of MSE scores:", mse_scores.std())

#%%
yhat_3 = model.predict(X_3)
plt.plot(yhat_3, y_3,'o')
#%% Save Ys for plot: y_train_123, y_test_123, yhat_train_123, yhat_test_123
# HR
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/y_train_1.csv',y_train_1,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/yhat_train_1.csv',yhat_train_1,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/y_test_1.csv',y_test_1,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/yhat_test_1.csv',yhat_test_1,delimiter=',')

# MR
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/y_train_2.csv',y_train_2,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/yhat_train_2.csv',yhat_train_2,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/y_test_2.csv',y_test_2,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/yhat_test_2.csv',yhat_test_2,delimiter=',')

# LR
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/y_3.csv',y_3,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_SVR_Results/SaveYs_SVR/yhat_3.csv',yhat_3,delimiter=',')




















