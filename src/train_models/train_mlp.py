# Import Libs:

import numpy as np

import matplotlib.pyplot as plt
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# load numpy array from csv file
from numpy import loadtxt

import matplotlib.pyplot as plt

#sklearn
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
#%%   Reading csv data and split them to define X, y in each resolution: HR images
#1#################################################################################

#load all data: X train and X test and y trainand y test
data_all_1 = loadtxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Data_3res_normDist/yX_um1_dist150.csv',delimiter=',')

print(f'the shape of the all data_1 (yX_dist150.csv) is : {data_all_1.shape}')
# X data
X_1 = data_all_1[:,1:]
print(f'X_1 shape : {X_1.shape}')
# y data
y_1 = data_all_1[:,0] 
print(f'y_1 shape : {y_1.shape}')

#Get 80% of the dataset as the training set and 20% as the test set
X_train_1,X_test_1, y_train_1, y_test_1 = train_test_split(X_1,y_1,test_size = 0.2,random_state = 42)

#print shape
print(f'the shape of the training set (input) is : {X_train_1.shape}')
print(f'the shape of the training set (target) is : {y_train_1.shape}')
print(f'the shape of the test set (input) is : {X_test_1.shape}')
print(f'the shape of the test set (target) is : {y_test_1.shape}')

#%% MR images
#2########################################################################################################################
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
#3########################################################################################################################
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

#Get 80% of the dataset as the training set and 20% as the test set
# X_train_3,X_test_3, y_train_3, y_test_3 = train_test_split(X_3,y_3,test_size = 0.2,random_state = 42)

kfold = KFold(n_splits = 10, shuffle=True, random_state=42)

# #print shape
# print(f'the shape of the training set (input) is : {X_train_3.shape}')
# print(f'the shape of the training set (target) is : {y_train_3.shape}')
# print(f'the shape of the test set (input) is : {X_test_3.shape}')
# print(f'the shape of the test set (target) is : {y_test_3.shape}')

#%%

tf.random.set_seed(42)

# Define model
print('model ... ')
model = Sequential([
        tf.keras.Input(shape = (14400,)),
        Dense(units = 192, activation = 'relu' , kernel_regularizer = tf.keras.regularizers.l2(0.01), name = 'Layer1'),
        Dense(units = 416, activation = 'relu' , kernel_regularizer = tf.keras.regularizers.l2(0.01), name = 'Layer2'),
        Dense(units = 80, activation = 'relu' , kernel_regularizer = tf.keras.regularizers.l2(0.01), name = 'Layer3'),
        Dense(units = 1, activation = 'linear')
        ], name = "my_model")

# compile
print('compile ... ')
model.compile(
    loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(0.0001)
    )
#%%
print('#1########################################################################################################################')
# Normalization
print('normalization data ... ')
norm_1 = tf.keras.layers.Normalization( axis = -1)
norm_1.adapt(X_train_1)
X_train_n_1 = norm_1(X_train_1)
#%%
# fit
print('fit ... ')
history_1 = model.fit(
    X_train_n_1, y_train_1,
    epochs = 300
    )
#%%
# plot history
plt.plot(history_1.history['loss'])
loss_HR=history_1.history['loss']
savetxt('loss_HR.csv',loss_HR,delimiter=',')
plt.title('model loss 1')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
#%%
#W and b
print('W and b for yX_um1_dist150.csv: ')
#Layer 1 : W1 and b1 
R_L_1 = model.get_layer('Layer1')
W1,b1 = R_L_1.get_weights()
print(f'W1 shape = {W1.shape} , b1 shape = {b1.shape}:')
savetxt('W1_um1_dist150.csv',W1,delimiter=',')
savetxt('b1_um1_dist150.csv',b1,delimiter=',')

#Layer 2 : W2 and b2 
R_L_2 = model.get_layer('Layer2')
W2,b2 = R_L_2.get_weights()
print(f'W2 shape = {W2.shape} , b2 shape = {b2.shape}:')
savetxt('W2_um1_dist150.csv',W2,delimiter=',')
savetxt('b2_um1_dist150.csv',b2,delimiter=',')

#Layer 3 : W3 and b3 
R_L_3 = model.get_layer('Layer3')
W3,b3 = R_L_3.get_weights()
print(f'W3 shape = {W3.shape} , b3 shape = {b3.shape}:')
savetxt('W3_um1_dist150.csv',W3,delimiter=',')
savetxt('b3_um1_dist150.csv',b3,delimiter=',')
#%%
# Record the training MSEs
yhat_train_1 = model.predict(X_train_n_1)
train_error_1 = mean_squared_error(y_train_1, yhat_train_1) 
print(f"training error_1 = {train_error_1}")
R2_train_1 = r2_score(y_train_1, yhat_train_1)
print(f"training R2_1 = {R2_train_1}")

#Record the test MSEs
x_test_n_1 = norm_1(X_test_1)
yhat_test_1 = model.predict(x_test_n_1)
test_error_1 = mean_squared_error(y_test_1, yhat_test_1) 
print(f"test error_1 = {test_error_1}")
R2_test_1 = r2_score(y_test_1, yhat_test_1)
print(f"test R2_1 = {R2_test_1}")
#%%
print('#2########################################################################################################################')
# Normalization
print('normalization data ... ')
norm_2 = tf.keras.layers.Normalization( axis = -1)
norm_2.adapt(X_train_2)
X_train_n_2 = norm_2(X_train_2)
#%%
# fit
print('fit ... ')
history_2 = model.fit(
    X_train_n_2, y_train_2,
    epochs = 300
    )
#%%
# plot history
plt.plot(history_2.history['loss'])
loss_MR=history_2.history['loss']
savetxt('loss_MR.csv',loss_MR,delimiter=',')
plt.title('model loss 2')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
#%%
#W and b
print('W and b for yX_um2_dist150.csv: ')
#Layer 1 : W1 and b1 
R_L_1 = model.get_layer('Layer1')
W1,b1 = R_L_1.get_weights()
print(f'W1 shape = {W1.shape} , b1 shape = {b1.shape}:')
savetxt('W1_um2_dist150.csv',W1,delimiter=',')
savetxt('b1_um2_dist150.csv',b1,delimiter=',')

#Layer 2 : W2 and b2 
R_L_2 = model.get_layer('Layer2')
W2,b2 = R_L_2.get_weights()
print(f'W2 shape = {W2.shape} , b2 shape = {b2.shape}:')
savetxt('W2_um2_dist150.csv',W2,delimiter=',')
savetxt('b2_um2_dist150.csv',b2,delimiter=',')

#Layer 3 : W3 and b3 
R_L_3 = model.get_layer('Layer3')
W3,b3 = R_L_3.get_weights()
print(f'W3 shape = {W3.shape} , b3 shape = {b3.shape}:')
savetxt('W3_um2_dist150.csv',W3,delimiter=',')
savetxt('b3_um2_dist150.csv',b3,delimiter=',')
#%%
# Record the training MSEs
yhat_train_2 = model.predict(X_train_n_2)
train_error_2 = mean_squared_error(y_train_2, yhat_train_2) 
print(f"training error_2 = {train_error_2}")
R2_train_2 = r2_score(y_train_2, yhat_train_2)
print(f"training R2_2 = {R2_train_2}")

#Record the test MSEs
x_test_n_2 = norm_2(X_test_2)
yhat_test_2 = model.predict(x_test_n_2)
test_error_2 = mean_squared_error(y_test_2, yhat_test_2) 
print(f"test error_2 = {test_error_2}")
R2_test_2 = r2_score(y_test_2, yhat_test_2)
print(f"test R2_2 = {R2_test_2}")
#%% LR cross validation
print('#3########################################################################################################################')
# Normalization x_3 LR data
print('normalization LR data ... ')
norm_3 = tf.keras.layers.Normalization( axis = -1)
# norm_3.adapt(X_3)
# X_n_3 = norm_3(X_3)
#%% Apply cross valid for train LR data:

# Loop over the folds:
r2_scores=[]
mse_scores=[]
for train_idx, test_idx in kfold.split(X_3):
    X_train_3, y_train_3= X_3[train_idx], y_3[train_idx]
    X_test_3 , y_test_3 = X_3[test_idx], y_3[test_idx]
    # Normalization x_3 LR data
    norm_3.adapt(X_train_3)
    X_train_n_3 = norm_3(X_train_3)
    x_test_n_3 = norm_3(X_test_3)
    # Define and compile the NN model here
    # Fit the model on the LR train data
    print('fit LR data with cross validation... ')
    history_3= model.fit(X_train_n_3, y_train_3, epochs = 300)
    
    # Evaluate the model on LR test data:
    yhat_test_3 = model.predict(x_test_n_3)
    test_r2 = r2_score(y_test_3, yhat_test_3)
    r2_scores.append(test_r2)
    test_mse= mean_squared_error(y_test_3, yhat_test_3) 
    mse_scores.append(test_mse)

# 
yhat_train_3 = model.predict(X_train_n_3)
train_error_3 = mean_squared_error(y_train_3, yhat_train_3) 
print(f"training error_3 = {train_error_3}")
R2_train_3 = r2_score(y_train_3, yhat_train_3)
print(f"training R2_3 = {R2_train_3}")
#
print("K-Fold cross validation results: ")
print("R-Squares: ", np.mean(r2_scores)) 
print("Test_MSE: ", np.mean(mse_scores)) 
# fit
# print('fit ... ')
# history_3 = model.fit(
#     X_train_n_3, y_train_3,
#     epochs = 300
#     )
#%%
# plot history
plt.plot(history_3.history['loss'])
loss_LR=history_3.history['loss']
savetxt('loss_LR.csv',loss_LR,delimiter=',')
plt.title('model loss 3')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.show()
#%%
#W and b
print('W and b for yX_um3_dist150.csv: ')
#Layer 1 : W1 and b1 
R_L_1 = model.get_layer('Layer1')
W1,b1 = R_L_1.get_weights()
print(f'W1 shape = {W1.shape} , b1 shape = {b1.shape}:')
savetxt('W1_um3_dist150.csv',W1,delimiter=',')
savetxt('b1_um3_dist150.csv',b1,delimiter=',')

#Layer 2 : W2 and b2 
R_L_2 = model.get_layer('Layer2')
W2,b2 = R_L_2.get_weights()
print(f'W2 shape = {W2.shape} , b2 shape = {b2.shape}:')
savetxt('W2_um3_dist150.csv',W2,delimiter=',')
savetxt('b2_um3_dist150.csv',b2,delimiter=',')

#Layer 3 : W3 and b3 
R_L_3 = model.get_layer('Layer3')
W3,b3 = R_L_3.get_weights()
print(f'W3 shape = {W3.shape} , b3 shape = {b3.shape}:')
savetxt('W3_um3_dist150.csv',W3,delimiter=',')
savetxt('b3_um3_dist150.csv',b3,delimiter=',')
#%%
# # Record the training MSEs
# yhat_3 = model.predict(X_train_n_3)
# train_error_3 = mean_squared_error(y_train_3, yhat_3) 
# print(f"training error_3 = {train_error_3}")
# R2_train_3 = r2_score(y_train_3, yhat_3)
# print(f"training R2_3 = {R2_train_3}")

#Record the test MSEs
# x_test_n_3 = norm_3(X_test_3)
# yhat_3 = model.predict(x_test_n_3)
# test_error_3 = mean_squared_error(y_test_3, yhat_3) 
# print(f"test error_3 = {test_error_3}")
# R2_test_3 = r2_score(y_test_3, yhat_3)
# print(f"test R2_3 = {R2_test_3}")

#%% Save Ys for plot: y_train_123, y_test_123, yhat_train_123, yhat_test_123
# HR
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/y_train_1.csv',y_train_1,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/yhat_train_1.csv',yhat_train_1,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/y_test_1.csv',y_test_1,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/yhat_test_1.csv',yhat_test_1,delimiter=',')

# MR
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/y_train_2.csv',y_train_2,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/yhat_train_2.csv',yhat_train_2,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/y_test_2.csv',y_test_2,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/yhat_test_2.csv',yhat_test_2,delimiter=',')

# LR
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/y_train_3.csv',y_train_3,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/yhat_train_3.csv',yhat_train_3,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/y_test_3.csv',y_test_3,delimiter=',')
savetxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Results/Final_NN_Results/Saved_Ys_traintest/yhat_test_3.csv',yhat_test_3,delimiter=',')





