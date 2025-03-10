# Import Libs:
# numpy
import numpy as np
#matplotlib
import matplotlib.pyplot as plt
# save numpy array as csv file
from numpy import asarray
from numpy import savetxt
# load numpy array from csv file
from numpy import loadtxt
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
from keras_tuner.tuners import BayesianOptimization
from keras_tuner import HyperModel
from tensorflow.keras import metrics

print('start')
#%%   Reading csv data and split them to define X, y in each resolution: HR images
#1#################################################################################
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

#print shape
print(f'the shape of the training set (input) is : {X_train_1.shape}')
print(f'the shape of the training set (target) is : {y_train_1.shape}')
print(f'the shape of the test set (input) is : {X_test_1.shape}')
print(f'the shape of the test set (target) is : {y_test_1.shape}')

#%%
print('#1########################################################################################################################')
# Normalization
print('normalization data ... ')
norm_1 = tf.keras.layers.Normalization( axis = -1)
norm_1.adapt(X_train_1)
X_train_n_1 = norm_1(X_train_1)

#%% Define the hyperparameters to be tuned:

tf.random.set_seed(42)

def build_model(hp):
    model = Sequential([
            tf.keras.Input(shape = (X_train_1.shape[1],)),
            Dense(units = hp.Int('units1', min_value=16, max_value=512, step=16), activation = 'relu' , kernel_regularizer = tf.keras.regularizers.l2(0.01), name = 'Layer1'),
            Dense(units = hp.Int('units2', min_value=16, max_value=512, step=16), activation = 'relu' , kernel_regularizer = tf.keras.regularizers.l2(0.01), name = 'Layer2'),
            Dense(units = hp.Int('units3', min_value=8, max_value=512, step=8), activation = 'relu' , kernel_regularizer = tf.keras.regularizers.l2(0.01), name = 'Layer3'),
            Dense(units = 1, activation = 'linear')
            ], name = "my_model")
    model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-3, 1e-4, 1e-5])),
                  loss='mse',
                  metrics=[metrics.MeanSquaredError(), metrics.MeanAbsoluteError()])
    return model


#%% Define the Bayesian optimization tuner:
tuner = BayesianOptimization(build_model,
                             objective='val_mean_squared_error',
                             max_trials=30,
                             directory='my_dir_Baysian',
                             project_name='my_project_Baysian')

#%% Perform hyperparameter tuning:
tuner.search(X_train_n_1, y_train_1, epochs=300, validation_split=0.2)

#%% Retrieve the best model and hyperparameters:
best_model = tuner.get_best_models(num_models=1)[0]
best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
#%% # Evaluate the best model on the test set
test_loss, test_mse, test_mae = best_model.evaluate(X_test_1, y_test_1)

print("Best hyperparameters: ")
print(best_hyperparameters.values)

print("Test loss: ", test_loss)
print("Test MSE: ", test_mse)
print("Test MAE: ", test_mae)
