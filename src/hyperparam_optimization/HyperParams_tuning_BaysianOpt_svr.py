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
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args


print('start')
#%%Loading data: 1-HR
#1##############################################################################
print('reading data from um1_dist150.csv ... ')
#load all data: X train and X test and y trainand y test
data_all_1 = loadtxt('/home/jafar/Nabipour/NeuralNetwork_Datasets-20230210T124646Z-001/NeuralNetwork_Datasets/Data_3res_normDist/yX_um1_dist150.csv',delimiter=',')
#data_all_1 = loadtxt('yX5_5.csv',delimiter=',')
print(f'the shape of the all data_1 (yX_dist150.csv) is : {data_all_1.shape}')
# X data
X_1 = data_all_1[:,1:]
print(f'X_1 shape : {X_1.shape}')
# y data
y_1 = data_all_1[:,0] 
print(f'y_1 shape : {y_1.shape}')

X_train_1,X_test_1, y_train_1, y_test_1 = train_test_split(X_1,y_1,test_size = 0.2,random_state = 42)

#print shape
print(f'the shape of the training set (input) is : {X_train_1.shape}')
print(f'the shape of the training set (target) is : {y_train_1.shape}')
print(f'the shape of the test set (input) is : {X_test_1.shape}')
print(f'the shape of the test set (target) is : {y_test_1.shape}')
#%%
# Tuning the hyperparameters of this model with Bayesian optimization method

# define the hyperparameters to tune and their search space
search_space = [
    Real(1e-4, 1e+4, name='C'),
    Real(1e-5, 1e-1, name='gamma'),
    Integer(2, 10, name='degree'),
    Real(1e-5, 2e-1, name='epsilon')
]

# define the objective function to minimize (in this case, the mean squared error)
@use_named_args(search_space)
def objective(**params):
    svr = SVR(kernel='rbf', **params)
    svr.fit(X_train_1, y_train_1)
    yhat_1 = svr.predict(X_test_1)
    
    mse = mean_squared_error(y_test_1, yhat_1)
    return mse

# perform Bayesian optimization to find the best hyperparameters
result = gp_minimize(objective, search_space, n_calls=50, random_state=42)

# print the best hyperparameters and their corresponding mean squared error
print("Best hyperparameters:", result.x)
print("Best mean squared error:", result.fun)


