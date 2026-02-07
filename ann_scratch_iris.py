# \brucedu\py_conda_venvs\venv1\ann_scratch_iris.py, B. Harmon 5/15/2020
# See also ma1.py, percept_iris.py, and Medium 1/28: ML from scratch, one hidden layer
# Iris with 2 features and binary output

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# activation function for output neuron
sigmoid = lambda x: 1 / (1 + np.exp(-x))  # output range 0 to 1

def init_parameters(n_x, n_h, n_y):
    W1=np.random.randn(n_h, n_x)
    b1=np.zeros((n_h, 1))
    W2=np.random.randn(n_y, n_h)
    b2=np.zeros((n_y, 1))
    parameters={'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return parameters

def forward_prop(X, parameters):
    W1=parameters['W1']; b1=parameters['b1']
    W2=parameters['W2']; b2=parameters['b2']
    Z1=np.dot(W1, X) + b1
    A1=np.tanh(Z1)                        # tanh range -1 to 1
    Z2=np.dot(W2, A1) + b2
    A2=sigmoid(Z2)
    cache={'A1': A1, 'A2': A2}
    return A2, cache

# Cost or loss function is binary cross-entropy loss
def calculate_cost(A2, y):
    cost= -np.sum(np.multiply(y, np.log(A2)) + np.multiply(1-y, np.log(1-A2)))/m
    return np.squeeze(cost)

def backward_prop(X, y, cache, parameters):
    A1=cache['A1'];  A2=cache['A2']
    W2=parameters['W2']
    dZ2=A2-y
    dW2=np.dot(dZ2, A1.T)/m
    db2=np.sum(dZ2, axis=1, keepdims=True)/m
    dZ1=np.multiply(np.dot(W2.T, dZ2), 1-np.power(A1,2))
    dW1=np.dot(dZ1, X.T)/m
    db1=np.sum(dZ1, axis=1, keepdims=True)/m
    grads={'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
    return grads

def update_parameters(parameters, grads, learning_rate):
    W1=parameters['W1']; b1=parameters['b1']
    W2=parameters['W2']; b2=parameters['b2']
    dW1=grads['dW1'];   db1=grads['db1']
    dW2=grads['dW2'];   db2=grads['db2']
    W1 -= dW1*learning_rate;  b1 -= db1*learning_rate
    W2 -= dW2*learning_rate;  b2 -= db2*learning_rate   
    new_parameters={'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return new_parameters

def model(X, y, n_x, n_h, n_y, num_of_iters, learning_rate):
    parameters=init_parameters(n_x, n_h, n_y)
    for i in range(num_of_iters+1):
        a2, cache = forward_prop(X, parameters)
        cost=calculate_cost(a2, y)
        grads=backward_prop(X,y,cache,parameters)
        parameters=update_parameters(parameters, grads, learning_rate)
        if i%3 == 0: print('Cost after iteration# {:d}: {:f}'.format(i, cost))
    return parameters

def predict(X, parameters):
    a2, cache = forward_prop(X, parameters)
    yhat=a2
    yhat=np.squeeze(yhat)
    if yhat >= 0.5: return 1
    else: return 0

np.random.seed(51)
dfi = sns.load_dataset('iris')

# A little EDA to show classification is feasible
sns.lmplot(x='petal_length', y='petal_width', data=dfi, hue='species', fit_reg=False)
plt.show()

# Make it binary, setosa and not setosa
dfi['species']=dfi['species'].map({'setosa':1, 'versicolor':0, 'virginica':0})

X = dfi.drop(['species', 'sepal_length', 'sepal_width'], axis=1)   # 2 features only
y = dfi['species']
print(X.head())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=32)

m=X_train.shape[1]      # Global variable: number of training columns is 2
print('The shape of X_train is {}'.format(X_train.shape))        # (120, 2)

# Get the DFs into arrays and the shapes in proper order for this framework
#  which is based on Medium article 1/28 and ma1.py which uses XOR

X_train_arr=X_train.to_numpy().T
X_test_arr=X_test.to_numpy().T
y_train_arr=y_train.to_numpy().reshape(-1, 120)
y_test_arr=y_test.to_numpy().reshape(-1, 30)

# Hyperparameters
n_x=2                 # No. of neurons in input layer
n_h=8                 # No. of neurons in hidden layer
n_y=1                 # No. of neurons in output layer
num_of_iters=10       # No. of epochs
learning_rate=0.01    # This was critical

trained_parameters=model(X_train_arr, y_train_arr, n_x, n_h, n_y, num_of_iters, learning_rate)

print("[[", end='')
for tup in zip(X_test_arr[0], X_test_arr[1]):
    X_test_temp=np.array([[tup[0]], [tup[1]]])
    y_pred=predict(X_test_temp, trained_parameters)
    print(y_pred, end=' ')
print("]]")

print(y_test_arr)
