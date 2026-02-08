# activations.py, B. Harmon 11/1/2021
# Activation functions illustrated
import numpy as np
import matplotlib.pyplot as plt

sigmoid =  lambda x: 1 / (1 + np.exp(-x))  #  0 to 1
activ =    lambda x: np.tanh(x/2)          # -1 to 1

relu = lambda x: np.maximum(0, x)
leaky_relu = np.vectorize(lambda x: x if x>=0 else 0.1*x)

x = np.linspace(-10, 10, 201)

fig,ax=plt.subplots(2,2,figsize=(13,6), sharex=True)
ax[0,0].plot(x, sigmoid(x))
ax[0,1].plot(x, activ(x))
ax[0,0].set_title('Sigmoid(x)')
ax[0,1].set_title('Tanh(x/2)')
ax[0,0].grid()
ax[0,1].grid()
ax[1,0].plot(x, relu(x)) 
ax[1,1].plot(x, leaky_relu(x))
ax[1,0].set_title('Relu(x)')
ax[1,1].set_title('Leaky Relu(x)')
ax[1,0].grid()
ax[1,1].grid()
plt.tight_layout()
plt.show()
