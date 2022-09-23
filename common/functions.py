import numpy as np
import matplotlib.pylab as plt
from PIL import Image

def step_function(x):
	return np.array(x > 0, dtype = np.int)

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def relu(x):
	return np.maximum(0, x)

def softmax(a):
	c = np.max(a)
	exp_a = np.exp(a - c)
	sum_exp_a = np.sum(exp_a)
	y = exp_a / sum_exp_a
	
	return y

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)

t=[0,0,1,0,0,0,0,0,0,0]
y=[0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.1,0.0,0.0]
print(mean_squared_error(np.array(y), np.array(t)))
