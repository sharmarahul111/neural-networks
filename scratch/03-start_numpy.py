import numpy as np
import matplotlib.pyplot as plt

# training data and extracting input and result
training_data = np.array([
	[5,10],
	[1,2],
	[2,4],
	[3,6]
])
x = training_data[0: , 0:1]
expected = training_data[0: , 1:]

def err_percent(a,b):
	return abs(a-b)/b*100

def cost(w, b):
	error = (forward(x, w, b) -expected)**2
	return np.sum(error)

def forward(x, w, b):
	# y = wx + b
	return x@w + b
# simulating one input one output

# setting learning rate and eps
eps = 1e-3
rate = 1e-2
# making the model with weights and biases
w = np.random.rand(1,1)*20-5
b = np.random.rand(1,1)*20-5
y = forward(x, w, b)

# learn/train
print("Pre-train error: ", cost(w,b))
dw_l = []
for i in range(1000):
	dw = (cost(w+eps, b)-cost(w, b))/eps
	db = (cost(w, b+eps)-cost(w, b))/eps
	w -= dw*rate
	b -= db*rate
print("Post-train error: ", cost(w,b))
for i in range(1, 10):
	print(f"{i} => {(i*w+b)[0][0]}")

print(w,b)