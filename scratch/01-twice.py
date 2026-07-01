"""
Gradiend Descent:
minimizing a loss function by repeatedly updating parameters in the direction that reduces error.
dx = ( f(x+h) - f(x) ) / h
"""

from random import random
# mapping y = 2x
expected = {
	0:0,
	1:2,
	2:4,
	3:6
}
# calcualtes summation of mean square error
def cost(w, b):
	error = 0
	for i in expected:
		# y = wx + b (b=0)
		prediction = w*i+b
		error += (prediction-expected[i])**2
	return error

learn_rate = 1e-2

w = random()*20-10
b = 0 #random()*20
w_init = w
eps = 1e-2

error = cost(w, b)
error_init = error

for i in range(2000):
	error = cost(w, b)
	dx = (cost(w+eps, b) - error)/eps
	w -= dx*learn_rate
	# print(dx*learn_rate)

print("Initial error:", error_init, ", New error:", error, ", w:", w);
print("Initial w:", w_init, "w: ", w)