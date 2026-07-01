import numpy as np
from network import Network
import matplotlib.pyplot as plt 

inp = np.random.rand(80,2)*2-1
radius = .75
expected = []
for [x,y] in inp:
	if x**2+y**2<=radius**2:
		expected.append(1)
	else:
		expected.append(0)
expected = np.array(expected).reshape(-1,1)
# print(np.sum(expected))
# exit()
# print(inp)
# print(expected)
EPS = 1e-3
RATE = 1e-0
network = Network(2,4,1)
print(network)

error = network.cost(inp, expected)
err = []
for i in range(1300):
	gradient = network.finite_diff(inp, expected, eps=EPS)
	# printing the gradient
	# disp_gradient = Network(*network.design)
	# disp_gradient.layers = gradient
	# print("----------------------------")
	# print(f"Gradient {i}:")
	# print(disp_gradient)
	# print("Cost:", network.cost(inp, expected))
	# end print
	network.descent(gradient, RATE)
	err.append(network.cost(inp, expected))

print(f"Cost: {error} => {network.cost(inp, expected)}")
# plt.plot(inp, expected)
plt.plot(err)
plt.show()
# print(error)
print("----------------------------")
print(network)
exit()
for i in np.linspace(0,.9, 10):
	result = network.forward(np.array([i]))
	print(f"{i}^2 = {result} ({i**2})")
		