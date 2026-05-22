import numpy as np
from network import Network
train_or = np.array([
	[0,0,0],
	[0,1,1],
	[1,0,1],
	[1,1,1]
])
train_xor = np.array([
	[0,0,0],
	[0,1,1],
	[1,0,1],
	[1,1,0]
])
inp = train_or[0: , :2]
expected = train_or[0: , 2:]
EPS = 1e-3
RATE = 1e-2
network = Network(2,1)
print(network)

error = network.cost(inp, expected)
for i in range(20000):
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

print(f"Cost: {error} => {network.cost(inp, expected)}")

# print(error)

print("----------------------------")
print(network)
for i in range(2):
	for j in range(2):
		result = network.forward(np.array([i,j]))
		print(f"{i} + {j} = {result}")
		