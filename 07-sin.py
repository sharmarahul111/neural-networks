# NOTE: uses tahn in internal network and linear output
import numpy as np
from network import Network
import matplotlib.pyplot as plt

inp = np.random.rand(40,1)*np.pi*2
expected = np.sin(inp)
EPS = 1e-2
RATE = 4e-2
network = Network(1,4,2,1)
print(network)

error = network.cost(inp, expected)
err = []
for i in range(1500):
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

print(f"Training cost: {error} => {network.cost(inp, expected)}")
# plt.plot(inp, expected)
plt.plot(err)
plt.show()
# print(error)
print("----------------------------")
# exit()
# print(network)
# exit()
# testing zone
plt.clf()
x = np.sort(np.random.rand(200) * 2*np.pi)
y_true = np.sin(x)
y_pred = []
for v in x:
    pred = network.forward(np.array([v]))
    y_pred.append(pred.item())

y_pred = np.array(y_pred)

plt.plot(x, y_true, label="sin(x)")
plt.plot(x, y_pred, label="network")

print("Testing cost: ", network.cost(x.reshape(-1,1), y_true))
plt.legend()
plt.show()