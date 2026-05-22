import numpy as np
class Layer:
	def __init__(self, weights=None, biases=None):
		self.weights = weights
		self.biases = biases

class Network():
	def __init__(self, *args: int):
		self.design = args
		self.layers = []
		for i in range(1,len(args)):
			weights = np.random.randn(args[i-1], args[i])*.5
			# biases = np.random.rand(args[i])
			biases = np.zeros(args[i])
			self.layers.append(Layer(weights, biases))
	
	def __str__(self):
		s = "\n["
		i=1
		for layer in self.layers:
			s+= f"\n\tlayer[{i}]: "
			# weights
			s+= "\n\t\tWeights ["
			for l in layer.weights:
				s+= "\n\t\t\t"
				s+= str(l)
				s+= " "
			s+= "\n\t\t]"

			# biases
			s+= "\n\t\tBiases ["
			for l in layer.biases:
				s+= "\n\t\t\t"
				s+= str(l)
				s+= " "
			s+= "\n\t\t]"

			i+=1
		s+= "\n]\n"
		return s

	def sigmoid(self, x):
		# return x
		return 1./(1+np.exp(-x))

	def tanh(self, x):
		return np.tanh(x)

	def forward(self, inp):
		activation = inp
		for i in range(len(self.layers)-1):
			activation = activation @ self.layers[i].weights + self.layers[i].biases
			activation = self.tanh(activation)
		activation = activation @ self.layers[-1].weights + self.layers[-1].biases
		activation = self.sigmoid(activation)
		return activation

	def cost(self,inputs, expected):
		error = 0
		for inp, ex in zip(inputs, expected):
			
			error += np.mean((self.forward(inp)-ex)**2)
		return error/len(inputs)
	
	def finite_diff(self, inp, expected, eps):
		gradient = []
		for layer in self.layers:
			# weights
			g = Layer()
			g.weights = np.zeros(np.shape(layer.weights))
			for i in range(g.weights.shape[0]):
				for j in range(g.weights.shape[1]):
					cost_original = self.cost(inp, expected)
					saved = layer.weights[i][j]
					layer.weights[i][j] += eps
					cost_new = self.cost(inp, expected)
					layer.weights[i][j] = saved
					g.weights[i][j] = (cost_new-cost_original)/eps

			# biases
			g.biases = np.zeros(np.shape(layer.biases))
			for i in range(g.biases.shape[0]):
				cost_original = self.cost(inp, expected)
				saved = layer.biases[i]
				layer.biases[i] += eps
				cost_new = self.cost(inp, expected)
				layer.biases[i] = saved
				g.biases[i] = (cost_new-cost_original)/eps

			gradient.append(g)
		return gradient

	def descent(self, gradient, rate=1e-3):
		for layer, g in zip(self.layers, gradient):
			layer.weights -= g.weights*rate
			layer.biases -= g.biases*rate