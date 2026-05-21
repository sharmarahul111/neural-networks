import numpy as np
class Neuron:
	def __init__(self, weights, biases):
		self.weights = weights
		self.biases = biases
class Network():
	def __init__(self, *args: int):
		self.inputs = args[0]
		self.outputs = args[-1]
		self.design = args
		self.layers = []
		for i in range(1,len(args)):
			weights = np.random.rand(args[i-1], args[i])
			biases = np.random.rand(args[i])
			self.layers.append(Neuron(weights, biases))
	
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
		return 1./(1+np.exp(-x))

	def forward(self, inp):
		activation = inp
		for layer in self.layers:
			activation = activation @ layer.weights + layer.biases
			activation = self.sigmoid(activation)
		return activation

	def cost(self,inputs, expected):
		error = np.zeros(np.shape(expected[0]))
		for inp, ex in zip(inputs, expected):
			error += (self.forward(inp)-ex)**2
		return np.sum(error)