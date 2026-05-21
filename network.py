import numpy as np
class Neuron:
	def __init__(self, weights, biases):
		self.weights = weights
		self.biases = biases
class Network():
	def __init__(self, *args: int):
		self.layers = []
		for i in range(1,len(args)):
			weights = np.random.rand(args[i-1], args[i])
			biases = np.random.rand(args[0], args[i])
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

if __name__ == "__main__":
	net = Network(2,3,2)
	print(net)