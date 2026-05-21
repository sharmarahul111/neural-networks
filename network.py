import numpy as np
class Network():
	def __init__(self, *args: int):
		self.layers = [np.random.rand(1, args[0])]
		for i in range(1,len(args)):
			self.layers.append(np.random.rand(args[i-1], args[i]))
	
	def __str__(self):
		s = "\n["
		i=1
		for layer in self.layers:
			s+= f"\n\tlayer[{i}]: "
			# s+= str(layer)
			s+= "["
			for l in layer:
				s+= "\n\t\t"
				s+= str(l)
				s+= " "
			s+= "\n\t]"
			i+=1
		s+= "\n]\n"
		return s


net = Network(1,3,2)
print(net)