# neural-networks
Learning neural networks from scratch, without any libraries or abstractions. Pure mathematics!!

## XOR operation using finite difference gradient descent
```
Network = 2-2-1
Rate: 1e-1
Eps: 1e-3
Iterations: 10000

Randomized initialization:
[
	layer[1]: 
		Weights [
			[0.10397148 0.35314678] 
			[0.72151571 0.22562783] 
		]
		Biases [
			0.6237433355905243 
			0.6047149341374166 
		]
	layer[2]: 
		Weights [
			[0.94829035] 
			[0.1645068] 
		]
		Biases [
			0.17306031088569518 
		]
]

Cost: 1.20619723236888 => 0.0038727474566097814
----------------------------

[
	layer[1]: 
		Weights [
			[6.18137884 4.14800897] 
			[6.16171423 4.14368296] 
		]
		Biases [
			-2.680936728821352 
			-6.365563164648691 
		]
	layer[2]: 
		Weights [
			[8.53527884] 
			[-9.25595301] 
		]
		Biases [
			-3.889945698548989 
		]
]

0 + 0 = [0.03361273]
0 + 1 = [0.97026304]
1 + 0 = [0.97029991]
1 + 1 = [0.0312498]

in 33.476 seconds
```