from random import random, randint
# y = 3x + 2
expected = {
	0:2,
	3:11,
	8:26,
	10:32
}
def cost(w, b):
	error = 0
	for i in expected:
		prediction = w*i+b
		error += (prediction-expected[i])**2
	return error

learn_rate = 1e-3

w = random()*20-10
b = random()*20-10
w_init = w
b_init = b
eps = 1e-3

error = cost(w, b)
error_init = error

for i in range(20000):
	dw = (cost(w+eps, b) - cost(w, b))/eps
	db = (cost(w, b+eps) - cost(w, b))/eps
	w -= dw*learn_rate
	b -= db*learn_rate
	# print(dw*learn_rate, db*learn_rate)
	# print(cost(w, b))

print("Initial error:", error_init, ", New:", error);
print("old w:", w_init, "w: ", w)
print("old b:", b_init, "b: ", b)