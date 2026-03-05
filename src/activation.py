import math

def linear(x):
    return x

def relu(x):
    return max(0, x)

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def tanh(x):
    return math.exp(x) - math.exp(-x) / (math.exp(x) + math.exp(-x))

def softmax(vector):
    exp_vector = [math.exp(x) for x in vector]
    sum = sum(exp_vector)
    return [i/sum for i in exp_vector]
