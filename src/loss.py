import math

def mse(predicted, actual):
    vector = [(actual[i] - predicted[i])**2 for i in range(len(predicted))]
    return sum(vector) / len(vector)

def binary_cross_entropy(predicted, actual):
    vector = [(actual[i] * math.log(predicted[i]) + (1 - actual[i]) * math.log(1 - predicted[i])) for i in range(len(predicted))]
    return -sum(vector) / len(vector)

def categorical_cross_entropy(predicted, actual):
    vector = [actual[i] * math.log(predicted[i]) for i in range(len(predicted))]
    return -sum(vector) / len(vector)