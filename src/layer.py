import numpy as np
import random

class DenseLayer:
    def __init__(self, input_size, output_size, activation, init_method, init_params=None):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        if init_params is None:
            init_params = {}

        self.weights = self.init_weights(init_method, init_params)  
        self.bias = np.zeros((1, output_size))

        self.input=None
        self.output=None
        self.z=None
        self.grad_weights=None
        self.grad_bias=None 

    def init_weights(self, init_method, init_params):
        if init_method == "zero":
            return np.zeros((self.input_size, self.output_size))
        elif init_method == "uniform":
            low = init_params.get("low", -1.0)
            high = init_params.get("high", 1.0)
            seed = init_params.get("seed", None)

            if seed is not None:
                np.random.seed(seed)

            return np.random.uniform(low, high, (self.input_size, self.output_size))
        elif init_method == "normal":
            mean = init_params.get("mean", 0.0)
            var = init_params.get("var", 1.0)
            seed = init_params.get("seed", None)

            if seed is not None:
                np.random.seed(seed)

            return np.random.normal(mean, np.sqrt(var), (self.input_size, self.output_size))
        else:
            raise ValueError("Invalid init_method") 

    def forward(self, x):
        self.input = x
        self.z = np.dot(x, self.weights) + self.bias
        self.output = self.activation.forward(self.z)
        return self.output

    def backward(self, grad_output):
        grad_activation = grad_output * self.activation.derivative(self.z)
        self.grad_weights = np.dot(self.input.T, grad_activation)
        self.grad_bias = np.sum(grad_activation, axis=0, keepdims=True)
        grad_input = np.dot(grad_activation, self.weights.T)
        return grad_input

    def update(self, learning_rate):
        self.weights -= learning_rate * self.grad_weights
        self.bias -= learning_rate * self.grad_bias
        
        