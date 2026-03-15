import numpy as np

class Linear:
    def forward(self, x):
        return x
    
    def derivative(self, x):
        return 1
    
class ReLU:
    def forward(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return (x > 0).astype(float)
    
class Sigmoid:
    def forward(self, x):
        return 1.0 / (1.0 + np.exp(-x))
    
    def derivative(self, x):
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1.0 - sigmoid_x)

class Tanh:
    def forward(self, x):
        return np.tanh(x)
    
    def derivative(self, x):
        tanh_x = self.forward(x)
        return 1 - tanh_x ** 2
    
class Softmax:
    def forward(self, x):
        shifted_x = x - np.max(x, axis=-1, keepdims=True)
        exp_x = np.exp(shifted_x)
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def derivative(self, x):
        softmax_x = self.forward(x)
        return softmax_x * (1.0 - softmax_x)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)
    
    def derivative(self, x):
        return np.where(x > 0, 1.0, self.alpha)
    

class ELU:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * (np.exp(x) - 1))
    
    def derivative(self, x):
        return np.where(x > 0, 1.0, self.forward(x) + self.alpha)