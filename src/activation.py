import math

class Linear:
    def forward(self, x):
        return x
    
    def derivative(self, x):
        return 1
    
class ReLU:
    def forward(self, x):
        return max(0, x)
    
    def derivative(self, x):
        return 1 if x > 0 else 0
    
class Sigmoid:
    def forward(self, x):
        return 1 / (1 + math.exp(-x))
    
    def derivative(self, x):
        sigmoid_x = self.forward(x)
        return sigmoid_x * (1 - sigmoid_x)

class Tanh:
    def forward(self, x):
        return math.tanh(x)
    
    def derivative(self, x):
        tanh_x = self.forward(x)
        return 1 - tanh_x ** 2
    
class Softmax:
    def forward(self, x):
        exp_x = [math.exp(i) for i in x]
        sum_exp_x = sum(exp_x)
        return [i / sum_exp_x for i in exp_x]
    
    def derivative(self, x):
        softmax_x = self.forward(x)
        return [s * (1 - s) for s in softmax_x]