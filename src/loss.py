import math

class MSE:
    def forward(self, predicted, actual):
        vector = [(actual[i] - predicted[i])**2 for i in range(len(predicted))]
        return sum(vector) / len(vector)
    def derivative(self, predicted, actual):
        return 2 * (predicted - actual) / len(predicted)
    
class BinaryCrossEntropy:
    def forward(self, predicted, actual):
        vector = [(actual[i] * math.log(predicted[i]) + (1 - actual[i]) * math.log(1 - predicted[i])) for i in range(len(predicted))]
        return -sum(vector) / len(vector)
    def derivative(self, predicted, actual):
        return predicted - actual
    
class CategoricalCrossEntropy:
    def forward(self, predicted, actual):
        vector = [actual[i] * math.log(predicted[i]) for i in range(len(predicted))]
        return -sum(vector) / len(vector)
    def derivative(self, predicted, actual):
        return predicted - actual