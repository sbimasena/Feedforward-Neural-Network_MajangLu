import math
import numpy as np

class MSE:
    def forward(self, predicted, actual):
        return np.mean((actual - predicted) ** 2)
    def derivative(self, predicted, actual):
        return 2 * (predicted - actual) / len(predicted)
    
class BinaryCrossEntropy:
    def forward(self, predicted, actual):
        eps = 1e-15
        predicted = np.clip(predicted, eps, 1 - eps)
        loss = actual * np.log(predicted) + (1 - actual) * np.log(1 - predicted)
        return -np.mean(loss)

    def derivative(self, predicted, actual):
        eps = 1e-15
        predicted = np.clip(predicted, eps, 1 - eps)
        N = len(predicted)
        return (predicted - actual) / (predicted * (1 - predicted) * N)
    
class CategoricalCrossEntropy:
    def forward(self, predicted, actual):
        eps = 1e-15
        predicted = np.clip(predicted, eps, 1 - eps)
        loss = actual * np.log(predicted)
        return -np.mean(np.sum(loss, axis=1))

    def derivative(self, predicted, actual):
        eps = 1e-15
        predicted = np.clip(predicted, eps, 1 - eps)
        N = predicted.shape[0]
        return -actual / (predicted * N)