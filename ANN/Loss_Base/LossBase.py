import numpy as np

class Loss:
    
    def calculate(self, yPred, yTruth):
        
        estimateLoss_vector = self.forward(yPred, yTruth)
        
        return np.mean(estimateLoss_vector)
