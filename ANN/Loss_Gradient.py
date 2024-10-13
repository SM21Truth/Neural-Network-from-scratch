import numpy as np

class LossGradient:
    
    def backward(self, yPred, yTrue):
        
        if len(yTrue)>1:
            yTrue = np.argmax(yTrue, axis = 0, keepdims=True)
        
        numberOfInstances = yTrue.shape[1]
        self.dinputs = yPred.copy()
        self.dinputs[yTrue, range(numberOfInstances)] -= 1
        self.dinputs = self.dinputs/numberOfInstances
