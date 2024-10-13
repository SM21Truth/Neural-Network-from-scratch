import numpy as np
from ANN.Loss_Base import LossBase

class CrossEntropy(LossBase.Loss):
    
    def forward(self, yPred, yTruth):
        
        yPred_Clipped            = np.clip(yPred, 1e-7, 1-1e-7)
        
        if len(yTruth) == 1:
            estimate = yPred_Clipped[yTruth,range(yTruth.shape[1])]
        else:
            estimate = np.sum(yPred_Clipped*yTruth, axis=0, keepdims=True)
            
        negative_log_likelihoods = -np.log(estimate)
        
        return negative_log_likelihoods
    
    def backward(self, yPred, ytrue):
        
        if len(ytrue)==1:
            ytrue = np.eye(yPred.shape[0])[ytrue[0,:]]
        
        self.dinputs = ytrue/yPred/yPred.shape[1]
