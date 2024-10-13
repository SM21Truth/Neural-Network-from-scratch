import numpy as np

class SoftMax():
    
    def forward(self, feedIn):
        
        exp_values = np.exp( feedIn -np.max(feedIn) )
        probabilities = exp_values / np.sum(exp_values, axis = 0, keepdims=True)
        
        self.feedOut = probabilities
    
    def backward(self, dvalues):
        
        self.dinputs = np.empty_like(dvalues)
        
        for i, single_feedOut, single_dvalues in enumerate(zip(self.feedOut.T, dvalues.T)):
            
            single_feedOut = single_feedOut.reshape(-1,1)
            jacobMatrix = np.diagflat(single_feedOut) - np.dot(single_feedOut, single_feedOut.T)
            
            self.dinputs[:,i] = np.dot(jacobMatrix, single_dvalues.reshape(-1,1))
