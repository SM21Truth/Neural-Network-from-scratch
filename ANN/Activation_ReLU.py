import numpy as np

class ReLU:
    
    def forward(self, feedIn):
        
        self.feedOut = np.maximum(0, feedIn)
        self.inputs  = feedIn
    
    def backward(self, dvalues):
        
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs<=0] =0

