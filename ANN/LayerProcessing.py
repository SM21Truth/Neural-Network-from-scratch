import numpy as np

class NeuralLayer:
    
    def __init__(self, numberOfNeurons, numberOfFeatures):
        
        # self.weights = np.random.randn(numberOfNeurons, numberOfFeatures)
        self.weights  = np.random.randn(numberOfFeatures, numberOfNeurons ).T
        self.biases   = np.zeros((numberOfNeurons,1))
        
    def forward(self, feedIn):
        
        self.feedOut  = np.matmul(self.weights, feedIn) + self.biases
        self.inputs   = feedIn
        
    def backward(self, dvalues):
        # dvalues: gradient (loss/alpha)
        self.dinputs  = np.matmul(self.weights.T, dvalues)
        self.dweights = np.matmul(dvalues, self.inputs.T)
        self.dbiases  = np.sum(dvalues, axis = 1, keepdims=True) # axis = 0 column wise/ 1 row wise
