import numpy as np
class SGD:
    
    def __init__(self, learningRate = .1, decay = 0, momentum = 0):
        self.learningRate         = learningRate
        self.decay                = decay
        self.currentLearningRate  = learningRate
        self.iterations           = 0
        self.momentum             = momentum
    
    def pre_updateParameters(self):
        if self.decay:
            self.currentLearningRate = self.learningRate*(1/(1+(self.decay * self.iterations)))
    
    def updateParameters(self, obj_layer):
        
        if self.momentum:
            if not hasattr(obj_layer, 'weight_momentums'):
                obj_layer.weight_momentums = np.zeros_like(obj_layer.weights)
                obj_layer.bias_momentums   = np.zeros_like(obj_layer.biases)
            weight_updates = self.momentum * obj_layer.weight_momentums \
                           - self.currentLearningRate * obj_layer.dweights
            obj_layer.weight_momentums = weight_updates
            
            bias_updates = self.momentum * obj_layer.bias_momentums \
                         - self.currentLearningRate * obj_layer.dbiases
            obj_layer.bias_momentums = bias_updates
        else:
            weight_updates = - self.currentLearningRate * obj_layer.dweights
            bias_updates   = - self.currentLearningRate * obj_layer.dbiases
            
        obj_layer.weights += weight_updates
        obj_layer.biases  += bias_updates
    
    def post_updateParameters(self):
        self.iterations += 1

