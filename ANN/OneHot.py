import numpy as np

class OneHot:
    
    def oneHot(self, vector):
        
        vectorShape = vector.shape
        numberOfClass = np.max(vector) + 1
        
        if vectorShape[0]==1:
            output = np.zeros((numberOfClass, vectorShape[1]))
            output[vector, range(vectorShape[1])] = 1 
        else:
            output = np.zeros((vectorShape[0], numberOfClass))
            output[range(vectorShape[0]), vector] = 1 
        
        return output