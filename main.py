import numpy as np
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

np.random.seed(40)
[x, y] = spiral_data(samples = 200, classes = 3)

if type(x) != np.ndarray:
    x = np.array(x)
    
if type(y) != np.ndarray:
    y = np.array(y)

################################### Plot ###################################
querry1 = input("Wanna see plot ?\n")
if querry1 == 'yes' :
    # index_Class1 = np.argwhere(y == 0)
    # index_Class2 = np.argwhere(y == 1)
    # index_Class3 = np.argwhere(y == 2)
    
    # plt.scatter(x[index_Class1, 0], x[index_Class1, 1], c = "blue")
    # plt.scatter(x[index_Class2, 0], x[index_Class2, 1], c = "red")
    # plt.scatter(x[index_Class3, 0], x[index_Class3, 1], c = "magenta")
    
    colors = ["blue","red","magenta","green","yellow","orange","cyan"]
    for i in range(np.max(y)+1):
        plt.scatter(x[np.argwhere(y == i), 0], x[np.argwhere(y == i), 1], c = colors[i])
        
    print(f"\nx : {type(x)}\ny : {type(y)}")
################################### Plot ###################################


################################# Transpose ################################
feedIn = x.T    # assumeing column in x represent features

if len( y.shape ) == 1 :
    
    y_true = y.reshape(1,-1)
    
    # from ANN import OneHot as oh
    # obj_OneHot = oh.OneHot()
    # y_true_OneHot = obj_OneHot.oneHot(y).T
    # y_true_OneHotMatrix = np.eye(np.max(y)+1)[y].T
    
else:
    if y.shape[0] ==1:
       y_true = y
    else:
       y_true = y.T
################################# Transpose ################################

############################## Neural Network ##############################
neurons_layer1 = 64
neurons_layer2 = np.max(y_true)+1

N              = 10000
Monitor        = np.zeros((N, 3))

from ANN import LayerProcessing    as lp
from ANN import Activation_ReLU    as ru
from ANN import Activation_Softmax as rs
from ANN import Cross_Entropy      as ce
from ANN import Loss_Gradient      as lg
from ANN import Optimizer_SGD      as op


obj_layer1       = lp.NeuralLayer(neurons_layer1, feedIn.shape[0])
obj_activation1  = ru.ReLU()
obj_layer2       = lp.NeuralLayer(neurons_layer2, neurons_layer1)
obj_activation2  = rs.SoftMax()
obj_loss         = ce.CrossEntropy()
obj_lossgradient = lg.LossGradient()
obj_optimizer    = op.SGD(learningRate=.9, decay = 0.001, momentum = 0.6)

for epoch in range(N):
    obj_layer1.forward(feedIn)
    obj_activation1.forward(obj_layer1.feedOut)
    obj_layer2.forward(obj_activation1.feedOut)
    obj_activation2.forward(obj_layer2.feedOut)
    
    
    predictions  = np.argmax( obj_activation2.feedOut, axis = 0, keepdims=True)
    
    if len(y_true) > 1:
        truth_SV = np.argmax(y_true, axis = 0, keepdims=True)
    else:
        truth_SV = y_true
    accuracy     = np.mean( predictions == truth_SV)
    
    loss         = obj_loss.calculate(obj_activation2.feedOut, y_true)
    
    obj_lossgradient.backward(obj_activation2.feedOut, y_true)
    obj_layer2.backward(obj_lossgradient.dinputs)
    obj_activation1.backward(obj_layer2.dinputs)
    obj_layer1.backward(obj_activation1.dinputs)
    
    obj_optimizer.pre_updateParameters()
    obj_optimizer.updateParameters(obj_layer1)
    obj_optimizer.updateParameters(obj_layer2)
    obj_optimizer.post_updateParameters()
    
    Monitor[epoch, 0] = accuracy*100
    Monitor[epoch, 1] = loss
    Monitor[epoch, 2] = obj_optimizer.currentLearningRate
    
    if not epoch % 100:
        print(f'epoch : {epoch},  accuracy : {accuracy:.3f},  loss : {loss:.3f}')
        
print(f'Accuracy is {accuracy*100} %')

fig, ax = plt.subplots(3, 1, sharex = True)
ax[0].plot(np.arange(N), Monitor[:,0])
ax[0].set_ylabel('accuracy [%]')

ax[1].plot(np.arange(N), Monitor[:,1])
ax[1].set_ylabel('loss')

ax[2].plot(np.arange(N), Monitor[:,2])
ax[2].set_ylabel(r'learningRate $\alpha$')
ax[2].set_xlabel('epoch')

plt.xscale('log',base=10)
############################## Neural Network ##############################

# import Write
# obj_write       = Write.WriteFile()
# obj_write.write('l1_out', obj_layer1.feedOut)
# obj_write.write('act1_out', obj_activation1.feedOut)
# obj_write.write('l2_out', obj_layer2.feedOut)
# obj_write.write('act2_out', obj_activation2.feedOut)
# obj_write.write('predictions', predictions)
# obj_write.write('accuracy', accuracy)
# obj_write.write('loss', loss)
# obj_write.write('l1_dinputs', obj_layer1.dinputs)
# obj_write.write('l1_dweights', obj_layer1.dweights)
# obj_write.write('l1_dbiases', obj_layer1.dbiases)
# 0.5933333333333334