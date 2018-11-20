
# coding: utf-8

# In[17]:


# Python imports
import numpy as np # Matrix and vector computation package
import matplotlib.pyplot as plt  # Plotting library
# Allow matplotlib to plot inside this notebook
# get_ipython().magic(u'matplotlib inline')
# get_ipython().magic(u"config InlineBackend.figure_format = 'svg'")
# Set the seed of the numpy random number generator so that the tutorial is reproducable
np.random.seed(seed=1)
from sklearn import datasets, cross_validation, metrics # data and evaluation utils
from matplotlib.colors import colorConverter, ListedColormap # some plotting functions
import itertools
import collections


# In[18]:


def add_one(data):
    ones = np.ones(tuple(data.shape[:-1]) + (1,))
    return np.concatenate([ones, data], axis=-1)


# In[19]:


digits = datasets.load_digits()

Data = digits.data
Target = np.zeros((digits.target.shape[0],10))
Target[np.arange(len(Target)), digits.target] += 1

X_train, X_test, T_train, T_test = cross_validation.train_test_split(
    Data, Target, test_size=0.4)
X_validation, X_test, T_validation, T_test = cross_validation.train_test_split(
    X_test, T_test, test_size=0.5)
# print(X_train.shape)


# In[20]:


# fig = plt.figure(figsize=(10, 1), dpi=100)
# for i in range(10):
#     ax = fig.add_subplot(1,10,i+1)
#     ax.matshow(digits.images[i], cmap='binary') 
#     ax.axis('off')
# # plt.show()


# In[21]:


# Define the non-linear functions used
def logistic(z): 
    return 1 / (1 + np.exp(-z))
    
def softmax(z): 
    return np.exp(z) / np.sum(np.exp(z), axis=-1, keepdims=True)

def threshold(z):
    
    #print("z.shape : " + str(z.shape))
    #print("z.type : " + str(type(z)))
    
    z[z < 0] = 0
    return np.sign(z)
    
def relu(z):
    return np.maximum(0, z)

def leaky_relu(z):
    return np.maximum(0.1*z, z)
    


# In[23]:


# Define the layers used in this model
class Layer(object):
    """Base class for the different layers.
    Defines base methods and documentation of methods."""
    
    def get_params(self):
        """Return an iterator over the parameters (if any).
        The iterator has the same order as get_params_grad.
        The elements returned by the iterator are editable in-place."""
        return []
    
    def get_grad(self, loss):
        """Return a list of gradients over the parameters.
        The list has the same order as the get_params_iter iterator.
        X is the input.
        output_grad is the gradient at the output of this layer.
        
        """
        return []
    
    def get_output(self, X):
        """Perform the forward step linear transformation.
        X is the input."""
        pass
    
    def update_param(self, loss, learning_rate):
        pass
    


# In[24]:


class LinearLayer(Layer):
    """The linear layer performs a linear transformation to its input."""
    
    def __init__(self, n_in, n_out, sigm, n_each):
        """
        Initialize hidden layer parameters.
        n_in is the number of input variables.
        n_out is the number of output variables.
        sigm is the std-deviation of noise.
        """
        self.sigm = sigm
        self.W = np.random.randn(n_in+1, n_out) * 0.1
        
    def get_params(self):
        """Return an iterator over the parameters."""
        return self.W
    
    def get_output(self, X):
        """Perform the forward step linear transformation."""
        self.noise = np.random.randn(*(X.shape[:-1] + (self.W.shape[-1], ))) * self.sigm
        self.input = add_one(X)
        return self.input.dot(self.W) + self.noise 
        
    def get_grad(self, loss):
        """Return a list of gradients over the parameters."""
#         return  (self.input[:, :, np.newaxis].dot(self.noise.reshape(1, self.noise.shape[0])) * loss / self.sigm**2).mean(axis=0)
        return  (np.einsum('abc,abd->abcd',(self.input * loss[:, :, np.newaxis]), self.noise)/(self.sigm**2)).mean(axis=(0, 1))
#         print((np.einsum('abc,abd->abcd',(self.input * loss[:, :, np.newaxis]), self.noise)/(self.sigm**2)).mean(axis=(0, 1))[0])
    
    
    def update_param(self, loss, learning_rate):
        self.W -= learning_rate * self.get_grad(loss)
    


# In[25]:


class LogisticLayer(Layer):
    """The logistic layer applies the logistic function to its inputs."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        re = logistic(X)
        #re = threshold(X)
        return re


# In[26]:


class SoftmaxOutputLayer(Layer):
    """The softmax output layer computes the classification propabilities at the output."""
    
    def get_output(self, X):
        """Perform the forward step transformation."""
        return softmax(X)
    
    def get_cost(self, Y, T, n_each, train=True):
        """Return the cost at the output of this output layer."""
        if train:
            T  = np.stack([T] * n_each).transpose([1, 0, 2])
            return -np.multiply(T, np.log(Y)).sum(axis=-1)
        else:
            return - np.multiply(T, np.log(Y)).sum() / Y.shape[0]


# In[27]:


# Define the forward propagation step as a method.
def forward_step(input_samples, layers, n_each, train=True):
    """
    Compute and return the forward activation of each layer in layers.
    Input:
        input_samples: A matrix of input samples (each row is an input vector)
        layers: A list of Layers
    Output:
        A list of activations where the activation at each index i+1 corresponds to
        the activation of layer i in layers. activations[0] contains the input samples.  
    """
#     activations = [input_samples] # List of layer activations
    # Compute the forward activations for each layer starting from the first
#     X = input_samples
    if train: 
        X = np.stack([input_samples] * n_each).transpose([1, 0, 2])
    else:
        X = input_samples
    for layer in layers:
        if train:
            pass
        else:
            layer.noise = 0 
        X = layer.get_output(X)
#         Y = layer.get_output(X)  # Get the output of the current layer
#         activations.append(Y)  # Store the output for future processing
#         X = activations[-1]  # Set the current input as the activations of the previous layer
#     return activations  # Return the activations of each layer
    return X


# In[28]:


# Define a method to update the parameters
def update_params(layers, loss, learning_rate):
    """
    Function to update the parameters of the given layers with the given gradients
    by gradient descent with the given learning rate.
    """
    for layer in layers:
        layer.update_param(loss, learning_rate)


# In[29]:


# Create the minibatches
batch_size = 25  # Approximately 25 samples per batch
nb_of_batches = X_train.shape[0] / batch_size  # Number of batches
# Create batches (X,Y) from the training set
X_batches = np.array_split(X_train, nb_of_batches, axis=0)  # X samples
T_batches = np.array_split(T_train, nb_of_batches, axis=0)  # Y targets


# In[30]:


sigm = [5,0.1,0.02,0.01] 
n_each = 10000
# Define a sample model to be trained on the data
hidden_neurons_1 = 20  # Number of neurons in the first hidden-layer
hidden_neurons_2 = 20  # Number of neurons in the second hidden-layer
hidden_neurons_3 = 20  # Number of neurons in the second hidden-layer
# Create the model
layers = [] # Define a list of layers
# Add first hidden layer
layers.append(LinearLayer(X_train.shape[1], hidden_neurons_1, sigm[0], n_each))
layers.append(LogisticLayer())
# Add second hidden layer
layers.append(LinearLayer(hidden_neurons_1, hidden_neurons_2, sigm[1], n_each))
layers.append(LogisticLayer())
# Add third hidden layer
layers.append(LinearLayer(hidden_neurons_2, hidden_neurons_3, sigm[2], n_each))
layers.append(LogisticLayer())
# Add output layer
layers.append(LinearLayer(hidden_neurons_3, T_train.shape[1], sigm[3], n_each))
layers.append(SoftmaxOutputLayer())


# In[31]:


# sigm=[2,0.1,0.02,0.01] æ¿€æ´»å‡½æ•?sigmoid,ä¸‰å±‚-> çœ‹acc


# In[32]:


# Perform backpropagation
# initalize some lists to store the cost for future analysis        
minibatch_costs = []
training_costs = []
validation_costs = []

max_nb_of_iterations = 300  # Train for a maximum of 300 iterations
min_nb_of_iterations = 100 
learning_rate = 0.1  # Gradient descent learning rate

# ³ÌÐò½éÉÜ
print("introduction : three hidden layers,sigmoid,sigm=" + str(sigm) + ",epoch=" + str(max_nb_of_iterations) + ",min_epoch=" + str(min_nb_of_iterations))

# Train for the maximum number of iterations
for iteration in range(max_nb_of_iterations):
    n_batch = 0
    for X, T in zip(X_batches, T_batches):  # For each minibatch sub-iteration
        n_batch += 1
        Y = forward_step(X, layers, n_each) # Get the activations
        minibatch_cost = layers[-1].get_cost(Y, T, n_each)  # Get cost
        minibatch_costs.append(minibatch_cost.mean())
        update_params(layers, minibatch_cost, learning_rate)  # Update the parameters
        if n_batch%10 == 1:
            #print('batch: ', n_batch, '/', nb_of_batches, ' loss: ' , minibatch_cost.mean())
            print "minibatch_cost : " + str(minibatch_cost.mean())
        
    # Get full training cost for future analysis (plots)
    activation = forward_step(X_train, layers, n_each, False)
    train_cost = layers[-1].get_cost(activation, T_train, n_each, False)
    training_costs.append(train_cost)
    # Get full validation cost
    activation = forward_step(X_validation, layers, n_each, False)
    validation_cost = layers[-1].get_cost(activation, T_validation, n_each, False)
    validation_costs.append(validation_cost)
    
    nb_of_iterations = iteration + 1  # The number of iterations that have been executed
    #print('epoch: ' , nb_of_iterations , '/' , max_nb_of_iterations, 'train_loss: ', train_cost, ' val_loss: ' , validation_cost)
    print("epoch : " + str(nb_of_iterations))
    
    if (iteration + 1) >= min_nb_of_iterations:
        if len(validation_costs) > 5:
            if validation_costs[-1] >= validation_costs[-2] >= validation_costs[-3] >= validation_costs[-4] >= validation_costs[-5]:
                break


# In[ ]:

print("introduction : three hidden layers,sigmoid,sigm=" + str(sigm))
# Get results of test data
y_true = np.argmax(T_validation, axis=1)  
activation = forward_step(X_validation, layers, n_each, False)  
y_pred = np.argmax(activation, axis=1)  
val_accuracy = metrics.accuracy_score(y_true, y_pred)  
print('The accuracy on the validation set is {:.2f}'.format(val_accuracy))


# In[ ]:


# Get results of test data
y_true = np.argmax(T_test, axis=1)  # Get the target outputs
activation = forward_step(X_test, layers, n_each, False)  # Get activation of test samples
y_pred = np.argmax(activation, axis=1)  # Get the predictions made by the network
test_accuracy = metrics.accuracy_score(y_true, y_pred)  # Test set accuracy
print('The accuracy on the test set is {:.2f}'.format(test_accuracy))


# In[ ]:


# Plot the minbatch, full training set, and validation costs
# minibatch_x_inds = np.linspace(0, nb_of_batches*nb_of_iterations, num = nb_of_batches*nb_of_iterations)
iteration_x_inds = np.linspace(0, nb_of_iterations, num = nb_of_iterations) 
# Plot the cost over the iterations
# plt.plot(minibatch_x_inds, minibatch_costs, 'r-', linewidth=2, label='cost minibatches')
plt.plot(iteration_x_inds, training_costs, 'b-', linewidth=3, label='cost full training set')
plt.plot(iteration_x_inds, validation_costs, 'k-', linewidth=0.5, label='cost validation set')
# Add labels to the plot
x_str = 'iteration   ' + 'val_acc={:.2f},test_acc={:.2f},epochces={:.0f},sigm='.format(val_accuracy,test_accuracy,int(iteration + 1)) + str(sigm)
plt.xlabel(x_str)
plt.ylabel('$\\xi$', fontsize=15)
plt.title('Decrease of cost over backprop iteration')
plt.legend()
x1,x2,y1,y2 = plt.axis()
plt.axis((0,nb_of_iterations,0,6.0))
plt.grid()
plt.savefig("/home1/kzw/mlp/learning_rate=0.1/SDENNet-new_three_sigmoid/loss" + str(sigm[0]) + ".png")
# plt.show()


# In[ ]:


# # Show confusion table
# conf_matrix = metrics.confusion_matrix(y_true, y_pred, labels=None)  # Get confustion matrix
# # Plot the confusion table
# class_names = ['${:d}$'.format(x) for x in range(0, 10)]  # Digit class names
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # Show class labels on each axis
# ax.xaxis.tick_top()
# major_ticks = range(0,10)
# minor_ticks = [x + 0.5 for x in range(0, 10)]
# ax.xaxis.set_ticks(major_ticks, minor=False)
# ax.yaxis.set_ticks(major_ticks, minor=False)
# ax.xaxis.set_ticks(minor_ticks, minor=True)
# ax.yaxis.set_ticks(minor_ticks, minor=True)
# ax.xaxis.set_ticklabels(class_names, minor=False, fontsize=15)
# ax.yaxis.set_ticklabels(class_names, minor=False, fontsize=15)
# # Set plot labels
# ax.yaxis.set_label_position("right")
# ax.set_xlabel('Predicted label')
# ax.set_ylabel('True label')
# fig.suptitle('Confusion table', y=1.03, fontsize=15)
# # Show a grid to seperate digits
# ax.grid(b=True, which=u'minor')
# # Color each grid cell according to the number classes predicted
# ax.imshow(conf_matrix, interpolation='nearest', cmap='binary')
# # Show the number of samples in each cell
# for x in xrange(conf_matrix.shape[0]):
#     for y in xrange(conf_matrix.shape[1]):
#         color = 'w' if x == y else 'k'
#         ax.text(x, y, conf_matrix[y,x], ha="center", va="center", color=color)       
# plt.show()

