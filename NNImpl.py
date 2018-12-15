import numpy as np

# Each row is a training example, each column is a feature  [X1, X2, X3]
X = np.array(([0, 1, 0, 0, 0, 1, 1, 0, 0], [ 1, 1, 0, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0, 1, 1, 1]), dtype=float)
y = np.array(([1, 0, 1], [1, 1, 0], [1, 0, 0],[0, 1, 1]), dtype=float)

# X = np.array(([0, 1, 0, 0, 0, 1, 1, 0, 0], [ 1, 1, 0, 0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 1, 1, 0, 0, 1], [1, 1, 1, 0, 1, 0, 1, 1, 1]), dtype=float)
# y = np.array(([1,0,1], [1,1,0], [1,0,0],[0 , 1, 1]), dtype=float)


# Define useful functions

# Activation function
def sigmoid(t):
    return 1 / (1 + np.exp(-t))
def relu(t):
    return t * (t > 0)
def tanh(x):

    return np.tanh(x)

# Derivative of sigmoid
def sigmoid_gradient(p):
    return p * (1 - p)

def tanh_gradient(p):
    return 1-(p**2)

def relu_gradient(z):
    z[z <= 0] = 0
    z[z > 0] = 1

    return z

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Class definition
class NeuralNetwork:
    def __init__(self, x, y):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 6)  # considering we have 4 nodes in the hidden layer
        self.weights2 = np.random.rand(6, 3)
        self.y = y
        self.output = np.zeros(y.shape)
        print(self.input.shape," shape: ", self.input.shape[1], " ***** ", self.y.shape)

    def feedforward(self):
        #print("Check feed foward******", np.dot(self.input, self.weights1))
        self.layer1 = sigmoid(np.dot(self.input, self.weights1))
        self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))
        #print("@@@@ ****** @@@@ ", np.dot(self.input, self.weights1).shape, np.dot(self.layer1, self.weights2).shape )

        return (self.layer2)

    def backprop(self):
        d_weights2 = np.dot(self.layer1.T, 2 * (self.y - self.output) * sigmoid_gradient(self.output))
        d_weights1 = np.dot(self.input.T, np.dot(2 * (self.y - self.output) * sigmoid_gradient(self.output),
                                                 self.weights2.T) * sigmoid_gradient(self.layer1))

        self.weights1 += d_weights1
        self.weights2 += d_weights2

    def train(self, X, y):
        self.output = self.feedforward()
        self.backprop()


NN = NeuralNetwork(X, y)
for i in range(1500):  # trains the NN 1,000 times
    if i % 100 == 0:
        print("for iteration # " + str(i) + "\n")
        print("Input : \n" + str(X))
        print("Actual Output: \n" + str(y))
        print("Predicted Output: \n" + str((NN.feedforward())))
        print("Loss: \n" + str(np.mean(np.square(y - NN.feedforward()))))  # mean sum squared loss
        print("\n")

    NN.train(X, y)
#print(tanh_gradient(4))
