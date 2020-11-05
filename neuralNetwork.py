# Video series this is based on: https://www.youtube.com/watch?v=aircAruvnKk
# Book the video is based on: http://neuralnetworksanddeeplearning.com/

from random import random
import pickle
import math
import random
import numpy as np # Dependency: pip install numpy

def sigmoid(z):
    # Sigmoid "Squishification" function
    return (1 / (1 + np.exp(-z)))

def sigmoidPrime(z):
    # Derivative of sigmoid function
    return (sigmoid(z)*(1-sigmoid(z)))

def cost (a, y):
    # Cost function C = 0.5 * sum((y - a)^2)
    # where a is the activations vector, and y is the correct outputs vector
    return(np.sum(np.square(y-a))*0.5)

def costDerivative(a, y):
    # Derivative of cost function with respect to the activations: C' = a-y
    # where a is the activations vector, and y is the correct outputs vector
    return(a-y)

class neuralNetwork:
    def __init__(self, networkShape):
        self.shape = networkShape
        #for i in self.shape:
            #j in range(0,i):
        #self.activations = [[0 for j in range(0,i)] for i in self.shape]
        self.activations = [np.zeros(i) for i in self.shape]
        self.z = [np.zeros(i) for i in self.shape]
        self.weights = [np.random.randn(i, j) for i, j in zip(self.shape[1:], self.shape[:-1])]
        self.biases = [np.random.randn(i, 1) for i in self.shape[1:]]
        #self.weights = [np.random.randn(y, x) for x, y in zip(self.shape[:-1], self.shape[1:])]
        #self.biases = [np.random.randn(y, 1) for y in self.shape[1:]]
        print('activations')
        print(self.activations)
        print()
        print('weights')
        print(self.weights)
        print()
        print('biases')
        print(self.biases)
        print()


    def feedForward(self, inputs):
        self.activations[0] = np.array(inputs)
        print("layer 0: ")
        print(self.activations[0])
        l = 0
        for b, w in zip(self.biases, self.weights):
            b = np.concatenate(b)
            z = np.dot(w, self.activations[l])+b
            self.z[l+1] = z
            self.activations[l+1] = sigmoid(z)
            print("layer " + str(l+1) + ":")
            print(self.activations[l+1])
            l += 1
        print()
        #for l in range(1, len(self.shape)): # for each layer, "l"
            # Each activation is the sigmoid-squished weighted sum of the activations in the previous layer
            # i.e. sigma(a_l = a_l-1*w_l + b_l)
            #w = self.weights[l-1]
            #a_prev = self.activations[l-1]
            #b = self.biases[l-1]
            #self.activations[l] = sigmoid(np.dot(w, a_prev) + b)
        #print(self.activations)
        print()

    def backPropagate(self, inputs, correctAnswers):
        '''
        See here for formulas: http://neuralnetworksanddeeplearning.com/chap2.html#the_code_for_backpropagation
        1. Input x: Set the corresponding activation a^1 for the input layer.
        2. Feedforward: For each l=2,3,…,L compute z^l and a^l.
        3. Output error δ^L: Compute the error vector δ^L.
        4. Backpropagate the error: For each l=L−1,L−2,…,2 compute δ^l).
        5. Output: The gradient of the cost function is given by (see formulas)
        '''
        gradC_b = [np.zeros(b.shape) for b in self.biases]
        gradC_w = [np.zeros(w.shape) for w in self.weights]

        # (1) and (2)
        self.feedForward(inputs)

        # (3)
        #print(self.activations[-1])
        #print()
        gradC_a = costDerivative(self.activations[-1], correctAnswers)
        print("gradC_a: ")
        print(gradC_a)
        print("z[-1]: ")
        print(self.z[-1])
        errorL = gradC_a * sigmoidPrime(self.z[-1])
        print("errorL: ")
        print(errorL)
        gradC_b[-1] = errorL
        print("gradC_b:")
        print(gradC_b[-1])
        print("activation:")
        a = self.activations[-2]
        a.shape = (1, len(a))
        errorL.shape = (len(errorL), 1)
        gradC_w[-1] = np.dot(errorL, a)
        print("gradC_w: ")
        print(gradC_w[-1])
        print("\n\n")

        # (4) and (5)
        for l in range(len(self.shape)-2, -1, -1):
            print("l: \n" + str(l))
            errorL = np.dot(self.weights[l].transpose(),errorL) * sigmoidPrime(self.z[l])
            gradC_b[l] = errorL
            print("errorL: ")
            print(errorL)
            print("activations:")
            print(self.activations[l-1].transpose())
            gradC_w[l] = np.dot(errorL, self.activations[l-1].transpose())

        return(gradC_b, gradC_w)




##### Test code #####

# Load abridged data
#abridgedData = pickle.load(open('abridgedData.pickle', 'rb'))
#pixels = abridgedData[0]
pixels = [random.random() for i in range(0, 4)]
correctAnswers = [random.random() for i in range(0, 2)]
#print(pixels)

# Create a neural network
myNN = neuralNetwork([4, 3, 3, 2])

# Get output from first training image
myNN.backPropagate(pixels, correctAnswers)
print("hello")
#print(myNN.getOutput())
print(myNN.activations[-1])

# Get correct answer from first training image
#answers = [0.0 for i in range(0,10)]
#answers[pixels[0]-1] = 1.0

# Get cost for this output
#myCost = myNN.cost(answers)
#print(myCost)

# Get current weights and biases of this network
#wb = myNN.getwb()
#print(len(wb)) # This thing is too big, so only print out the length to make sure it's the right length

# Get gradient of this network (note this is the positive gradient and not the negative gradient)
#grad = numpy.gradient(wb)
#print(grad[:10])
