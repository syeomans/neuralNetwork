# Video series this is based on: https://www.youtube.com/watch?v=aircAruvnKk
# Book the video is based on: http://neuralnetworksanddeeplearning.com/

from random import random
import pickle
import math
import random
import numpy as np # Dependency: pip install numpy

def sigmoid(z):
	# Sigmoid "Squishification" function
	return (1 / (1 + np.exp(-x)))

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
		self.activations = [np.array([0 for i in range(0,i)]) for i in self.shape]
		self.weights = [np.array([random.random() for i in range(0,i*j)]) for i,j in zip(self.shape[:-1], self.shape[1:])]
		self.biases = [np.array([random.random() for i in range(0,i)]) for i in self.shape[1:]]
	
	def feedForward(inputs):
		self.activations[0] = inputs
		for l in range(1, len(activations)):
			# Each activation is the sigmoid-squished weighted sum of the activations in the previous layer
			# i.e. sigma(a_l = a_l-1*w_l + b_l)
			w = self.weights[l]
			a_prev = self.activations[l-1]
			b = self.biases[l]
			self.activations[l] = sigmoid(np.dot(a_prev, w) + b)
			

			
			
##### Test code #####			
			
# Load abridged data
abridgedData = pickle.load(open('abridgedData.pickle', 'rb'))
pixels = abridgedData[0]

# Create a neural network
myNN = neuralNetwork([28*28, 16, 16, 10])

# Get output from first training image
myNN.feedForward(pixels)
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
print(len(wb)) # This thing is too big, so only print out the length to make sure it's the right length

# Get gradient of this network (note this is the positive gradient and not the negative gradient)
#grad = numpy.gradient(wb)
#print(grad[:10])
			
				
		
