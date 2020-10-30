# Video series this is based on: https://www.youtube.com/watch?v=aircAruvnKk
# Book the video is based on: http://neuralnetworksanddeeplearning.com/

from random import random
import pickle
import math
import numpy as np # Dependency: pip install numpy

def sigmoid(x):
	# Sigmoid "Squishification" function
	return 1 / (1 + math.exp(-x))\

def sigmoidPrime(z):
  # Derivative of sigmoid function
  return sigmoid(z)*(1-sigmoid(z))
  
def cost (a, y):
 # Cost function C = 0.5 * sum((y - a)^2)
 # where a is the activations vector, and y is the correct outputs vector
 return(np.sum(np.square(y-a))*0.5)

def costDerivative(a, y):
  # Derivative of cost function with respect to the activations: C' = a-y 
  # where a is the activations vector, and y is the correct outputs vector
  return(a-y)
