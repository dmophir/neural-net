import pandas as pd
import numpy as np
import sys

class Perceptron:
	# initialize weights to a small random number
	def __init__(self, dimension):
		self.weights = np.random.rand(dimension)
		self.bias = np.random.rand()

	# weight the inputs
	def activation_function(self, example):
		total = 0
		for i in range(len(self.weights)):
			total += self.weights[i] * example[i]
		total += self.bias
		return total

	# simple step function base on sign of weighted inputs
	def step_function(self, example):
		return np.sign(self.activation_function(example))

	# Sigmoid function based on weighted inputs
	def sigmoid_function(self, example):
		activator = self.activation_function(example)
		stepper = 1 / (1 + np.exp(-activator))
		return stepper

	def hyper_tangent(self, example):
		activator = self.activation_function(example)
		numerator = np.exp(activator) - np.exp(-activator)
		denominator = np.exp(activator) + np.exp(-activator)
		return numerator/denominator

	# Update weights based on gradient descent
	def update_weights(self, example):
		for i in range(len(self.weights)):
			self.weights[i] += example[-1] * example[i]
		self.bias += example[-1]

class Network:
	def __init__(self, ni, nh, no, dim):
		self.input_layer	=	Layer(ni, dim)
		self.hidden_layer	=	Layer(nh, dim)
		self.output_layer	=	Layer(no, dim)

	def feedforward(self, example):
		il_output = self.input_layer.send_input(example)

class Layer:
	def __init__(self, size, dim):
		self.perceptrons	=	self.build_layer(size, dim)

	def build_layer(self,size, dimension):
		layer = []
		for i in range(size):
			layer.append(Perceptron(dimension))
		return layer

	def send_input(self, example):
		results = []
		for neuron in self.perceptrons:
			results.append(neuron.step_function(example))
		return results


if __name__ == "__main__":
	# open data files
	input_data = pd.read_csv('data.csv', header=None, index_col=False)
	perceptron_weights = []

	# create a perceptron object
	percy = Perceptron(input_data.shape[1]-1)

	# train perceptron object
	converged = False
	while not converged:
		converged = True
		for point in input_data.itertuples(index=False):
			result = percy.step_function(point)
			if not result == point[-1]:
				converged = False
				percy.update_weights(point)
		iteration = list(percy.weights)
		iteration.append(percy.bias)
		perceptron_weights.append(tuple(iteration))

	# write output to csv
	output = pd.DataFrame(perceptron_weights)
	output.to_csv('results.csv', header=None, index=False)
	print(output)