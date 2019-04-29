import numpy as np 
from lstm_cell import cell
from activation_funcs import *

import numpy as np

class lstm:
	#input (in t), expected output (shifted by one time step), num of words (num of recurrences), array of expected outputs, learning rate
	
	def __init__ (self, input, output, recurrences, expected_output, learning_rate):
		#initial input 
		self.x = np.zeros(input)
		#input size 
		self.input = input
		#expected output (shifted by one time step)
		self.y = np.zeros(output)
		#output size
		self.output = output
		#weight matrix for interpreting results from hidden state cell (num words x num words matrix)
		#random initialization is crucial here
		self.w = np.full((output, output), 1 / (output * output))
		#matrix used in RMSprop in order to decay the learning rate
		self.G = np.zeros_like(self.w)
		#length of the recurrent network - number of recurrences i.e num of inputs
		self.recurrences = recurrences
		#learning rate 
		self.lr = learning_rate
		#array for storing inputs
		self.ia = np.zeros((recurrences+1,input))
		#array for storing cell states
		self.ca = np.zeros((recurrences+1,output))
		#array for storing outputs
		self.oa = np.zeros((recurrences+1,output))
		#array for storing hidden states
		self.ha = np.zeros((recurrences+1,output))
		#forget gate 
		self.af = np.zeros((recurrences+1,output))
		#input gate
		self.ai = np.zeros((recurrences+1,output))
		#cell state
		self.ac = np.zeros((recurrences+1,output))
		#output gate
		self.ao = np.zeros((recurrences+1,output))
		#array of expected output values
		self.expected_output = np.vstack((np.zeros(expected_output.shape[0]), expected_output.T))
		#declare lstm cell (input, output, amount of recurrence, learning rate)
		self.cell = cell(input, output, recurrences, learning_rate)

	
	#Here is where magic happens: We apply a series of matrix operations to our input and compute an output 
	def forward_prop(self):
		for i in range(1, self.recurrences+1):
			self.cell.x = np.hstack((self.ha[i-1], self.x))
			cs, hs, f, c, o = self.cell.forward_prop()
			#store cell state from the forward propagation
			self.ca[i] = cs #cell state
			self.ha[i] = hs #hidden state
			self.af[i] = f #forget state
			self.ai[i] = inp #inpute gate
			self.ac[i] = c #cell state
			self.ao[i] = o #output gate
			self.oa[i] = sigmoid(np.dot(self.w, hs)) #activate the weight*input
			self.x = self.expected_output[i-1]
		return self.oa
   
	
	def backward_pop(self):
		#backward_popagation of our weight matrices (Both in our Recurrent network + weight matrices inside cell)
		#start with an empty error value 
		totalError = 0
		#initialize matrices for gradient updates
		#First, these are RNN level gradients
		#cell state
		dfcs = np.zeros(self.output)
		#hidden state,
		dfhs = np.zeros(self.output)
		#weight matrix
		tu = np.zeros((self.output,self.output))
		#Next, these are level gradients
		#forget gate
		tfu = np.zeros((self.output, self.input+self.output))
		#input gate
		tiu = np.zeros((self.output, self.input+self.output))
		#cell unit
		tcu = np.zeros((self.output, self.input+self.output))
		#output gate
		tou = np.zeros((self.output, self.input+self.output))
		#loop backwards through recurrences
		for i in range(self.recurrences, -1, -1):
			#error = calculatedOutput - expectedOutput
			error = self.oa[i] - self.expected_output[i]
			#calculate update for weight matrix
			#Compute the partial derivative with (error * derivative of the output) * hidden state
			tu += np.dot(np.atleast_2d(error * sigmoid_derivative(self.oa[i])), np.atleast_2d(self.ha[i]).T)
			#Now propagate the error back to exit of  cell
			#1. error * RNN weight matrix
			error = np.dot(error, self.w)
			#2. set input values of  cell for recurrence i (horizontal stack of array output, hidden + input)
			self.cell.x = np.hstack((self.ha[i-1], self.ia[i]))
			#3. set cell state of cell cell for recurrence i (pre-updates)
			self.cell.cs = self.ca[i]
			#Finally, call the cell cell's backward_pop and retreive gradient updates
			#Compute the gradient updates for forget, input, cell unit, and output gates + cell states + hiddens states
			fu, iu, cu, ou, dfcs, dfhs = self.cell.backward_prop(error, self.ca[i-1], self.af[i], self.ai[i], self.ac[i], self.ao[i], dfcs, dfhs)
			#Accumulate the gradients by calculating total error (not necesarry, used to measure training progress)
			totalError += np.sum(error)
			#forget gate
			tfu += fu
			#input gate
			tiu += iu
			#cell state
			tcu += cu
			#output gate
			tou += ou
		#update cell matrices with average of accumulated gradient updates    
		self.cell.update(tfu/self.recurrences, tiu/self.recurrences, tcu/self.recurrences, tou/self.recurrences) 
		#update weight matrix with average of accumulated gradient updates  
		self.update(tu/self.recurrences)
		#return total error of this iteration
		return totalError
	
	def update(self, u):
		#Implementation of RMSprop in the vanilla world
		#We decay our learning rate to increase convergence speed
		self.G = 0.95 * self.G + 0.1 * u**2  
		self.w -= self.lr/np.sqrt(self.G + 1e-8) * u
		return
	
	#We define a sample function which produces the output once we trained our model
	#let's say that we feed an input observed at time t, our model will produce an output that can be 
	#observe in time t+1 
	def sample(self):
		 #loop through recurrences - start at 1 so the 0th entry of all output will be an array of 0's
		for i in range(1, self.recurrences+1):
			#set input for cell cell, combination of input (previous output) and previous hidden state
			self.cell.x = np.hstack((self.ha[i-1], self.x))
			#run forward prop on the cell cell, retrieve cell state and hidden state
			cs, hs, f, inp, c, o = self.cell.forward_prop()
			#store input as vector
			maxI = np.argmax(self.x)
			self.x = np.zeros_like(self.x)
			self.x[maxI] = 1
			self.ia[i] = self.x #Use np.argmax?
			#store cell states
			self.ca[i] = cs
			#store hidden state
			self.ha[i] = hs
			#forget gate
			self.af[i] = f
			#input gate
			self.ai[i] = inp
			#cell state
			self.ac[i] = c
			#output gate
			self.ao[i] = o
			#calculate output by multiplying hidden state with weight matrix
			self.oa[i] = sigmoid(np.dot(self.w, hs))
			#compute new input
			maxI = np.argmax(self.oa[i])
			newX = np.zeros_like(self.x)
			newX[maxI] = 1
			self.x = newX
		#return all outputs    
		return self.oa
