import numpy as np 
from activation_funcs import *
	
class cell:
	# LSTM cell (input, output, amount of recurrence, learning rate)
	def __init__ (self, input, output, recurrences, learning_rate):
		#input size
		self.x = np.zeros(input+output)
		#input size
		self.input = input + output
		#output 
		self.y = np.zeros(output)
		#output size
		self.output = output
		#cell state intialized as size of prediction
		self.cs = np.zeros(output)
		#how often to perform recurrence
		self.recurrences = recurrences
		#balance the rate of training (learning rate)
		self.learning_rate = learning_rate
		#init weight matrices for our gates
		#forget gate
		self.f = np.random.random((output, input+output))
		#input gate
		self.i = np.random.random((output, input+output))
		#cell state
		self.c = np.random.random((output, input+output))
		#output gate
		self.o = np.random.random((output, input+output))
		#forget gate gradient
		self.Gf = np.zeros_like(self.f)
		#input gate gradient
		self.Gi = np.zeros_like(self.i)
		#cell state gradient
		self.Gc = np.zeros_like(self.c)
		#output gate gradient
		self.Go = np.zeros_like(self.o)

	
	#Here is where magic happens: We apply a series of matrix operations to our input and compute an output 
	def forward_prop(self):
		f = sigmoid(np.dot(self.f, self.x))
		self.cs *= f
		i = sigmoid(np.dot(self.i, self.x))
		c = tanh(np.dot(self.c, self.x))
		self.cs += i * c
		o = sigmoid(np.dot(self.o, self.x))
		self.y = o * tanh(self.cs)
		return self.cs, self.y, f, i, c, o
	
   
	def backward_prop(self, e, pcs, f, i, c, o, dfcs, dfhs):
		#error = error + hidden state derivativeative. we clip the value between -6 and 6 to prevent vanishing
		e = np.clip(e + dfhs, -6, 6)
		#multiply error by activated cell state to compute output derivativeative
		do = tanh(self.cs) * e
		#output update = (output derivative * activated output) * input
		ou = np.dot(np.atleast_2d(do * tanh_derivative(o)).T, np.atleast_2d(self.x))
		#derivativeative of cell state = error * output * derivativeative of cell state + derivative cell
		#compute gradients and update them in reverse order!
		dcs = np.clip(e * o * tanh_derivative(self.cs) + dfcs, -6, 6)
		#derivativeative of cell = derivativeative cell state * input
		dc = dcs * i
		#cell update = derivativeative cell * activated cell * input
		cu = np.dot(np.atleast_2d(dc * tanh_derivative(c)).T, np.atleast_2d(self.x))
		#derivativeative of input = derivative cell state * cell
		di = dcs * c
		#input update = (derivative input * activated input) * input
		iu = np.dot(np.atleast_2d(di * sigmoid_derivative(i)).T, np.atleast_2d(self.x))
		#derivative forget = derivative cell state * all cell states
		df = dcs * pcs
		#forget update = (derivative forget * derivative forget) * input
		fu = np.dot(np.atleast_2d(df * sigmoid_derivative(f)).T, np.atleast_2d(self.x))
		#derivative cell state = derivative cell state * forget
		dpcs = dcs * f
		#derivative hidden state = (derivative cell * cell) * output + derivative output * output * output derivative input * input * output + derivative forget
		#* forget * output
		dphs = np.dot(dc, self.c)[:self.output] + np.dot(do, self.o)[:self.output] + np.dot(di, self.i)[:self.output] + np.dot(df, self.f)[:self.output] 
		#return update gradients for forget, input, cell, output, cell state, hidden state
		return fu, iu, cu, ou, dpcs, dphs
			
	def update(self, fu, iu, cu, ou):
		#Update forget, input, cell, and output gradients
		self.Gf = 0.9 * self.Gf + 0.1 * fu**2 
		self.Gi = 0.9 * self.Gi + 0.1 * iu**2   
		self.Gc = 0.9 * self.Gc + 0.1 * cu**2   
		self.Go = 0.9 * self.Go + 0.1 * ou**2   
		
		#Update our gates using our gradients
		self.f -= self.learning_rate/np.sqrt(self.Gf + 1e-8) * fu
		self.i -= self.learning_rate/np.sqrt(self.Gi + 1e-8) * iu
		self.c -= self.learning_rate/np.sqrt(self.Gc + 1e-8) * cu
		self.o -= self.learning_rate/np.sqrt(self.Go + 1e-8) * ou
		return

	def train(self):
		pass