import numpy as np 
from activation_funcs import sigmoid, tanh_activation

# When looking at images, here is the common representation of the parameters
# prev_cell_state: C(t-1)	-- numpy array
# prev_action_state: a(t-1) or h(t-1)	-- numpy array
# data_set: x(t)	-- numpy array
# weights:	-- dict 
# 	forget weights: f(w) -- float
# 	input data weights: i(w) -- float
# 	gate weights: g(w) -- float
#	output weights: o(w) -- float
#
# NOTE: difference between np.matmul and np.multiply is
#		matmul is matrix multiplication and multiply is element wise
#
# returns the cell activations, current cell state, current action state
def cell(prev_cell_state, prev_action_state, data_set, weights):
	f_w = weights['f_w']
	i_w = weights['i_w']
	g_w = weights['g_w']
	o_w = weights['o_w']

	# activation state concatenate with the data set
	activated_dataset = np.concatenate((data_set, prev_action_state), axis=1)

	#activated inputs for each of the weights
	#sigmoids
	#forget
	f_a = sigmoid(np.matmul(activated_dataset, f_w)) 
	#input
	i_a = sigmoid(np.matmul(activated_dataset, i_w))
	#output
	o_a = sigmoid(np.matmul(activated_dataset, o_w))

	#tanh
	#gate
	g_a = tanh_activation(np.matmul(activated_dataset, g_w))

	#store the activations to be modified in the back propagation
	activations = {}
	activations['f_a'] = f_a
	activations['i_a'] = i_a
	activations['o_a'] = o_a
	activations['g_a'] = g_a

	# output of c(t)
	curr_cell_state = np.multiply(prev_cell_state, f_a) + np.multiply(i_a, g_a)

	# output of the cell, AKA action state
	curr_action_state = np.multiply(tanh_activation(curr_cell_state), o_a)

	return activations, curr_cell_state, curr_action_state
