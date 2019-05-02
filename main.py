from lstm import musicLSTM
from nn import musicNN
import numpy as np
import sys
import pickle

def main(args):
	if '-t' in args[0]:

		# read in piana inputs
		piano_inputs = pickle.load(open('pickle/piano_inputs', 'rb'))
		# read in piano outputs
		piano_outputs = pickle.load(open('pickle/piano_inputs', 'rb'))

		# run piano through lstm
		in_shape = (piano_inputs.shape[1], piano_inputs.shape[2])
		out_size = piano_outputs.shape[1]
		piano_lstm = musicLSTM(in_shape, out_size)
		piano_lstm.train(piano_inputs, piano_outputs, 'piano_model.h5')

		# read in bass outputs
		bass_outputs = pickle.load(open('pickle/bass_outputs', 'rb'))

		# reshape piano inputs
		piano_nn_inputs = piano_inputs.reshape((piano_inputs.shape[0] * piano_inputs.shape[1], piano_inputs.shape[2]))
		nn_in_shape = (piano_nn_inputs.shape[1],)

		# run bass through nn
		bass_nn = musicNN(nn_in_shape, out_size)
		bass_nn.train(piano_nn_inputs, bass_outputs, 'bass_model.h5')

		# read in sax outputs
		sax_outputs = pickle.load(open('pickle/sax_outputs', 'rb'))

		# run sax through nn
		sax_nn = musicNN(nn_in_shape, out_size)
		sax_nn.train(piano_nn_inputs, sax_outputs, 'sax_model.h5')


	elif '-g' in args[0]:
		
		pass

def train():
	pass

def generate():
	pass

if __name__ == '__main__':
	main(sys.argv[1:])