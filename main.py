from lstm import musicLSTM
import numpy as np
import sys
import pickle

def main(args):
	if '-t' in args[0]:
		inputs = pickle.load(open('pickle/classical_inputs', 'rb'))
		outputs = pickle.load(open('pickle/classical_outputs', 'rb'))
		in_shape = (inputs.shape[1], inputs.shape[2])
		out_size = outputs.shape[1]
		piano_lstm = musicLSTM(in_shape, out_size)
		piano_lstm.train(inputs, outputs, 'model.h5')

	elif '-g' in args[0]:
		piano_lstm = musicLSTM((1,1), 5)
		piano_lstm.predict('model.h5')
	pass

def train():
	pass

def generate():
	pass

if __name__ == '__main__':
	main(sys.argv[1:])