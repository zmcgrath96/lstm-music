from lstm import musicLSTM
import numpy as np
import sys
import pickle
import os

def main(args):
	arch = int(args[1])
	if '-t=piano' in args[0]:

		# get inputs and outputs
		piano_in, piano_out = get_input_and_output('piano', arch)

		# get shapes for input and output
		in_shape = (piano_in.shape[1], piano_in.shape[2])
		out_shape = piano_out.shape[1]

		# train 
		path = 'models/architecture{}'.format(arch)
		name = '/piano-lstm'
		if not os.path.exists(path):
			os.makedirs(path)
		lstm = musicLSTM(in_shape, out_shape)
		lstm.train(piano_in, piano_out, path + name, it=10, batch=64)

	elif '-t=bass' in args[0]:
		# get inputs and outputs
		bass_in, bass_out = get_input_and_output('bass', arch)

		# get shapes for input and output
		in_shape = (bass_in.shape[1], bass_in.shape[2])
		out_shape = bass_out.shape[1]

		# train 
		path = 'models/architecture{}'.format(arch)
		name = '/bass-lstm'
		if not os.path.exists(path):
			os.makedirs(path)
		lstm = musicLSTM(in_shape, out_shape)
		lstm.train(bass_in, bass_out, path + name, it=10, batch=64)

	elif '-t=sax' in args[0]:
		# get inputs and outputs
		sax_in, sax_out = get_input_and_output('sax', arch)

		# get shapes for input and output
		in_shape = (sax_in.shape[1], sax_in.shape[2])
		out_shape = sax_out.shape[1]

		# train 
		path = 'models/architecture{}'.format(arch)
		name = '/sax-lstm'
		if not os.path.exists(path):
			os.makedirs(path)
		lstm = musicLSTM(in_shape, out_shape)
		lstm.train(bass_in, bass_out, path + name, it=10, batch=64)

	elif '-g' in args[0]:
		
		pass

def get_input_and_output(inst, arch):
	# load embeddings and encodings
	arch_path = 'pickle/architecture{}/{}_'.format(arch, inst)
	input = pickle.load(open(arch_path + 'inputs', 'rb'))
	output = pickle.load(open(arch_path + 'outputs', 'rb'))
	input = np.array(input).reshape((len(input), 100, 1))
	output = np.array(output).reshape((len(output), len(output[0])))
	return input, output

if __name__ == '__main__':
	main(sys.argv[1:])