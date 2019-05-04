from lstm import musicLSTM
import numpy as np
import sys
import pickle

def main(args):
	if '-t' in args[0]:

		# get inputs and outputs
		piano_in, piano_out = get_input_and_output(0)

		# get shapes for input and output
		in_shape = (piano_in.shape[1], piano_in.shape[2])
		out_shape = piano_out.shape[1]

		# train 
		lstm = musicLSTM(in_shape, out_shape)
		lstm.train(piano_in, piano_out, 'piano_lstm.h5', it=10)

		# get inputs and outputs
		bass_in, bass_out = get_input_and_output(1)

		# get shapes for input and output
		in_shape = (bass_in.shape[1], bass_in.shape[2])
		out_shape = bass_out.shape[1]

		# train 
		lstm = musicLSTM(in_shape, out_shape)
		lstm.train(bass_in, bass_out, 'bass_lstm.h5', it=10)

		# get inputs and outputs
		sax_in, sax_out = get_input_and_output(1)

		# get shapes for input and output
		in_shape = (sax_in.shape[1], sax_in.shape[2])
		out_shape = sax_out.shape[1]

		# train 
		lstm = musicLSTM(in_shape, out_shape)
		lstm.train(bass_in, bass_out, 'sax_lstm.h5', it=10)

	elif '-g' in args[0]:
		
		pass

def get_input_and_output(index):
	# load embeddings and encodings
	embedded = pickle.load(open('pickle/clean_jazz_embedded', 'rb'))
	encodings = pickle.load(open('pickle/clean_jazz_encodings', 'rb'))
	num_classes = len(encodings)

	# convert embedded to numpy array
	embedded = np.array(embedded)

	# splice of the portions for each instrument
	instrument_embedded = embedded[:,index]
	input_embedded = embedded[:,0]

	# convert to sequences of 100 inputs and get outputs
	instrument_in, instrument_out = convert_to_seq_of_100(input_embedded, instrument_embedded, num_classes)
	return instrument_in, instrument_out

def convert_to_seq_of_100(input_arr, output_arr, num_classes):
	inputs = []
	outputs = []
	for i in range(len(input_arr)):
		for j in range(99, len(input_arr[i]) - 1):
			inputs.append(input_arr[i][j-99:j+1])
			outputs.append(to_one_hot(output_arr[i][j+1], num_classes))
	inputs = np.array(inputs).reshape((len(inputs), 100, 1))
	inputs = np.array(inputs).reshape((len(inputs), 100, 1))
	outputs = np.array(outputs)
	return inputs, outputs

def to_one_hot(value, n_unique_notes):
	x = np.zeros(n_unique_notes, dtype=np.int8)
	x[int(value * n_unique_notes)] = 1
	return x


if __name__ == '__main__':
	main(sys.argv[1:])