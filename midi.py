import pickle
import numpy as np
import glob
import os
import sys
from music21 import converter, instrument, note, chord

directory = 'classical'

def main():
	
	# check if given directory exists
	exists = os.path.isdir(directory)
	if not exists:
		sys.exit('Error: given directory "' + directory + '" does not exist')

	# get list of all notes
	notes = get_notes()

	# get list of all unique notes
	n_unique_notes = len(set(notes))

	# create network inputs and outputs, x and y
	# x = a specified sequence of notes (ex. first 10 notes)
	# y = the next note in the sequence (ex. the 11th note)
	x, y = prepare_sequences(notes, n_unique_notes)

	pickle.dump(x, open('pickle/' + directory + '_inputs', 'wb'))
	pickle.dump(y, open('pickle/' + directory + '_outputs', 'wb'))

# returns a sequential list of all notes from all songs in ./classical directory
def get_notes():

	# if pickled result exists, return that
	exists = os.path.isfile('pickle/' + directory + '_notes')	
	if exists:
		return pickle.load(open('pickle/' + directory + '_notes', 'rb'))

	# if pickled result does not exist, create it and pickle it
	notes = []

	for file in glob.glob(directory + "/*.mid"):
		midi = converter.parse(file)

		print("Parsing %s" % file)

		notes_to_parse = None

		try: # file has instrument parts
			s2 = instrument.partitionByInstrument(midi)
			notes_to_parse = s2.parts[0].recurse() 
		except: # file has notes in a flat structure
			notes_to_parse = midi.flat.notes

		for element in notes_to_parse:
			if isinstance(element, note.Note):
				notes.append(str(element.pitch))
			elif isinstance(element, chord.Chord):
				notes.append('.'.join(str(n) for n in element.normalOrder))

	pickle.dump(notes, open('pickle/' + directory + '_notes', 'wb'))

	return notes

# creates neural network inputs and outputs
def prepare_sequences(notes, n_unique_notes):
	sequence_length = 100

	# get all pitch names
	pitchnames = sorted(set(item for item in notes))

	 # create a dictionary to map pitches to integers
	note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

	network_input = []
	network_output = []

	# create input sequences and the corresponding outputs
	for i in range(0, len(notes) - sequence_length, 1):
		sequence_in = notes[i:i + sequence_length]
		sequence_out = notes[i + sequence_length]
		network_input.append([note_to_int[char] for char in sequence_in])
		network_output.append(note_to_int[sequence_out])

	n_patterns = len(network_input)

	# reshape the input into a format compatible with LSTM layers
	network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
	# normalize input
	network_input = network_input / float(n_unique_notes)

	network_output = to_one_hot(network_output, n_unique_notes)

	return (network_input, network_output)

# converts array of values to a one-hot representation
def to_one_hot(array, n_unique_notes):
	x = np.zeros((len(array), n_unique_notes), dtype=np.int8)
	x[np.arange(len(array)), array] = 1
	return x

if __name__ == '__main__':
	main()