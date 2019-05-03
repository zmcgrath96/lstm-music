import pickle
import numpy as np
import glob
import os
import sys
from music21 import converter, instrument, note, chord
from utils import *
import math
from clean_data import clean

# music genre
directory = 'clean_jazz'

def main():
	# check if given directory exists
	exists = os.path.isdir(directory)
	if not exists:
		sys.exit('Error: given directory "' + directory + '" does not exist')

	# get list note/chord/rest encodings
	exists = os.path.isfile('pickle/' + directory + '_encodings') and os.path.isfile('pickle/' + directory + '_embedded')
	if not exists:
		encodings, embedded_songs = get_notes()
	else:
		encodings = pickle.load(open('pickle/' + directory + '_encodings', 'rb'))
		embedded_songs = pickle.load(open('pickle/' + directory + '_embedded', 'rb'))

	# create network inputs and outputs, x and y
	# x = a specified sequence of notes (ex. first 10 notes)
	# y = the next note in the sequence (ex. the 11th note)

	# embedded songs order (piano -> bass -> sax)
	prepare_sequences(embedded_songs, len(encodings))

# returns a sequential list of all notes from all songs in ./classical directory
def get_notes(note_width=.25):
	# if pickled result exists, return that
	# exists = os.path.isfile('pickle/' + directory + '_notes')	
	# if exists:
	# 	return pickle.load(open('pickle/' + directory + '_notes', 'rb'))

	# if pickled result does not exist, create it and pickle it
	piano = []
	bass = []
	sax = []
	songs = {}
	num_songs = 0
	for file in glob.glob(directory + "/*.mid"):
		midi = None
		try:
			midi = converter.parse(file)
		except Exception as e:
			print('Could not parse file: {}'.format(file))
			continue

		print("Parsing %s" % file)
		num_songs += 1
		songs[file] = {}
		instruments = instrument.partitionByInstrument(midi)
		for i in instruments:
			name = i.getInstrument().instrumentName
			if name == 'Piano':
				songs[file]['piano'] = i
				piano.append(i)
			elif name == 'Acoustic Bass':
				songs[file]['bass'] = i
				bass.append(i)
			elif name == 'Saxophone':
				songs[file]['sax'] = i
				sax.append(i)

	enumerated_notes, embedded = clean(songs, note_width)

	output = []

	for s in embedded:
		if len(s[2]) != 0:
			output.append(s)

	pickle.dump(enumerated_notes, open('pickle/' + directory + '_encodings', 'wb'))
	pickle.dump(output, open('pickle/' + directory + '_embedded', 'wb'))

	return enumerated_notes, output


# creates neural network inputs and outputs
def prepare_sequences(songs, n_unique_notes, input_sequence_length=100):

	os.mkdir('pickle/architecture1')
	os.mkdir('pickle/architecture2')
	os.mkdir('pickle/architecture3')

	#TODO: figure out how to structure neural network inputs / outputs

	## arch 1

	# get piano sequences
	input_piano_sequences = []
	output_piano_notes = []

	for s in songs:
		piano_part = s[0]
		for i in range(0, len(piano_part) - input_sequence_length):
			sequence_in = piano_part[i:i + input_sequence_length]
			note_out = piano_part[i + input_sequence_length]

			input_piano_sequences.append(sequence_in)
			output_piano_notes.append(to_one_hot(note_out, n_unique_notes))

	pickle.dump(input_piano_sequences, open('pickle/architecture1/piano_inputs', 'wb'))
	pickle.dump(output_piano_notes, open('pickle/architecture1/piano_outputs', 'wb'))

	# get bass and sax notes
	input_bass_notes = [] # piano notes
	output_bass_notes = []

	input_sax_notes = [] # bass notes
	output_sax_notes = []

	for s in songs:
		piano = s[0]
		bass = s[1]
		sax = s[2]

		for i in range(len(piano)):
			input_bass_notes.append(piano[i])
			output_bass_notes.append(to_one_hot(bass[i], n_unique_notes))

			input_sax_notes.append(bass[i])
			output_bass_notes.append(to_one_hot(sax[i], n_unique_notes))

	pickle.dump(input_bass_notes, open('pickle/architecture1/bass_inputs', 'wb'))
	pickle.dump(output_bass_notes, open('pickle/architecture1/bass_outputs', 'wb'))

	pickle.dump(input_sax_notes, open('pickle/architecture1/sax_inputs', 'wb'))
	pickle.dump(output_sax_notes, open('pickle/architecture1/sax_outputs', 'wb'))

	## arch 2

	# piano inputs and outputs same as architecture 1
	pickle.dump(input_piano_sequences, open('pickle/architecture2/piano_inputs', 'wb'))
	pickle.dump(output_piano_notes, open('pickle/architecture2/piano_outputs', 'wb'))

	# get bass notes (piano sequence to next bass note)
	
	input_bass_sequences = [] # piano sequences
	output_bass_notes = []

	for s in songs:
		piano_part = s[0]
		bass_part = s[1]

		for i in range(0, len(piano_part) - input_sequence_length):
			sequence_in = piano_part[i:i + input_sequence_length]
			note_out = bass_part[i + input_sequence_length]

			input_bass_sequences.append(sequence_in)
			output_bass_notes.append(to_one_hot(note_out, n_unique_notes))

	pickle.dump(input_bass_sequences, open('pickle/architecture2/bass_inputs', 'wb'))
	pickle.dump(output_bass_notes, open('pickle/architecture2/bass_outputs', 'wb'))

	# get sax

	input_sax_sequences = [] # bass sequences
	output_sax_notes = []

	for s in songs:
		bass_part = s[1]
		sax_part = s[2]

		for i in range(0, len(bass_part) - input_sequence_length):
			sequence_in = bass_part[i:i + input_sequence_length]
			note_out = bass_part[i + input_sequence_length]

			input_sax_sequences.append(sequence_in)
			output_sax_notes.append(to_one_hot(note_out, n_unique_notes))

	pickle.dump(input_sax_sequences, open('pickle/architecture2/sax_inputs', 'wb'))
	pickle.dump(output_sax_notes, open('pickle/architecture2/sax_outputs', 'wb'))

	## arch 3

	input_piano_sequences = []
	output_piano_notes = []
	output_bass_notes = []
	output_sax_notes = []

	for s in songs:
		piano_part = s[0]
		bass_part = s[1]
		sax_part = s[2]

		for i in range(0, len(piano_part) - input_sequence_length):
			sequence_in = piano_part[i:i + input_sequence_length]
			piano_note_out = piano_part[i + input_sequence_length]
			bass_note_out = bass_part[i + input_sequence_length]
			sax_note_out = sax_part[i + input_sequence_length]

			input_piano_sequences.append(sequence_in)
			output_piano_notes.append(to_one_hot(piano_note_out, n_unique_notes))
			output_bass_notes.append(to_one_hot(bass_note_out, n_unique_notes))
			output_sax_notes.append(to_one_hot(sax_note_out, n_unique_notes))

	pickle.dump(input_piano_sequences, open('pickle/architecture3/piano_inputs', 'wb'))
	pickle.dump(input_piano_sequences, open('pickle/architecture3/bass_inputs', 'wb'))
	pickle.dump(input_piano_sequences, open('pickle/architecture3/sax_inputs', 'wb'))

	pickle.dump(output_piano_notes, open('pickle/architecture3/piano_outputs', 'wb'))
	pickle.dump(output_bass_notes, open('pickle/architecture3/bass_outputs', 'wb'))
	pickle.dump(output_sax_notes, open('pickle/architecture3/sax_outputs', 'wb'))

# converts value to a one-hot representation
def to_one_hot(value, n_unique_notes):
	x = np.zeros(n_unique_notes, dtype=np.int8)
	x[int(value * n_unique_notes)] = 1
	return x

if __name__ == '__main__':
	main()