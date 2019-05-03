import pickle
import numpy as np
import glob
import os
import sys
from music21 import converter, instrument, note, chord
from utils import *
import math

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
	x, y = prepare_sequences(embedded_songs)

	pickle.dump(x, open('pickle/' + directory + '_inputs', 'wb'))
	pickle.dump(y, open('pickle/' + directory + '_outputs', 'wb'))

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

	num = 0
	# get all of the unique combinations of notes and chords
	# after saving all of the notes as embeddings in the enumerated_notes dict, 
	# add all of the notes to the array for each song for each of the instruments
	print('Getting encodings')
	enumerated_notes = {}
	embedded = [[] for _ in range(len(songs))]
	s_count = 0
	for s in songs:
		embedded[s_count] = [[] for _ in range(3)]
		s_l = max([len(songs[s][i]) for i in songs[s]])
		scaled_s_l = math.ceil(s_l / note_width)
		i_count = 0
		for inst in ('piano', 'bass', 'sax'):
			if len(songs[s]) < 3:
				continue
			notes = songs[s][inst].notesAndRests.activeElementList
			embedded[s_count][i_count] = [0 for _ in range(scaled_s_l)]
			n_count = 0
			for i in range(1, len(notes)):
				e_complete, off_n_count = get_all_in_offset(notes, i)
				if e_complete != '':
					if e_complete not in enumerated_notes:
						enumerated_notes[e_complete] = num
						num += 1
					offset = notes[i].offset
					if offset % note_width == 0:
						embedded[s_count][i_count][n_count] = enumerated_notes[e_complete]
					n_count += off_n_count
			i_count += 1
		s_count += 1

	print('done')

	output = []

	for s in embedded:
		if len(s[2]) != 0:
			output.append(s)

	pickle.dump(enumerated_notes, open('pickle/' + directory + '_encodings', 'wb'))
	pickle.dump(output, open('pickle/' + directory + '_embedded', 'wb'))

	return enumerated_notes, output

def get_all_in_offset(notes, i):
	n = notes[i]
	offset = n.offset
	e_note = []
	e_chord = []
	j = 0
	while i+j < len(notes) and notes[i+j].offset == offset:
		if not isinstance(n, note.Note) and not isinstance(n, chord.Chord) and not isinstance(n, note.Rest):
			j += 1
			continue
		if n.isChord:
			e_chord += chord_to_notes(n)
		elif n.isRest:
			e_note.append(n.name)
		else:
			e_note.append(n.nameWithOctave)
		j += 1

	return stringify_notes(sort_notes(e_note)) + stringify_chords(sort_chords(e_chord)), j

# creates neural network inputs and outputs
def prepare_sequences(songs, input_sequence_length=100):

	#TODO: figure out how to structure neural network inputs / outputs

	network_input = []
	network_output = []

	# create input sequences and the corresponding outputs
	for i in range(0, len(notes) - input_sequence_length, 1):
		sequence_in = notes[i:i + input_sequence_length]
		sequence_out = notes[i + input_sequence_length]
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