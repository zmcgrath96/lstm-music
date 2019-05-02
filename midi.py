import pickle
import numpy as np
import glob
import os
import sys
from music21 import converter, instrument, note, chord
from utils import *

# music genre
directory = 'clean_jazz'

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
def get_notes(note_width=.25):
	# if pickled result exists, return that
	exists = os.path.isfile('pickle/' + directory + '_notes')	
	if exists:
		return pickle.load(open('pickle/' + directory + '_notes', 'rb'))

	# if pickled result does not exist, create it and pickle it
	piano = []
	bass = []
	sax = []
	songs = {}
	for file in glob.glob(directory + "/*.mid"):
		midi = None
		try:
			midi = converter.parse(file)
		except Exception as e:
			print('Could not parse file: {}'.format(file))
			continue

		print("Parsing %s" % file)
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

	enumerated_notes = {}
	num = 0
	# get all of the unique combinations of notes and chords
	print('Getting all unique notes')
	for s in songs:
		for inst in songs[s]:
			notes = songs[s][inst].notesAndRests.activeElementList
			for i in range(1, len(notes)):
				e_complete = get_all_in_offset(notes, i)
				if e_complete not in enumerated_notes:
					enumerated_notes[e_complete] = num
					num += 1
	print('done')


	# Go through the songs and build [] for each instrument
	# {song: {piano:[], bass: [], sax:[]}}

	for s in songs:
		for inst in songs[s]:

	#pickle.dump(notes, open('pickle/' + directory + '_notes', 'wb'))

	return notes

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
			e_chord.append(n.commonName)
		else:
			e_note.append(n.name)
		j += 1

	return stringify_notes(sort_notes(e_note)) + stringify_chords(sort_chords(e_chord))


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