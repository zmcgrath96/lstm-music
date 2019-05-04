from music21 import note, chord 
import math
from utils import *
import numpy as np
# PARAMETER: diction of songs for instrument
def clean(songs, song_lengths, note_width):
	# get all of the unique combinations of notes and chords
	# after saving all of the notes as embeddings in the enumerated_notes dict, 
	# add all of the notes to the array for each song for each of the instruments
	print('Getting encodings')

	# dictionary of encodings
	rest_name = note.Rest().name
	start_name = 'start'
	end_name = 'end'
	enumerated_notes = {}
	enumerated_notes[rest_name] = 0
	enumerated_notes[start_name] = 1
	enumerated_notes[end_name] = 2
	num = 3
	
	# fill arrays of length of songs with note names
	encoded_step_1 = [[] for _ in range(len(songs))]
	s_index = 0
	for s_key in songs:
		encoded_step_1[s_index] = [[] for _ in range(len(songs[s_key]))]
		song = songs[s_key]
		i_index = 0
		for i_key in song:
			encoded_step_1[s_index][i_index] = [[rest_name] for i in range(math.ceil(song_lengths[s_index] / note_width))]
			inst = song[i_key].notesAndRests.activeElementList
			for n in range(1, len(inst)):
				if not (isinstance(inst[n], note.Note) or isinstance(inst[n], chord.Chord)):
					continue
				start = math.ceil(inst[n].offset / note_width)
				end = start + math.ceil(inst[n].duration.quarterLength / note_width)
				for i in range(start, end):
					if i >= len(encoded_step_1[s_index][i_index]):
						break
					to_append = rest_name
					if isinstance(inst[n], note.Note):
						to_append = inst[n].nameWithOctave
						encoded_step_1[s_index][i_index][i].append(to_append)
					elif isinstance(inst[n], chord.Chord):
						to_append = [x.nameWithOctave for x in inst[n].pitches]
						encoded_step_1[s_index][i_index][i] += to_append
			i_index += 1
		s_index += 1

	# go through the array of that = [songs][instruments][notes] and make encoded[that] = encoded version of encoded_step_1[that]
	encoded = [[] for _ in range(len(encoded_step_1))]
	for s in range(len(encoded_step_1)):
		encoded[s] = [[] for _ in range(len(encoded_step_1[s]))]
		for i in range(len(encoded_step_1[s])):
			encoded[s][i] = [0 for _ in range(len(encoded_step_1[s][i]))]
			for n in range(len(encoded[s][i])):
				clean_note = stringify_notes(sort_notes(remove_rest(encoded_step_1[s][i][n]))) if len(encoded_step_1[s][i][n]) > 1 else encoded_step_1[s][i][n][0]
				if clean_note not in enumerated_notes:
					enumerated_notes[clean_note] = num
					num += 1
				encoded[s][i][n] = enumerated_notes[clean_note] 

	# normalize all of the data
	start_list  = [enumerated_notes[start_name]] * 100
	num_unique_notes = len(enumerated_notes)
	for s in range(len(encoded)):
		for i in range(len(encoded[s])):
			encoded[s][i] = start_list + encoded[s][i] + [enumerated_notes[end_name]]
			for n in range(len(encoded[s][i])):
				encoded[s][i][n] /= num_unique_notes


	print('done')
	return enumerated_notes, encoded
