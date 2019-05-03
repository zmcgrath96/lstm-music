from music21 import note, chord 
import math
from utils import *
import numpy as np

# PARAMETER: diction of songs for instrument
def clean(songs, note_width):
	# get all of the unique combinations of notes and chords
	# after saving all of the notes as embeddings in the enumerated_notes dict, 
	# add all of the notes to the array for each song for each of the instruments
	print('Getting encodings')
	start_name = 'start'
	end_name = 'end'
	enumerated_notes = {}
	rest_name = note.Rest().name
	enumerated_notes[rest_name] = 0
	enumerated_notes[start_name] = 1
	enumerated_notes[end_name] = 2
	num = 3
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
			embedded[s_count][i_count] = [enumerated_notes[rest_name]  for x in range(scaled_s_l)]
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
			embedded[s_count][i_count] = [enumerated_notes[start_name]] * 100 + embedded[s_count][i_count] + [enumerated_notes[end_name]]
			i_count += 1
		s_count += 1
	
	# normalize data. can't make a numpy matrix so we gotta do it manually
	for s in range(len(embedded)):
		for i in range(len(embedded[s])):
			for n in range(len(embedded[s][i])):
				embedded[s][i][n] /= len(enumerated_notes)

	print('done')
	return enumerated_notes, embedded




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