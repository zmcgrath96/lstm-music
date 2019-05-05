# Modified from https://towardsdatascience.com/how-to-generate-music-using-a-lstm-neural-network-in-keras-68786834d4c5
# Modified by Austin Irvine, ...
# Date: May 2019
# Purpose: Place notes & chords into midi output or song format to export to a midi-player(song creation tool)
from music21 import converter, instrument, note, chord, stream
import pickle

encodings = pickle.load(open('pickle/clean_jazz_encodings', 'rb'))

# get the note string from corresponding encoded int
def get_note(index):
	return list(encodings.keys())[list(encodings.values()).index(index)]

# create note and chord objects based on the values generated by the model
def create_part(prediction_output, instrumentType):
	
	offset = 0
	output_notes = []

	for pattern in prediction_output:

		# pattern is invalid (start or end symbol)
		if pattern == 1 or pattern == 2:
			continue
		
		# pattern is a rest
		if pattern == 0:
			new_note = note.Rest()
		else:
			pattern = get_note(pattern)		
			
			# pattern is a chord
			if (',' in pattern) or pattern.isdigit():
				notes_in_chord = pattern.split(',')
				new_note = chord.Chord(notes_in_chord)

			# pattern is a note
			else:
				new_note = note.Note(pattern)

		# assign instrument
		if instrumentType == 'piano':
			new_note.storedInstrument = instrument.Piano()
		elif instrumentType == 'bass':
			new_note.storedInstrument = instrument.Bass()
		else:
			new_note.storedInstrument = instrument.Saxophone()

		#assign note duration
		new_note.duration.quarterLength = .25

		# assign and update offset
		new_note.offset = offset
		offset += .25

		# add note to part
		output_notes.append(new_note)

	return output_notes
	
def create_song(prediction_output, output_file_path):
	piano_part = stream.Part(create_part(prediction_output[0], 'piano'))
	bass_part = stream.Part(create_part(prediction_output[1], 'bass'))
	sax_part = stream.Part(create_part(prediction_output[2], 'sax'))

	piano_part.insert(0, instrument.Piano())
	bass_part.insert(0, instrument.Bass())
	sax_part.insert(0, instrument.Saxophone())

	song = [piano_part, bass_part, sax_part]

	midi_stream = stream.Stream(song)
	midi_stream.write('midi', fp=output_file_path)