from music21 import note, chord

def sort_notes(l):
	return list(set(sorted(l)))

def stringify_notes(l):
	return ','.join(l)

def sort_chords(l):
	return list(set(sorted(l)))

def stringify_chords(l):
	return ','.join(l)

def chord_to_notes(n):
	c_temp = []
	for c in n.pitches:
		c_temp.append(c.nameWithOctave)
	return c_temp 

def notes_to_chord(l):
	return chord.Chord(l)
