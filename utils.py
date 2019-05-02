from music21 import note, chord

CHORD_NAME = ' chord '

def sort_notes(l):
	return list(set(sorted(l)))

def stringify_notes(l):
	return ','.join(l)

def sort_chords(l):
	return list(set(sorted(l)))

def stringify_chords(l):
	return CHORD_NAME + ','.join(l)

def break_notes_and_chords(s):
	i = s.find(CHORD_NAME)
	notes = s[:i].split(',')
	chords = '' if i == -1 else s[i+len(CHORD_NAME):].split(',')
	return notes, chords

def chord_to_notes(n):
	c_temp = []
	for c in n.pitches:
		c_temp.append(c.nameWithOctave)
	return c_temp 

def notes_to_chord(l):
	return chord.Chord(l)
