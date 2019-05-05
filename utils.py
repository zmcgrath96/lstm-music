from music21 import note

REST_NAME = note.Rest().name

def pick_last(l):
	return l[-1]

def stringify_notes(l):
	return ','.join(l)

def remove_rest(l):
	l.remove(REST_NAME)
	return l

