from music21 import note

REST_NAME = note.Rest().name

def sort_notes(l):
	return list(set(sorted(l)))

def stringify_notes(l):
	return ','.join(l)

def remove_rest(l):
	l.remove(REST_NAME)
	return l

