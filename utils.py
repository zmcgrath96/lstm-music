CHORD_NAME = 'chord'

def sort_notes(l):
	return sorted(l)

def stringify_notes(l):
	return ','.join(l)

def sort_chords(l):
	return sorted(l)

def stringify_chords(l):
	return CHORD_NAME + ','.join(l)

def break_notes_and_chords(s):
	i = s.find(CHORD_NAME)
	notes = s[:i].split(',')
	chords = '' if i == -1 else s[i+len(CHORD_NAME):].split(',')
	return notes, chords
