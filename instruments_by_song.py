import pickle
from music21 import *
import glob

instruments_by_song = []

for file in glob.glob('jazz/*.mid'):
	print('getting instruments from', file)
	midi = converter.parse(file)
	a = instrument.partitionByInstrument(midi)
	if a is None:
		continue

	current_instruments = []

	for i in a:
		name = i.getInstrument().instrumentName

		if name is not None:
			current_instruments.append(name)

	if len(current_instruments) < 3:
		continue

	instruments_by_song.append((file, current_instruments))

pickle.dump(instruments_by_song, open('pickle/instruments_by_song.p', 'wb'))