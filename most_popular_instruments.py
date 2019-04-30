import pickle
from music21 import *
import glob

instruments = {}

for file in glob.glob('jazz/*.mid'):
	print('getting instruments from', file)
	midi = converter.parse(file)
	a = instrument.partitionByInstrument(midi)
	if a is None:
		continue
	for i in a:
		name = i.getInstrument().instrumentName

		if name is not None:
			if name in instruments:
				instruments[name] += 1
			else:
				instruments[name] = 1

instruments = sorted(instruments.items(), key=lambda kv: kv[1], reverse=True)

pickle.dump(instruments, open('pickle/most_popular_instruments.p', 'wb'))