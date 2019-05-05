from lstm import musicLSTM
import numpy as np
import sys
import pickle
import os
import datetime

def main(args):
	# determine which architecture to use
	arch = [s for s in args if '-arch=' in s]
	if len(arch) is not 1:
		print('Error: only one -arch= parameter is allowed')
		sys.exit(1)
	arch = int(arch[0].split('=')[1])
	if arch > 3 or arch < 1:
		print('Error: arch must be 1, 2, or 3')
		sys.exit(1)

	if '-t=piano' in args:
		
		# get inputs and outputs
		piano_in, piano_out = get_input_and_output('piano', arch)
		print(piano_in.shape)
		# get shapes for input and output
		in_shape = (piano_in.shape[1], piano_in.shape[2])
		out_shape = piano_out.shape[1]

		# train 
		path = 'models/architecture{}'.format(arch)
		name = '/piano-lstm'
		if not os.path.exists(path):
			os.makedirs(path)
		lstm = musicLSTM(in_shape, out_shape)
		lstm.train(piano_in, piano_out, path + name, it=20, batch=64)

	elif '-t=bass' in args:
		# get inputs and outputs
		bass_in, bass_out = get_input_and_output('bass', arch)

		path = 'models/architecture{}'.format(arch)
		name = '/bass-lstm'
		if not os.path.exists(path):
				os.makedirs(path)

		if arch is not 1:
			
			# get shapes for input and output
			in_shape = (bass_in.shape[1], bass_in.shape[2])
			out_shape = bass_out.shape[1]

			# train 
			lstm = musicLSTM(in_shape, out_shape)
			lstm.train(bass_in, bass_out, path + name, it=20, batch=64)
		else:
			bass_dist = create_prob_dict(bass_in, bass_out)
			np.save(path + '/bass-dist', bass_dist)
		

	elif '-t=sax' in args:
		
		# get inputs and outputs
		sax_in, sax_out = get_input_and_output('sax', arch)
		path = 'models/architecture{}'.format(arch)
		name = '/sax-lstm'
		if not os.path.exists(path):
			os.makedirs(path)
		
		if arch is not 1:
			
			# get shapes for input and output
			in_shape = (sax_in.shape[1], sax_in.shape[2])
			out_shape = sax_out.shape[1]
			lstm = musicLSTM(in_shape, out_shape)
			lstm.train(sax_in, sax_out, path + name, it=20, batch=64)
		else:
			sax_dist = create_prob_dict(sax_in, sax_out)
			np.save(path + '/sax-dist', sax_dist)

	elif '-g' in args:
		# get the desired song length
		song_length = [s for s in args if '-len=' in s]
		if len(song_length) is 0:
			song_length = 100
		elif len(song_length) > 1:
			print('Error: only one -len= parameter is allowed')
			sys.exit(1)
		else:
			song_length = int(song_length[0].split('=')[1])

		if arch is 1:
			song = generate_arch_1(song_length)
		elif arch is 2:
			song = generate_arch_2(song_length)
		elif arch is 3:
			song = generate_arch_3(song_length)

		now = datetime.datetime.now()
		from create_song import create_song
		if not os.path.exists('songs/architecture{}'.format(arch)):
			os.makedirs('songs/architecture{}'.format(arch))
		create_song(song, 'songs/architecture{}/output-arch{}-{}{}{}-{}{}{}.mid'.format(arch, arch, now.year, now.month, now.day, now.hour, now.minute, now.second))

def get_input_and_output(inst, arch):
	# load embeddings and encodings
	arch_path = 'pickle/architecture{}/{}_'.format(arch, inst)
	inst_input = pickle.load(open(arch_path + 'inputs', 'rb'))
	inst_output = pickle.load(open(arch_path + 'outputs', 'rb'))
	if inst is 'piano' or (inst is not 'piano' and arch is not 1):
		inst_input = np.array(inst_input).reshape((len(inst_input), 100, 1))
		inst_output = np.array(inst_output).reshape((len(inst_output), len(inst_output[0])))
	else:
		inst_input = (np.array(inst_input) * len(inst_output[0])).astype(int)
	return inst_input, inst_output

def create_prob_dict(inst_input, inst_output):

	# create empty distribution
	prob_dist = np.zeros((len(inst_output[0]), len(inst_output[0])))

	# create an array of outputs without one hot
	output_max = np.argmax(np.array(inst_output), axis=1)

	# iterate through in and out and increment probabilty
	for in_note, out_note in zip(inst_input, output_max):
		prob_dist[in_note, out_note] += 1
	
	# sum each row of distribution
	sums = np.sum(prob_dist, axis=1)

	# normalize the distributions
	for i in range(len(prob_dist)):
		if sums[i] > 0.0:
			prob_dist[i,:] = prob_dist[i,:] / sums[i]
		else:
			prob_dist[i,:] = prob_dist[i,:]  * 0
	return prob_dist

def generate_arch_1(length):
	
	# create input and fill with 1's
	piano_input = np.full((1, 100, 1), 1)
	
	# load models and distributions
	piano_model_path, _ = get_last_model_path('piano', 1)
	bass_dist = np.load('models/architecture1/bass-dist.npy')
	sax_dist = np.load('models/architecture1/sax-dist.npy')
	piano_lstm = musicLSTM(filepath=piano_model_path)
	num_classes = len(bass_dist)

	# set up outputs
	piano_out = []
	bass_out = []
	sax_out = []

	# for _ in range(50):
	# 	p_out = np.random.choice(num_classes, p=piano_lstm.predict(piano_input)[0])
	# 	if p_out is 2:
	# 		continue
	# 	piano_out.append(p_out)
	# 	b_out = np.random.choice(num_classes, p=bass_dist[p_out])
	# 	bass_out.append(b_out)
	# 	s_out = np.random.choice(num_classes, p=sax_dist[b_out])
	# 	sax_out.append(s_out)
	# 	piano_input = np.roll(piano_input, -1, axis=1)
	# 	piano_input[-1] = p_out


	for _ in range(length * 4):
		p_out = np.random.choice(num_classes, p=piano_lstm.predict(piano_input)[0])
		piano_out.append(p_out)
		b_out = np.random.choice(num_classes, p=bass_dist[p_out])
		bass_out.append(b_out)
		s_out = np.random.choice(num_classes, p=sax_dist[b_out])
		sax_out.append(s_out)
		piano_input = np.roll(piano_input, -1, axis=1)
		piano_input[-1] = p_out
		if p_out is 2:
			break

	return [piano_out, bass_out, sax_out]
	

def generate_arch_2(length):
	# create input and fill with 1's
	piano_input = np.array((1, 100, 1)).fill(1)
	bass_input = np.array((1, 100, 1)).fill(1)
	
	# get paths to most recent models
	piano_model_path, _ = get_last_model_path('piano', 2)
	bass_model_path, _ = get_last_model_path('bass', 2)
	sax_model_path, _ = get_last_model_path('sax', 2)
	bass_dist = np.load('models/architecture1/bass-dist.npy')
	num_classes = len(bass_dist)

	# load models and distributions
	piano_lstm = musicLSTM(filepath=piano_model_path)
	bass_lstm = musicLSTM(filepath=bass_model_path)
	sax_lstm = musicLSTM(filepath=sax_model_path)

	# set up outputs
	piano_out = []
	bass_out = []
	sax_out = []

	# generate output
	for _ in range(50):
		p_out = np.random.choice(num_classes, p=piano_lstm.predict(piano_input)[0])
		piano_out.append(p_out)
		b_out = np.argmax(bass_lstm.predict(piano_input)[0])
		bass_out.append(b_out)
		s_out = np.argmax(sax_lstm.predict(bass_input)[0])
		sax_out.append(s_out)
		piano_input = np.roll(piano_input, -1, axis=1)
		piano_input[-1] = p_out
		bass_input = np.roll(bass_input, -1, axis=1)
		bass_input[-1] = b_out
	
	for _ in range(50, length * 4):
		p_out = np.argmax(piano_lstm.predict(piano_input)[0])
		piano_out.append(p_out)
		b_out = np.argmax(bass_lstm.predict(piano_input)[0])
		bass_out.append(b_out)
		s_out = np.argmax(sax_lstm.predict(bass_input)[0])
		sax_out.append(s_out)
		piano_input = np.roll(piano_input, -1, axis=1)
		piano_input[-1] = p_out
		bass_input = np.roll(bass_input, -1, axis=1)
		bass_input[-1] = b_out
		if p_out is 2:
			break

	return [piano_out, bass_out, sax_out]

def generate_arch_3(length):
	# create input and fill with 1's
	piano_input = np.array((1, 100, 1)).fill(1)

	# get paths to most recent models
	piano_model_path, _ = get_last_model_path('piano', 3)
	bass_model_path, _ = get_last_model_path('bass', 3)
	sax_model_path, _ = get_last_model_path('sax', 3)

	# load models and distributions
	piano_lstm = musicLSTM(filepath=piano_model_path)
	bass_lstm = musicLSTM(filepath=bass_model_path)
	sax_lstm = musicLSTM(filepath=sax_model_path)
	bass_dist = np.load('models/architecture1/bass-dist.npy')
	num_classes = len(bass_dist)

	# set up outputs
	piano_out = []
	bass_out = []
	sax_out = []

	# generate output
	for _ in range(50):
		p_out = np.random.choice(num_classes, p=piano_lstm.predict(piano_input)[0])
		piano_out.append(p_out)
		b_out = np.argmax(bass_lstm.predict(piano_input)[0])
		bass_out.append(b_out)
		s_out = np.argmax(sax_lstm.predict(piano_input)[0])
		sax_out.append(s_out)
		piano_input = np.roll(piano_input, -1, axis=1)
		piano_input[-1] = p_out

	for _ in range(50, length * 4):
		p_out = np.argmax(piano_lstm.predict(piano_input)[0])
		piano_out.append(p_out)
		b_out = np.argmax(bass_lstm.predict(piano_input)[0])
		bass_out.append(b_out)
		s_out = np.argmax(sax_lstm.predict(piano_input)[0])
		sax_out.append(s_out)
		piano_input = np.roll(piano_input, -1, axis=1)
		piano_input[-1] = p_out
		if p_out is 2:
			break

	return [piano_out, bass_out, sax_out]


def get_last_model_path(inst, arch):
	path = ''
	num = -1
	for i in os.listdir('models/architecture{}'.format(arch)):
		if i.startswith(inst + '-lstm'):
			curr_num = int(i.split('-')[2].split('.')[0])
			if curr_num > num:
				num = curr_num
				path = i
	if path is '':
		return None, None
	else:
		return 'models/architecture{}/'.format(arch) + path , num
	

if __name__ == '__main__':
	main(sys.argv[1:])