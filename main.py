from lstm import lstm
import sys

def main(args):
	if '-t' in args[0]:
		train()
	elif '-g' in args[0]:
		generate()
	pass

def train():
	pass

def generate():
	pass

if __name__ == '__main__':
	main(sys.argv[1:])