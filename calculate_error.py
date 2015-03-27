import theano.tensor as T
from theano import function
import numpy as np

def calculate_error(error_func, label_48, vy_48):
	vlabel = np.zeros(48)
	vlabel[label_48] = 1
	err, errd = error_func(vlabel, vy_48)
	return err, errd

def error_func_norm2(v1, v2):
	x = T.fvector('x')
	z = T.sqrt(T.sum(T.sqr(x)))
	norm2 = function([x], z)
	norm2d = function([x], T.grad(z, x))
	
	d = abs(v2 - v1)
	df = np.array(d, dtype = 'f')
	return norm2(df), norm2d(df)

def read_label_map(file_label, file_48_39):
	d_1943_index = dict()
	d_48_index = dict();
	count = 0

	# map phonemes(sil, aa,..) to their indices(int), 0:47
	with open(file_48_39, 'r') as f:
		for line in f:
			tokens = line.strip().split('\t')
			d_48_index[tokens[0]] = count
			count += 1

	# map labels(like "maeb0_si1411_1") to their phonemes' indices, 1943->48
	with open(file_label, 'r') as f:
		for line in f:
			tokens = line.strip().split(',')
			d_1943_index[tokens[0]] = d_48_index[tokens[1]]
			
	return d_1943_index

def main():
	print 'test calculate_error'

	labelmap = read_label_map('train.lab', '48_39.map')
	y_48 = np.random.random(48)
	label = 'maeb0_si1411_3'

	# print 'y_48:' + str(y_48)
	# print 'label_1943:' + str(label)
	# print 'labelmap:' + str(labelmap)

	print 'start'
	err, errd = calculate_error(error_func_norm2, labelmap[label], y_48)
	print err
	print errd
	print 'end'


if __name__ == '__main__':
	main() 

