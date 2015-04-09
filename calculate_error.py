import theano.tensor as T
from theano import function
import numpy as np
import sys

def calculate_error(phoneme_num, label_48_list, vy_48_list, labelmap, error_func, namda, n, WList):
	batch_size = len(label_48_list)
	
	err_total = 0
	errd_total = np.zeros(phoneme_num)
        K = 0
        for W in WList:
            for x in np.nditer(W):
                K += x*x


        regularization = (namda*K)/(2*n)
	for i in range(batch_size):
		label_idx = labelmap[label_48_list[i]]
		vlabel = np.zeros(phoneme_num)
		vlabel[label_idx] = 1

		err, errd = error_func(vlabel, vy_48_list[i])
		err_total += err
		errd_total += errd
	return np.nan_to_num(err_total / float(batch_size))+regularization,\
               np.nan_to_num(errd_total / float(batch_size))+regularization


# theano_ function for error_func_norm2
_n_x = T.dvector('x') # sigmoid(a) 
_n_y = T.dvector('y') # label [0,0,1,0,0,...,0]
_n_z = T.sqrt(T.sum((_n_x - _n_y) ** 2)) / T.shape(_n_x)[0]
_n_norm2 = function([_n_x, _n_y], _n_z)
_n_norm2d = function([_n_x, _n_y], T.grad(_n_z, _n_x))
	
def error_func_norm2(v_lab, v_pred):
	return _n_norm2(v_pred, v_lab), _n_norm2d(v_pred, v_lab)

# theano function for error_func_cross_entropy
_ce_x = T.dvector('x')
_ce_y = T.dvector('y')
_ce_rx = 1 - _ce_x;
_ce_ln_y = T.log(_ce_y)
_ce_ln_ry = T.log(1 - _ce_y)
_ce_z = (T.dot(_ce_x, _ce_ln_y) + T.dot(_ce_rx, _ce_ln_ry)) * (-1) / T.shape(_ce_x)[0];
_ce_cross_h = function([_ce_x, _ce_y], _ce_z, allow_input_downcast=True)
_ce_cross_h_grad = function([_ce_x, _ce_y], T.grad(_ce_z, _ce_y), allow_input_downcast=True)


def error_func_cross_entropy(v_lab, v_pred):
	return _ce_cross_h(v_lab, v_pred), _ce_cross_h_grad(v_lab, v_pred)

def get_answer(phonemes, speech_id, predict_y_labels, label_map, sol_map):
    valid_answer = []
    predict_answer = [] 
    for i in range(len(speech_id)):
        label_idx = label_map[speech_id[i]]
        valid_answer.append(sol_map[label_idx])
        label_idx = predict_y_labels[i].argmax(axis = 0)
        predict_answer.append(sol_map[label_idx])
    return valid_answer, predict_answer

def print_fscore(v1, v2):
    match = 0
    for n in range(len(v1)):
        if v1[n] == v2[n]:
            match += 1
    accuracy = float(match)/float(len(v1))*100.0
    print_file = '''
Accuracy:  {a} % ({match}/{total})
==========================================
    '''.format(a = accuracy, match = match, total = len(v1))
    print print_file

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
	y_48s = [np.random.random(48),
			 np.random.random(48),
			 np.random.random(48)]
	labels = ['maeb0_si1411_1',
			  'maeb0_si1411_2',
			  'maeb0_si1411_3']

	# print 'y_48:' + str(y_48)
	# print 'label_1943:' + str(label)
	# print 'labelmap:' + str(labelmap)

	print 'start'
	err, errd = calculate_error(48, labels, y_48s, labelmap, error_func_norm2)
	print err
	print errd
	print 'end'


if __name__ == '__main__':
	main() 

