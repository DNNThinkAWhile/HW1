import theano.tensor as T
from theano import function
import numpy as np

def calculate_error(phoneme_num, label_48_list, vy_48_list, labelmap, error_func):
	batch_size = len(label_48_list)
	
	err_total = 0
	errd_total = np.zeros(phoneme_num)
	for i in range(batch_size):
		label_idx = labelmap[label_48_list[i]]
		vlabel = np.zeros(phoneme_num)
		vlabel[label_idx] = 1

		err, errd = error_func(vlabel, vy_48_list[i])
		err_total += err
		errd_total += errd
	return (err_total / float(batch_size)), (errd_total / float(batch_size))

def error_func_norm2(v1, v2):
	x = T.fvector('x')
	z = T.sqrt(T.sum(T.sqr(x)))
	norm2 = function([x], z)
	norm2d = function([x], T.grad(z, x))
	
	d = abs(v2 - v1)
	df = np.array(d, dtype = 'f')
	return norm2(df), norm2d(df)

def error_func_cross_entropy(v_lab, v_pred):
	x = T.fvector('x')
	y = T.fvector('y')
	rx = 1 - x;
	ln_y = T.log(y)
	ln_ry = T.log(1 - y)
	z = (T.dot(x, ln_y) + T.dot(rx, ln_ry)) * (-1) / T.shape(x)[0];
	cross_h = function([x, y], z, allow_input_downcast=True)
	cross_h_grad = function([x, y], T.grad(z, y), allow_input_downcast=True)

	return cross_h(v_lab, v_pred), cross_h_grad(v_lab, v_pred)

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
    accuracy = match/len(v1)*100.0
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

