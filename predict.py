import numpy as np
import sys
import csv
from forward import *

def load_model(load_model_path):
   theta = np.load(load_model_path)
   return theta

def predict_outfile(speech_id, Y):
   outfile_path = 'solution.csv'
   with open(outfile_path, 'w') as csvfile:
      writer = csv.writer(csvfile)
      for idx in range(len(Y)):
         sol = [str(speech_id[idx]), str(Y[idx])]
         writer.writerow(sol)

def predict(test_file_path, theta, sol_map):
	print 'prediting...'
	speech_id, mfcc_data = read_file(test_file_path)
	y_labs = []
	l_progress = 0
	for i in range(len(speech_id)):
		selected_speech_id, feature_set, a_list, z_list = forward([mfcc_data[i]], [speech_id[i]], theta, 1, True)
		y = a_list[0][-1]
		max_idx = y.argmax(axis = 0)
		y_labs.append(sol_map[max_idx])
		# if speech_id != selected_speech_id:
		#     print 'warning! debug!'
		progress = i * 100 / len(speech_id)
		if progress - l_progress >= 10:
			l_progress = progress
			print str(i) + '(' + str(progress) + '%)'
	print 'predict done'
	return speech_id, y_labs

def create_sol_map(map_48_39_file, labelnum):
	print 'creating solution map with ' + str(labelnum) + ' labels'
	count = 0
	sol_map = [None] * labelnum
	with open(map_48_39_file, 'r') as f:
		for line in f:
			label = line.strip().split('\t')[0]
			sol_map[count] = label
			count += 1
	return sol_map

def main():
	if ( len(sys.argv) != 2 ) :
		print 'predict <model_id>'
		quit()
	
	label_num = 48
	map_48_39_file = 'MLDS_HW1_RELEASE_v1/phones/48_39.map'
	sol_map = create_sol_map(map_48_39_file, label_num)

	load_model_path = 'model_' + str(sys.argv[1]) + '.npy'
	theta = load_model(load_model_path)
	test_file_path = 'MLDS_HW1_RELEASE_v1/mfcc/test.ark'
	speech_id, y_labs = predict(test_file_path, theta, sol_map)
	predict_outfile(speech_id, y_labs)

if __name__ == '__main__':
	main()
