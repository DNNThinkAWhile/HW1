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
      writer.writerow(['Id','Prediction'])
      for idx in range(len(Y)):
         sol = [str(speech_id[idx]), str(Y[idx])]
         writer.writerow(sol)

def predict(test_file_path, theta, sol_map, sol_48_39_map):
	speech_id, mfcc_data = read_file(test_file_path)
	y_labs = []
	selected_speech_id, feature_set, a_list, z_list = forward(mfcc_data, speech_id, theta, len(speech_id) ,len(speech_id), True)
        y_list = [a[-1] for a in a_list]
	max_idx = [y.argmax(axis = 0) for y in y_list ]
	y_labs = [ sol_48_39_map[sol_map[idx]] for idx in max_idx ]
		# if speech_id != selected_speech_id:
		#     print 'warning! debug!'
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

def create_48_39_map(map_48_39_file):
	d = dict()
	with open(map_48_39_file, 'r') as f:
		for line in f:
			toks = line.strip().split('\t')
			d[toks[0]] = toks[1]
	return d

def main():
	if ( len(sys.argv) != 2 ) :
		print 'predict <model_id>'
		quit()
	
	label_num = 48
	map_48_39_file = 'MLDS_HW1_RELEASE_v1/phones/48_39.map'
	sol_map = create_sol_map(map_48_39_file, label_num)
	sol_48_39_map = create_48_39_map(map_48_39_file)

	load_model_path = str(sys.argv[1])
	theta = load_model(load_model_path)
	test_file_path = 'MLDS_HW1_RELEASE_v1/mfcc/test.ark'
	speech_id, y_labs = predict(test_file_path, theta, sol_map, sol_48_39_map)
	predict_outfile(speech_id, y_labs)

if __name__ == '__main__':
	main()
