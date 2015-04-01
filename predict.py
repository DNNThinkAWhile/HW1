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
            sol = str(speech_id[idx]) + ',' + str(Y[idx])
            writer.writerow(sol)

def predict(test_file_path, theta):
    speech_id, mfcc_data = read_file(test_file_path)
    y_list = [];
    for i in range(len(speech_id)):
        selected_speech_id, feature_set, a_list, z_list = forward(mfcc_data[i], speech_id[i], theta, 1, True)
        y = a_list[0][-1]
        y_list.append(y)
        # if speech_id != selected_speech_id:
        #     print 'warning! debug!'
    return speech_id, y_list

def main():
    if ( len(sys.argv) != 2 ) :
        print 'predict <model_id>'
        quit()
    
    load_model_path = 'model_' + str(sys.argv[1]) + '.npy'
    theta = load_model(load_model_path)
    test_file_path = 'MLDS_HW1_RELEASE_v1/mfcc/test.ark'
    speech_id, Y = predict(test_file_path, theta)
    predict_outfile(speech_id, Y)

if __name__ == '__main__':
    main()
