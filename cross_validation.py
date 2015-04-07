import sys

if( len(sys.argv) != 7 ) :
    print '    cross_validation.py <layer> <# of neuron in each layer> <batch_size> <iteration> <learning_rate> <K-fold>'
    print 'ex. cross_validation 3 5,6 10 20 0.01 10'
    quit()

import numpy as np
from forward import *           # init(), forward()
from calculate_error import *   # read_label_map(), calculate_error()
from backpro import *           # backpropagation()
from update import *            # update(), save_model()
from predict import *           # load_model(), create_sol_map()

TEST_NUM = 10000;

# K-FOLD CROSS-VALIDATION 

def cut_file(orig_data_path, K):
    with open(orig_data_path, 'r') as f:
        orig_data = f.readlines()
    num_lines = sum(1 for line in orig_data)
    for k in range(K):
        #test_cnt = 0;
        train_outfile = open('train_data_'+str(k+1), 'w')
        test_outfile = open('test_data_'+str(k+1), 'w')
        for idx in range(num_lines):
            if idx%K==k:
                test_outfile.write(orig_data[idx])
            else:
                #if test_cnt < TEST_NUM:
                train_outfile.write(orig_data[idx])
                #    test_cnt += 1
def score(speech_ids, y, label_map):
    tp = 0
    fp = 0
    fn = 0
    
def test(test_file, w_and_b):
    predict_speech_ids, predict_features = read_file(test_file)
    speech_ids, features, a_list, z_list = forward(predict_features, predict_speech_ids, w_and_b, len(predict_speech_ids), True)
    #predict_speech_id, predict_y_labels = predict(cv_predict_feature_file, w_and_b, sol_map)
    predict_y_labels = [a[-1] for a in a_list]
    valid_answer, predict_answer = get_answer(phonemes, speech_ids, predict_y_labels, label_map, sol_map)
    print_fscore(valid_answer, predict_answer) 


# Parameter setting
raw = 39
phonemes = 48
layer = int(sys.argv[1])
neuron = []
neuron.append(raw)
for l in range(layer-1):
    neuron.append(int(sys.argv[2].split(',')[l]))
neuron.append(phonemes)
batch_size = int(sys.argv[3])
iteration = int(sys.argv[4])
learning_rate = float(sys.argv[5])
K = int(sys.argv[6])

# Path setting
train_label_file = 'MLDS_HW1_RELEASE_v1/label/train.lab'
map_48_39_file = 'MLDS_HW1_RELEASE_v1/phones/48_39.map'
features_file = 'MLDS_HW1_RELEASE_v1/mfcc/train.ark'

print 'Start training models with', K, '-fold cross validation...'
w_and_b = init(layer, neuron)
# test model_9901
#load_model_path = 'model_9901.npy'
#w_and_b = np.load(load_model_path)
label_map = read_label_map(train_label_file, map_48_39_file)
sol_map = create_sol_map(map_48_39_file, phonemes)

cut_file(features_file, K)
for k in range(1, K+1):
    print k, '-fold'
    cv_train_feature_file = 'train_data_' + str(k)
    cv_predict_feature_file = 'test_data_' + str(k)
    for i in range(iteration):
        print 'iteration', i
        cv_train_speech_ids, cv_train_features = read_file(cv_train_feature_file)
        speech_ids, features, a_list, z_list = forward(cv_train_features, cv_train_speech_ids, w_and_b, batch_size, False)
        y_list = [a[-1] for a in a_list]
        #err, gradC = calculate_error(phonemes, speech_ids, y_list, label_map, error_func_norm2)
        err, gradC = calculate_error(phonemes, speech_ids, y_list, label_map, error_func_cross_entropy)
        print 'err:', err
        C = backpropagate(gradC, z_list, a_list, w_and_b, features, batch_size)
        w_and_b = update(learning_rate, w_and_b[0], w_and_b[1], C, i)
        if i % 50 == 0 and i > 0:
            test(cv_predict_feature_file, w_and_b)
            #test(cv_train_feature_file, w_and_b)
    print '------------------------------------'
    # cv_predict_speech_ids, cv_predict_features = read_file(cv_predict_feature_file)
    # speech_ids, features, a_list, z_list = forward(cv_predict_features, cv_predict_speech_ids, w_and_b, len(cv_predict_speech_ids), True)
    # predict_y_labels = [a[-1] for a in a_list]
    # valid_answer, predict_answer = get_answer(phonemes, speech_ids, predict_y_labels, label_map, sol_map)
    # print_fscore(valid_answer, predict_answer) 

