import sys

if( len(sys.argv) != 7 ) :
    print '    cross_validation.py <layer> <# of neuron in each layer> <batch_size> <epoch> <learning_rate> <K-fold>'
    print 'ex. cross_validation 3 500,600 20 5 0.1 5'
    quit()

import numpy as np
from forward import *           # init(), forward()
from calculate_error import *   # read_label_map(), calculate_error()
from backpro import *           # backpropagation()
from update import *            # update(), save_model()
from predict import *           # load_model(), create_sol_map()


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
    
def test(pred_speech_ids, pred_features, w_and_b):
    speech_ids, features, a_list, z_list = \
        forward(pred_features, pred_speech_ids, w_and_b, len(pred_speech_ids), 1, True)
    #predict_speech_id, predict_y_labels = predict(cv_predict_feature_file, w_and_b, sol_map)
    predict_y_labels = [a[-1] for a in a_list]
    valid_answer, predict_answer = get_answer(phonemes, speech_ids, predict_y_labels, label_map, sol_map)
    print_fscore(valid_answer, predict_answer) 


# Parameter setting
raw = 39*9
phonemes = 48
layer = int(sys.argv[1])
neuron = []
neuron.append(raw)
for l in range(layer-1):
    neuron.append(int(sys.argv[2].split(',')[l]))
neuron.append(phonemes)
batch_size = int(sys.argv[3])
max_epoch = int(sys.argv[4]) # how many epochs we want to run for
learning_rate = float(sys.argv[5])
K = int(sys.argv[6])
train_size = 0
iterations_epoch = 0


# Path setting
train_label_file = 'MLDS_HW1_RELEASE_v1/label/train.lab'
map_48_39_file = 'MLDS_HW1_RELEASE_v1/phones/48_39.map'
features_file = 'MLDS_HW1_RELEASE_v1/mfcc/train.normalize.ark'

print 'Start training models with', K, '-fold cross validation...'
w_and_b = init(layer, neuron)
label_map = read_label_map(train_label_file, map_48_39_file)
sol_map = create_sol_map(map_48_39_file, phonemes)

cut_file(features_file, K)
for k in range(1, K+1):
    print k, '-fold'
    cv_train_feature_file = 'train_data_' + str(k)
    cv_predict_feature_file = 'test_data_' + str(k)
    cv_train_speech_ids, cv_train_features = read_file(cv_train_feature_file)
    cv_train_speech_ids, cv_train_features = shuffle(cv_train_speech_ids, cv_train_features)
    cv_predict_speech_ids, cv_predict_features = read_file(cv_predict_feature_file)


    train_size = len(cv_train_speech_ids)
    iterations_epoch = train_size / batch_size
    for epoch in range(max_epoch):
        print 'epoch ', epoch

        for i in range(iterations_epoch):
            print 'iteration', i
            
            speech_ids, features, a_list, z_list = \
                forward(cv_train_features, cv_train_speech_ids, w_and_b, batch_size, i, False)
            
            y_list = [a[-1] for a in a_list]
            err, gradC = calculate_error(phonemes, speech_ids, y_list, label_map, error_func_cross_entropy)
            
            print 'err:', err

            C = backpropagate(gradC, z_list, a_list, w_and_b, features, batch_size)
            w_and_b = update(learning_rate, w_and_b[0], w_and_b[1], C)
            
            if i % 1000 == 0 and i > 0:
                print 'current model saved'
                save_model(w_and_b, epoch, i)
                print 'start predicting'
                test(cv_predict_speech_ids, cv_predict_features, w_and_b)

    print '------------------------------------'
    # cv_predict_speech_ids, cv_predict_features = read_file(cv_predict_feature_file)
    # speech_ids, features, a_list, z_list = forward(cv_predict_features, cv_predict_speech_ids, w_and_b, len(cv_predict_speech_ids), True)
    # predict_y_labels = [a[-1] for a in a_list]
    # valid_answer, predict_answer = get_answer(phonemes, speech_ids, predict_y_labels, label_map, sol_map)
    # print_fscore(valid_answer, predict_answer) 

