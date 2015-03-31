import sys
from forward import *           # init(), forward()
from calculate_error import *   # read_label_map(), calculate_error()
from backpro import *          # backpropagation()
from update import *            # update(), save_model()
#from predict import *           # load_model()

if( len(sys.argv) != 6 ) :
    print '    run <layer> <# of neuron in each layer> <batch_size> <iteration> <learning_rate>'
    print 'ex. run 3 5,6,7 10 20 0.01'
    quit()


# Parameter setting
raw = 39
phonemes = 48
layer = int(sys.argv[1])
neuron = []
neuron.append(raw)
for l in range(layer):
    neuron.append(int(sys.argv[2].split(',')[l]))
neuron.append(phonemes)
batch_size = int(sys.argv[3])
iteration = int(sys.argv[4])
learning_rate = float(sys.argv[5])

# Path setting
train_label_file = 'MLDS_HW1_RELEASE_v1/label/train.lab'
map_48_39_file = 'MLDS_HW1_RELEASE_v1/phones/48_39.map'
features_file = 'MLDS_HW1_RELEASE_v1/mfcc/train.ark'

w_and_b = init(layer, neuron)
label_map = read_label_map(train_label_file, map_48_39_file)

print 'Start Training...'
for i in range(iteration):
    print 'iteration: ', i
    
    all_speech_ids, all_features = read_file(features_file)
    speech_ids, features, a_list, z_list = forward(all_features, all_speech_ids, w_and_b, batch_size, False)
    
    y_list = [a[-1] for a in a_list]
    err, gradC = calculate_error(phonemes, speech_ids, y_list, label_map, error_func_norm2)
    
    print 'err: ' + str(err)
    print 'gradC: ' + str(gradC)
    
    print a_list

    C = backpropagate(gradC, z_list, a_list, layer, w_and_b, features, batch_size)
    update(learning_rate, w_and_b[0], w_and_b[1], C, layer, i)
    print '------------------------------------'


