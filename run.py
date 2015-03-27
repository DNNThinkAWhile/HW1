import sys
#from forward import *           # init(), forward()
#from calculate_error import *   # read_label_map(), calculate_error()
#from backprop import *          # backpropagation()
#from update import *            # update(), save_model()
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
for layer in range(layer):
    neuron.append(int(sys.argv[2].split(',')[layer]))
neuron.append(phonemes)
batch_size = int(sys.argv[3])
iteration = int(sys.argv[4])
learning_rate = int(sys.argv[5])

# Path setting
train_label_file = 'MLDS_HW1_RELEASE_v1/label/train.lab'
48_39_map_file = 'MLDS_HW1_RELEASE_v1/phones/48_39.map'

W, B = init(layer, neuron)
print 'Start Training...'
for i in range(iteration):
    print 'iteration: ', i
    speech_id, Y = forward(W, B, layer, batch_size)
    label_map = read_label_map(train_label_file, 48_39_map_file)
    err, gradC = calculate_error(phonemes, speech_id, Y, label_map, error_func_norm2)
    print 'err: '
    C = backprop(gradC, layer)
    update(learning_rate, C, layer, i)
    print '------------------------------------'


