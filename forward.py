import numpy as np
import theano.tensor as T
from theano import function
import random
import math

MFCC_trainFilePath = '/home/rason/MLDS_HW1_RELEASE_v1/mfcc/test'

w_Min = -1.0
w_Max = 1.0

###            set theano matrix dot function            ### 

p = T.dmatrix('p')
q = T.dvector('q')
r = T.dvector('r')
s = T.dot(p,q) + r 
f_matrix_dot = function( [p,q,r] , s)

x = T.dvector('x')
z = 1/(T.exp((-1)*x)+1)
sigmoid = function([x],z)


#return (W,B)
def init(layer,neuron):
   	w_List = []
   	b_List = []
   
   	LAYER_NUM = layer;
   	MFCC_DIM = neuron[0]
   	NEURON_NUM = neuron[1:-1]
   	PHONE_NUM = neuron[-1]
   # print 'layer:' + str(layer)
   # print 'neuron:' + str(neuron)
   # print 'MFCC_DIM:' + str(MFCC_DIM)
   # print 'NEURON_NUM:' + str(NEURON_NUM)
   # print 'PHONE_NUM:' + str(PHONE_NUM)

	w1 = np.random.uniform(w_Min/np.sqrt(MFCC_DIM)*3,w_Max/np.sqrt(MFCC_DIM)*3,(NEURON_NUM[0],MFCC_DIM))
	b1 = np.random.uniform(w_Min/np.sqrt(MFCC_DIM)*3,w_Max/np.sqrt(MFCC_DIM)*3,(NEURON_NUM[0]))
	w_List.append(w1);
	b_List.append(b1);
	idx = 0
	for i in range(len(NEURON_NUM)):
		iNeuron = NEURON_NUM[i]
	  	if i == len(NEURON_NUM) - 1:
			w_i = np.random.uniform(w_Min/np.sqrt(iNeuron)*3,w_Max/np.sqrt(iNeuron)*3,(PHONE_NUM,iNeuron))
			b_i = np.random.uniform(w_Min/np.sqrt(iNeuron)*3,w_Max/np.sqrt(iNeuron)*3,(PHONE_NUM))
		else:
			w_i = np.random.uniform(w_Min/np.sqrt(NEURON_NUM[idx])*3,w_Max/np.sqrt(NEURON_NUM[idx])*3,(NEURON_NUM[idx+1],NEURON_NUM[idx]))
	 		b_i = np.random.uniform(w_Min/np.sqrt(NEURON_NUM[idx])*3,w_Max/np.sqrt(NEURON_NUM[idx])*3,(NEURON_NUM[idx+1]))
		w_List.append(w_i)
		b_List.append(b_i)
	 	idx += 1
   
	rList = []
	rList.append(w_List);
	rList.append(b_List);
   
   	return rList


def read_file(filePath):
   	List_speakID = []
   	List2D_MFCC_data = []
   
   	with open(filePath, 'r') as f:
		MFCC_trainFile = f.readlines()
		num_lines = sum(1 for line in MFCC_trainFile)
		for line in MFCC_trainFile:
			partition = line.split()
			List_speakID.append(partition[0])
			List2D_MFCC_data.append(np.asfarray(partition[1:]))

   	f.close()
   
  	return (List_speakID, List2D_MFCC_data)


# shuffle the samples
def shuffle(speech_ids, features):
	length = len(speech_ids)
	idx = [i for i in range(length)]
	random.shuffle(idx)
	new_speech_ids = [speech_ids[i] for i in idx]
	new_features = [features[i] for i in idx]
	return new_speech_ids, new_features


### get Count of "BATCH_SIZE" data to train for one time ### 

def forward(List2D_MFCC_data, List_speakID, WandB, BATCH_SIZE, iteration, isTest):
	w_List = WandB[0]
	b_List = WandB[1]
	a_List = []
	z_List = []
	#   y_List = []
	LAYER_NUM = len(w_List)

	trainDataSetNum = len(List2D_MFCC_data) 
	OneTime_train_set = []
	OneTime_train_speechID = []
	if isTest == 0:
		start = iteration * BATCH_SIZE
		features_part = List2D_MFCC_data[start: start + BATCH_SIZE]
		OneTime_train_set.extend(features_part)
		OneTime_train_speechID.extend(List_speakID[start: start + BATCH_SIZE])
   	else:
	  	for i in range(BATCH_SIZE):
			arr = List2D_MFCC_data[i]
			OneTime_train_set.append(arr)
			OneTime_train_speechID.append(List_speakID[i])

   	for array in OneTime_train_set:
		temp_a_List = []
		temp_z_List = []
		z_layer = array
	  	for layer in range(0,LAYER_NUM,1):
			#print str(w_List[layer].shape) + str(z_layer.shape) + str(b_List[layer].shape)
			z_layer = f_matrix_dot( w_List[layer] , z_layer , b_List[layer])
			temp_z_List.append(z_layer)
			z_layer = sigmoid(z_layer)
			temp_a_List.append(z_layer)
		 
#      y_List.append(z_layer)
		a_List.append(temp_a_List)
		z_List.append(temp_z_List)      
   
#   print(len(a_List))
#   print(len(z_List))
#   print(len(y_List))
#   print(len(OneTime_train_speechID))

#   for i in range(0,len(y_List),1):
#      print(y_List[i])
#      print(OneTime_train_speechID[i])
   
#   for i in range(0,len(a_List),1):
#      print(a_List[i].shape)

   

	return (OneTime_train_speechID, OneTime_train_set , a_List, z_List)



def main():
   init(2,[3, 5, 6])
   forward()

   MFCC_DIM = 39
   LAYER_NUM = 5
   NEURON_NUM = 100
   PHONE_NUM = 48
   BATCH_SIZE = 10

if __name__ == '__main__':
   main()



