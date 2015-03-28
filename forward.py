import numpy as np
import theano.tensor as T
from theano import function
import random
import math

MFCC_trainFilePath = '/home/rason/MLDS_HW1_RELEASE_v1/mfcc/test'

w_Min = -1
w_Max = 1

###            set theano matrix dot function            ### 

p = T.dmatrix('p')
q = T.dvector('q')
r = T.dvector('r')
s = T.dot(p,q) + r 
f_matrix_dot = function( [p,q,r] , s)

t = T.dvector('t')
u = T.nnet.sigmoid(t)
f_sigmoid = function([t],u)


#return (W,B)
def init(layer,neuron):
   w_List = []
   b_List = []
   
   LAYER_NUM = layer;
   MFCC_DIM = neuron[0]
   NEURON_NUM = neuron[1]
   PHONE_NUM = neuron[layer]
   # print 'layer:' + str(layer)
   # print 'neuron:' + str(neuron)
   # print 'MFCC_DIM:' + str(MFCC_DIM)
   # print 'NEURON_NUM:' + str(NEURON_NUM)
   # print 'PHONE_NUM:' + str(PHONE_NUM)

   w1 = np.random.uniform(w_Min,w_Max,(NEURON_NUM,MFCC_DIM))
   b1 = np.random.uniform(w_Min,w_Max,(NEURON_NUM))
   w_List.append(w1);
   b_List.append(b1);
   
   for i in range(1,LAYER_NUM,1):
      if i == LAYER_NUM - 1:
         w_i = np.random.uniform(w_Min,w_Max,(PHONE_NUM,NEURON_NUM))
         b_i = np.random.uniform(w_Min,w_Max,(PHONE_NUM))
      else:
         w_i = np.random.uniform(w_Min,w_Max,(NEURON_NUM,NEURON_NUM))
         b_i = np.random.uniform(w_Min,w_Max,(NEURON_NUM))
      w_List.append(w_i)
      b_List.append(b_i)
   
   rList = []
   rList.append(w_List);
   rList.append(b_List);
   
   return rList


def read_file(filePath):
   List_speakID = []
   List2D_MFCC_data = []
   
   MFCC_trainFile = open(filePath) 
   while 1:
      lines = MFCC_trainFile.readlines(10000)
      if not lines:
         break;
      for line in lines:
         partition = line.split()
         List_speakID.append(partition[0])
         List2D_MFCC_data.append(partition[1:])

   MFCC_trainFile.close()
   
   return (List_speakID, List2D_MFCC_data)


### get Count of "BATCH_SIZE" data to train for one time ### 

def forward(List2D_MFCC_data, List_speakID, WandB, BATCH_SIZE, isTest):
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
      for i in range(0,BATCH_SIZE,1):
         idx = (int)(random.random()*trainDataSetNum)
         arr = np.asfarray(List2D_MFCC_data[idx])
         OneTime_train_set.append(arr)
         OneTime_train_speechID.append(List_speakID[idx])
   else:
      OneTime_train_set = List2D_MFCC_data
      OneTime_train_speechID = List_speakID

   for array in OneTime_train_set:
      temp_a_List = []
      temp_z_List = []
      z_layer = array
      for layer in range(0,LAYER_NUM,1):
         z_layer = f_matrix_dot( w_List[layer] , z_layer , b_List[layer])
         temp_a_List.append(z_layer)
         z_layer = f_sigmoid(z_layer)
         temp_z_List.append(z_layer)
         
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



