import numpy as np
import theano.tensor as T
from theano import function
import random
import math
from calculate_error import *

MFCC_trainFilePath = '/home/rason/MLDS_HW1_RELEASE_v1/mfcc/test'

w_Min = -1
w_Max = 1

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

    for i in range(layer):
        w_i = np.random.uniform(w_Min/np.sqrt(neuron[i])*3, w_Max/np.sqrt(neuron[i])*3, (neuron[i+1], neuron[i]))
        b_i = np.random.uniform(w_Min/np.sqrt(neuron[i])*3, w_Max/np.sqrt(neuron[i])*3, neuron[i+1])
        w_List.append(w_i)
        b_List.append(b_i)

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

# def forward(List2D_MFCC_data, List_speakID, WandB, BATCH_SIZE, iteration, isTest):
#     w_List = WandB[0]
#     b_List = WandB[1]
#     a_List = []
#     z_List = []
#     #   y_List = []
#     LAYER_NUM = len(w_List)

#     trainDataSetNum = len(List2D_MFCC_data) 
#     OneTime_train_set = []
#     OneTime_train_speechID = []
#     if isTest == 0:
#         start = iteration * BATCH_SIZE
#         features_part = List2D_MFCC_data[start: start + BATCH_SIZE]
#         OneTime_train_set.extend(features_part)
#         OneTime_train_speechID.extend(List_speakID[start: start + BATCH_SIZE])
#     else:
#         for i in range(BATCH_SIZE):
#             arr = List2D_MFCC_data[i]
#             OneTime_train_set.append(arr)
#             OneTime_train_speechID.append(List_speakID[i])

#     for array in OneTime_train_set:
#         temp_a_List = []
#         temp_z_List = []
#         z_layer = array
#         for layer in range(0,LAYER_NUM,1):
#             #print str(w_List[layer].shape) + str(z_layer.shape) + str(b_List[layer].shape)
#             z_layer = f_matrix_dot( w_List[layer] , z_layer , b_List[layer])
#             temp_z_List.append(z_layer)
#             z_layer = sigmoid(z_layer)
#             temp_a_List.append(z_layer)

#         a_List.append(temp_a_List)
#         z_List.append(temp_z_List)      


#     return (OneTime_train_speechID, OneTime_train_set , a_List, z_List)





backpro_A = T.dvector('backpro_A')
backpro_B = T.dvector('backpro_B')
funcMultiply = function([backpro_A, backpro_B], backpro_A * backpro_B)

backpro_E = T.dvector('backpro_E')
backpro_F = T.dmatrix('backpro_F')
funcMatrixMulti = function([backpro_E, backpro_F], backpro_E * backpro_F)

backpro_ADD1 = T.dvector('backpro_ADD1')
backpro_ADD2 = T.dvector('backpro_ADD2')
funcAdd = function([backpro_ADD1, backpro_ADD2], backpro_ADD1 + backpro_ADD2)

backpro_2MatrixAddA = T.dmatrix('backpro_2MatrixAddA')
backpro_2MatrixAddB = T.dmatrix('backpro_2MatrixAddB')
func2MatrixAdd = function([backpro_2MatrixAddA, backpro_2MatrixAddB], backpro_2MatrixAddA+backpro_2MatrixAddB)

backpro_toDotMatrix = T.dmatrix('backpro_toDotMatrix')
backpro_toDotVector = T.dvector('backpro_toDotVector')
backpro_ansDot = T.dot(backpro_toDotMatrix, backpro_toDotVector)
funcDot = function([backpro_toDotMatrix, backpro_toDotVector], backpro_ansDot)

backpro_2vectorDotA = T.dvector('backpro_2vectorDotA')
backpro_2vectorDotB = T.dvector('backpro_2vectorDotB')
func2vectorDot = function([backpro_2vectorDotA, backpro_2vectorDotB], backpro_2vectorDotA*backpro_2vectorDotB)


#funcCalLum = func2vectorDot(A,funcDot(B,C))     func2vectorDot : two vector element wise multiply  ;  funcDot : matrix dot vector (matrix multiply) 
backpro_calLum_z = T.dvector('backpro_calLum_z')
backpro_calLum_WT = T.dmatrix('backpro_calLum_WT')
backpro_calLum_nextLum = T.dvector('backpro_calLum_nextLum')
backpro_calLum_zSig = 1/(T.exp((-1)*backpro_calLum_z)+1)
backpro_calLum_zSigGrad = backpro_calLum_zSig*(1-backpro_calLum_zSig)
backpro_Lum = backpro_calLum_zSigGrad * T.dot(backpro_calLum_WT, backpro_calLum_nextLum)
funcCalLum = function([backpro_calLum_z, backpro_calLum_WT, backpro_calLum_nextLum],  backpro_Lum)



backpro_dotMatrixA = T.dmatrix('backpro_dotMatrixA') 
backpro_dotMatrixB = T.dmatrix('backpro_dotMatrixB') 
backpro_2MatrixDotAns = T.dot(backpro_dotMatrixA, backpro_dotMatrixB)
func2MatrixDot = function([backpro_dotMatrixA, backpro_dotMatrixB], backpro_2MatrixDotAns)

backpro_t = T.dvector('backpro_t')
backpro_u = 1 / (T.exp((-1)*backpro_t)+1)
backpro_u_grad = backpro_u*(1-backpro_u)
funcSigmoid = function([backpro_t], backpro_u)
funcSigmoidGrad = function([backpro_t],backpro_u_grad)

def numpy_sigmoid_grad(vec):
    sig = 1 / (np.exp((-1) * vec) + 1)
    return sig * (1 - sig)
    


# def backpropagate (gradC = np.zeros(1) , z  = [],  a = [] , theta = [], featureset = [], batchsize = 0):

#     C = []
#     ans  = []
#     WC = []
#     lum = []
#     WT = []
#     W = theta[0]
#     b = theta[1]
#         layer = len(W)
#     for i in range(len(W)):
#         WT.append(np.transpose(W[i]))


#     finalWC = [None] * (layer)
#     finallum = [None] * (layer)

#     for j in range(batchsize):
#         WC = [None] * (layer)
#         lum = [None] * (layer)
#         for i in range(layer-1 , -1, -1):
#             if i == layer -1:
#                 lum[i] = funcMultiply(funcSigmoidGrad(z[j][i]), gradC)
#             else:
#                 lum[i] = func2vectorDot(funcSigmoidGrad(z[j][i]),funcDot(WT[i+1], lum[i+1]))
#             if i > 0:
#                 WC[i] = func2MatrixDot( np.matrix(lum[i]).transpose() , np.matrix(a[j][i-1]) )
#             else:
#                 WC[i] = func2MatrixDot( np.matrix(lum[i]).transpose() , np.matrix(featureset[j]) )
#     return (finalWC, finallum)


class Dnn():
    def __init__(self, theta, batch_size):
        self.theta = theta
        w = theta[0]
        b = theta[1]
        self.layer_num = len(w)
        self.wc = [None] * (self.layer_num)
        self.lum = [None] * (self.layer_num)
        for i in range(self.layer_num):
            self.wc[i] = np.matrix(np.empty((w[i].shape)))
            self.lum[i] = np.empty((b[i].shape))

        # for back_batch
        self.wc_b = [None] * (self.layer_num)
        self.lum_b = [None] * (self.layer_num)
        self.batch_size = batch_size
        for i in range(self.layer_num):
            self.wc_b[i] = np.empty((w[i].shape))#np.matrix(np.empty((w[i].shape)))
            self.lum_b[i] = np.empty((b[i].shape[0], self.batch_size))


    def backprop(self, grad_c, z, a, feat):
        w = self.theta[0]    
        b = self.theta[1]
        layer_num = self.layer_num
        wc = self.wc
        lum = self.lum
        wt = []
        for i in range(layer_num):
            wt.append(np.transpose(w[i]))

        for i in range(layer_num - 1 , -1, -1):

            v_right = np.empty(lum[i].shape)
            if i == layer_num - 1:
                v_right = grad_c
            else:
                #v_right = funcDot(wt[i + 1], lum[i + 1])
                np.dot(wt[i + 1], lum[i + 1], v_right)
            # lum[i] = funcMultiply(funcSigmoidGrad(z[i]), v_right)
            np.multiply(numpy_sigmoid_grad(z[i]), v_right, lum[i])

            if i > 0:
                #wc[i] = func2MatrixDot( np.matrix(lum[i]).transpose() , np.matrix(a[i-1]) )
                # print wc[i].shape
                # print np.dot(np.matrix(lum[i]).transpose(), np.matrix(a[i-1])).shape
                np.dot(np.matrix(lum[i]).transpose(), np.matrix(a[i-1]), wc[i])
            else:
                #wc[i] = func2MatrixDot( np.matrix(lum[i]).transpose() , np.matrix(feat) )
                np.dot(np.matrix(lum[i]).transpose(), np.matrix(feat), wc[i])
        return wc, lum

    def backprop_batch(self, grad_c_b, z_b, a_b, feats_b, batch_size):
        w = self.theta[0]    
        b = self.theta[1]
        layer_num = self.layer_num
        wc_b = self.wc_b
        lum_b = self.lum_b
        lum = self.lum
        
        wt = []
        for i in range(layer_num):
            wt.append(np.transpose(w[i]))

        #z_b = numpy_sigmoid_grad(z_b)
        
        for i in range(layer_num - 1, -1, -1):
            vs_right = np.empty(lum_b[i].shape)
            if i == layer_num - 1:
                vs_right = grad_c_b
            else:
                # print 'i ', i
                # print 'a ', wt[i+1].shape
                # print 'b ', lum_b[i+1].shape
                # print 'c ', vs_right.shape
                np.dot(wt[i+1], lum_b[i+1], vs_right)

            np.multiply(numpy_sigmoid_grad(z_b[i]), vs_right, lum_b[i])

            if i > 0:
                # print 'aaa' ,lum_b[i].shape
                # print 'bbb', a_b[i-1].transpose().shape
                # print 'ccc', wc_b[i].shape
                np.dot(lum_b[i], a_b[i-1].transpose(), wc_b[i])
            else:
                # print 'ddddd'
                # print 'aaa' ,lum_b[i].shape
                # print 'bbb', feats_b.shape
                # print 'ccc', wc_b[i].shape
                np.dot(lum_b[i], feats_b.transpose(), wc_b[i])

            wc_b[i] /= batch_size
            lum[i] = np.mean(lum_b[i], axis=1)
        return wc_b, lum


    def forward(self, feat, spch_id):
        w = self.theta[0]
        b = self.theta[1]
        layer_num = self.layer_num
        a = []
        z = []
        a_layer = feat
        for l in range(0,layer_num,1):
            z_layer = f_matrix_dot(w[l], a_layer , b[l])
            z.append(z_layer)
            a_layer = sigmoid(z_layer)
            a.append(a_layer)
        return a, z


    def forward_batch(self, feats_b, batch_size):
        w = self.theta[0]
        b = self.theta[1]
        layer_num = self.layer_num
        a_b = []
        z_b = []
        a_layer = feats_b
        for l in range(0,layer_num,1):
            #z_layer = f_matrix_dot(w[l], a_layer , b[l])
            z_layer = np.dot(w[l], a_layer)
            for i in range(batch_size):
                z_layer[:,i] += b[l]
            z_b.append(z_layer)
            #a_layer = sigmoid(z_layer)
            a_layer = 1 / (np.exp((-1) * z_layer) + 1)
            a_b.append(a_layer)
        return a_b, z_b        


    def update(self, learning_rate, C):
        layer_num = self.layer_num
        for l in range(layer_num):
            self.theta[0][l] -= learning_rate*C[0][l]
            self.theta[1][l] -= learning_rate*C[1][l]
        

        # theta_log = '''
        # ++ W weighting matrix ++
        # W:       
        # {W}
      
        # ++ B matrix ++
        # {B}

        # =======================================================================
        # '''.format(W=self.theta[0], B=self.theta[1])


    def train(self, speech_ids, features, batch_size, iteration, phoneme_num, label_map, error_func, learning_rate):
        
        start = iteration * batch_size
        feats_part = features[start: start + batch_size]
        spch_ids_part = speech_ids[start: start + batch_size]
        d_w_all = [None] * batch_size
        d_b_all = [None] * batch_size

        # for back_batch
        layer_num = self.layer_num
        w = self.theta[0]
        b = self.theta[1]
        
        z_b = [None] * layer_num
        a_b = [None] * layer_num
        #feats_b = np.empty((w[0].shape[1], batch_size))
        feats_b = np.array(feats_part).transpose()
        grad_c_b = np.empty((w[-1].shape[0], batch_size))
        for i in range(self.layer_num):
            z_b[i] = np.empty((w[i].shape[0], batch_size))
            a_b[i] = np.empty((w[i].shape[0], batch_size))

        a_b,z_b = self.forward_batch(feats_b, batch_size)

        err_total = 0

        y_list = []
        for i in range(batch_size):
            y_list.append(a_b[-1][:,i])

        #exit(0)
        for i in range(batch_size):    

            spch_id = spch_ids_part[i]
            # feat = feats_part[i]

            # a, z = self.forward(feat, spch_id)

            y = a_b[-1][:,i]
            err, grad_c = calculate_error(phoneme_num, [spch_id], [y], label_map, error_func)

            # for back_batch
            # for l in range(layer_num):
            #     a_b[l][:,i] = a[l]
            #     z_b[l][:,i] = z[l]
            # feats_b[:,i] = feat
            grad_c_b[:,i] = grad_c
            err_total += err
            #(d_w_all[i], d_b_all[i]) = self.backprop(grad_c, z, a, feat)

        d_w, d_b = self.backprop_batch(grad_c_b, z_b, a_b, feats_b, batch_size)

        # print '~~~~~~~~~~~~~~~~~~'
        # print d_w[0].shape
        # print d_w[1].shape
        # print d_w[2].shape
        # print '~~~~~~~~~~~~~~~~~~~~~~~~ '
        # print d_b[0].shape
        # print d_b[1].shape
        # print d_b[2].shape
        
            # print 'd_w_all[i] len ' , len(d_w_all[i])
            # print 'd_w_all[i][0]: ', d_w_all[i][0]
            # print 'd_w_all[i][1] shape', d_w_all[i][1].shape
            # exit(0)


        # print 'd_w_all ', d_w_all
        # a = np.mean(d_w_all, axis=0)
        # print 'd_b_all ', d_b_all
        # b = np.mean(d_b_all, axis=0)
        
        #return np.mean(d_w_all, axis=0), np.mean(d_b_all, axis=0), err
        return d_w, d_b, err_total / batch_size



