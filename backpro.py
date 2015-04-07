
#from theano import * 
import numpy as np
import theano.tensor as T
from theano import function



def backpropagate (gradC = np.zeros(1) , z  = [],  a = [] , theta = [], featureset = [], batchsize = 0):
        #t = T.dvector('t')
        #u = T.nnet.sigmoid(t)
        #funcSigmoid = function([t],u)
        a = T.dvector('a')
        b = T.dvector('b')
        aMb = a * b
        aAb = a + b
        funcMutiply = function([a, b], aMb)
        funcAdd = function([a, b], aAb)
	C = []
	ans  = []
	WC = []
	lum = []
	WT = []
	W = theta[0]
	b = theta[1]
        layer = len(W)
	for i in range(len(W)):
		WT.append(np.transpose(W[i]))

	# print 'W len=' + str(len(W))
	# print 'WT len=' + str(len(WT))

	for j in range(batchsize):
		WC = [None] * (layer + 1)
		lum = [None] * (layer + 1)
		for i in range(layer , -1, -1):
                        print 'for i'
			if i == layer :
				lum[i] = funcMutiply(derOfSigmoid(z[j][i - 1]), gradC)
				if i != 0 :
					WC[i] = funcMutiply(a[j][i-1], lum[i])
				else :
					WC[i] = funcMutiply(featureset[j], lum[i])
				if j == 0:
					#print 'i=' + str(i) 
					C.append(WC[i])
					C.append(lum[i])
				else:
					C[(layer  - i) * 2] = funcAdd(C[(layer - i) * 2], WC[i])
					C[(layer  - i) * 2 + 1] = funcAdd(C[(layer - i)*2 + 1], lum[i])
			else:
				# print 'sigz shape ' + str(sig(z[j][i]).shape)
				# print 'wt shape ' + str(WT[i+1].shape)
				# print 'lum shape ' + str(lum[i - 1].shape)
				#print 'i=' + str(i)
				
				if i != 0 :
					# lum[i] =  (sig(z[j][i - 1]) * WT[i].transpose() ).transpose().dot(lum[i+1])
                                        lum[i] = T.dot(funcMutiply(derOfSigmoid(z[j][i - 1]),WT[i].T).T, lum[i + 1])
					WC[i] = funcMutiply(a[j][i-1], lum[i])
				else :
					# lum[i] =  ( featureset[j] * WT[i].transpose() ).transpose().dot(lum[i+1]) 
                                        lum[i] = T.dot(funcMutiply(featureset[j], WT[i].T).T, lum[i + 1])
					WC[i] = funcMutiply(featureset[j], lum[i])
				if j == 0 :
					C.append(WC[i])
					C.append(lum[i])
				else:
					C[(layer  - i) * 2] = funcAdd(C[(layer - i) * 2], WC[i])
					C[(layer  - i) * 2 + 1] = funcAdd(C[(layer - i)*2 + 1], lum[i])


	for i in range(layer ):
		C[2*i] = C[2*i] / batchsize
		C[2*i + 1] = C[2*i + 1] / batchsize

	return C

def derOfSigmoid(z):
    temp = 1 / (1 + math.exp(-1*z))
    return = temp*(1-temp)



def main():
	backpropagate()

if __name__ == "__main__":
	main()
