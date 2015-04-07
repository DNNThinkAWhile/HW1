
from theano import function 
import numpy as np
import theano.tensor as T


def backpropagate (gradC = np.zeros(1) , z  = [],  a = [] , theta = [], featureset = [], batchsize = 0):
        A = T.dvector('A')
        B = T.dvector('B')
        funcMutiply = function([A, B], A * B)

        t = T.dvector('t')
        u = 1 / (T.exp((-1)*t)+1)
        u_grad = u*(1-u)
        funcSigmoid = function([t], u)
        funcSigmoidGrad = function([t],u_grad)

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
			if i == layer :
                                #lum[i] = funcSigmoid(z[j][i-1]) * gradC
				lum[i] = funcMutiply(funcSigmoidGrad(z[j][i - 1]), gradC)
				if i != 0 :
					WC[i] = funcMutiply(a[j][i-1], lum[i])
                                        #WC[i] = a[j][i-1] * lum[i]
				else :
					WC[i] = funcMutiply(featureset[j], lum[i])
                                        #WC[i] = featureset[j] * lum[i]
				if j == 0:
					#print 'i=' + str(i) 
					C.append(WC[i])
					C.append(lum[i])
				else:
					C[(layer  - i) * 2] = C[(layer-i)*2] + WC[i] # funcAdd(C[(layer - i) * 2], WC[i])
					C[(layer  - i) * 2 + 1] = C[(layer-i)*2+1] + lum[i]  # funcAdd(C[(layer - i)*2 + 1], lum[i])
			else:
				# print 'sigz shape ' + str(sig(z[j][i]).shape)
				# print 'wt shape ' + str(WT[i+1].shape)
				# print 'lum shape ' + str(lum[i - 1].shape)
				#print 'i=' + str(i)
				
				if i != 0 :
					# lum[i] =  (sig(z[j][i - 1]) * WT[i].transpose() ).transpose().dot(lum[i+1])
                                        lum[i] = (funcSigmoidGrad(z[j][i-1]) * WT[i].transpose()).transpose().dot(lum[i+1])
                                        #lum[i] = (funcMutiply(funcSigmoid(z[j][i-1]), WT[i].transpose())).transpose().dot(lum[i+1])
					WC[i] = funcMutiply(a[j][i-1], lum[i])
				else :
                                        lum[i] = (featureset[j]*WT[i].transpose()).transpose().dot(lum[i+1])
                                        #lum[i] = (funcMultiply(featureset[j], WT[i].transpose())).transpose().dot(lum[i+1])
					WC[i] = funcMutiply(featureset[j], lum[i])
                                        #WC[i] = featureset[j] * lum[i]
				if j == 0 :
					C.append(WC[i])
					C.append(lum[i])
				else:
					C[(layer  - i) * 2] = C[(layer-i)*2] + WC[i] #funcAdd(C[(layer - i) * 2], WC[i])
					C[(layer  - i) * 2 + 1] = C[(layer-i)*2 + 1] + lum[i] #funcAdd(C[(layer - i)*2 + 1], lum[i])


	for i in range(layer ):
		C[2*i] = C[2*i] / batchsize
		C[2*i + 1] = C[2*i + 1] / batchsize

	return C


def main():
	backpropagate()

if __name__ == "__main__":
	main()
