
from theano import function 
import numpy as np
import theano.tensor as T


backpro_A = T.dvector('backpro_A')
backpro_B = T.dvector('backpro_B')
funcMultiply = function([backpro_A, backpro_B], backpro_A * backpro_B)

backpro_E = T.dvector('backpro_E')
backpro_F = T.dmatrix('backpro_F')
funcMatrixMulti = function([backpro_E, backpro_F], backpro_E * backpro_F)

backpro_ADD1 = T.dvector('backpro_ADD1')
backpro_ADD2 = T.dvector('backpro_ADD2')
funcAdd = function([backpro_ADD1, backpro_ADD2], backpro_ADD1 + backpro_ADD2)

backpro_toDotMatrix = T.dmatrix('backpro_toDotMatrix')
backpro_toDotVector = T.dvector('backpro_toDotVector')
backpro_ansDot = T.dot(backpro_toDotMatrix, backpro_toDotVector)
funcDot = function([backpro_toDotMatrix, backpro_toDotVector], backpro_ansDot)

backpro_t = T.dvector('backpro_t')
backpro_u = 1 / (T.exp((-1)*backpro_t)+1)
backpro_u_grad = backpro_u*(1-backpro_u)
funcSigmoid = function([backpro_t], backpro_u)
funcSigmoidGrad = function([backpro_t],backpro_u_grad)

def backpropagate (gradC = np.zeros(1) , z  = [],  a = [] , theta = [], featureset = [], batchsize = 0):

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
				lum[i] = funcMultiply(funcSigmoidGrad(z[j][i - 1]), gradC)
				if i != 0 :
					WC[i] = funcMultiply(a[j][i-1], lum[i])
                                        #WC[i] = a[j][i-1] * lum[i]
				else :
					WC[i] = funcMultiply(featureset[j], lum[i])
                                        #WC[i] = featureset[j] * lum[i]
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
                                        #lum[i] = (funcSigmoidGrad(z[j][i-1]) * WT[i].transpose()).transpose().dot(lum[i+1])
                                        #lum[i] = funcMatrixMulti(funcSigmoidGrad(z[j][i-1]), WT[i].transpose()).transpose().dot(lum[i+1])
                                        toDot = funcMatrixMulti(funcSigmoidGrad(z[j][i-1]), WT[i].transpose()).transpose()
                                        lum[i] = funcDot(toDot, lum[i+1])
					WC[i] = funcMultiply(a[j][i-1], lum[i])
				else :
                                        #lum[i] = (featureset[j]*WT[i].transpose()).transpose().dot(lum[i+1])
                                        toDot = funcMatrixMulti(featureset[j], WT[i].transpose()).transpose()
                                        lum[i] = funcDot(toDot, lum[i+1])
					WC[i] = funcMultiply(featureset[j], lum[i])
                                        #WC[i] = featureset[j] * lum[i]
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


def main():
	backpropagate()

if __name__ == "__main__":
	main()
