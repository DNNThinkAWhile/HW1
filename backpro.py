
#from theano import * 
import numpy as np
#import theano.tensor as T

def sig (x):
	return 1 / (1 + np.exp(-x))

def backpropagate (gradC = np.zeros(1) , z  = [],  a = [] , layer = 0, theta = [], featureset = [], batchsize = 0):
	C = []
	ans  = []
	WC = []
	lum = []
	WT = []
	W = theta[0]
	b = theta[1]
	for i in range(len(W)):
		WT.append(np.transpose(W[i]))

	print 'W len=' + str(len(W))
	print 'WT len=' + str(len(WT))

	for j in range(batchsize):
		WC = []
		lum = []
		for i in range(layer - 1, -1, -1):
			if i == layer - 1 :
				lum.append(sig(z[j][i])*gradC)
				if i != 0 :
					print 'a:' + str(a[j][i - 1])
					print 'lum:' + str(lum[layer - 1 - i])
					WC.append(a[j][i]*lum[layer - 1 - i])
				else :
					WC.append(featureset[j] * lum[layer - 1 - i])
				if j == 0:
					print 'i=' + str(i) 
					C.append(WC[layer - 1 - i])
					C.append(lum[layer - 1 - i])
				else:
					C[(layer - 1 - i) * 2] = C[(layer - 1 - i) * 2] + WC[i]
					C[(layer - 1 - i) * 2 + 1] = C[(layer - 1 - i) * 2 + 1] + lum[i]
			else:
				print 'sigz shape ' + str(sig(z[j][i]).shape)
				print 'wt shape ' + str(WT[i+1].shape)
				print 'lum shape ' + str(lum[i - 1].shape)
				print 'i=' + str(i)
				lum.append( (sig(z[j][i]) * WT[i + 1].transpose() ).transpose().dot(lum[i - 1]) )
				if i != 0 :
					WC.append(a[j][i]*lum[layer - 1 - i])
				else :
					WC.append(featureset[j] * lum[layer - 1 - i])
				if j == 0 :
					C.append(WC[layer - 1 - i])
					C.append(lum[layer - 1 - i])
				else:
					C[(layer - 1 - i) * 2] = C[(layer - 1 - i) * 2] + WC[i]
					C[(layer - 1 - i) * 2 + 1] = C[(layer - 1 - i) * 2 + 1] + lum[i]


	for i in range(layer - 1):
		C[2*i] = C[2*i] / batchsize
		C[2*i + 1] = C[2*i + 1] / batchsize

	return C
	   
def main():
	backpropagate()

if __name__ == "__main__":
	main()
