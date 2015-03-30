
#from theano import * 
import numpy as np
#import theano.tensor as T

def sig (x):
    return 1 / (1 + np.exp(-x))

def backpropagate (gradC = np.zeros(1) , z  = [],  a = [] , layer = 0, theta = [], featureset = []):
    C = []
    WC = []
    lum = []
    WT = []
    W = theta[0]
    b = theta[1]
    sizez = len(z)
    for Warray in W:
        WT.append(np.transpose(Warray))

	for j in sizez:
		for i in range(layer - 1, -1, -1):
			if i == layer - 1 :
				lum.append(sig(z[j][i])*gradC)
				if i != 0 :
					WC.append(a[j][i - 1]*lum[layer - 1 - i])
				else :
					WC.append(featureset * lum[layer - 1 - i])
				C.append(WC[i])
				C.append(lum[i])
			else:
				lum.append(sig(z[j][i])*WT[i]*lum[layer - i])
				if i != 0 :
					WC.append(a[j][i - 1]*lum[layer - 1 - i])
				else :
					WC.append(featureset * lum[layer - 1 - i])
				C.append(WC[i])
				C.append(lum[i])

				
    return C
       
def main():
    backpropagate()

if __name__ == "__main__":
    main()
