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
    if len(z) != len(W) or len(a) != len(b) or len(W) != L or len(b) != L :
        print "Error List Size"
        return

    for Warray in W:
        WT.append(np.transpose(Warray))

    for i in range(L - 1, -1, -1):
        if i == L - 1 :
            lum.append(sig(z[i])*gradC)
            if i != 0 :
                WC.append(a[i - 1]*lum[L - 1 - i])
            else :
                WC.append(data * lum[L - 1 - i])
            C.append(Wans[i])
            C.append(lum[i])
        else:
            lum.append(sig(z[i])*WT[i]*lum[L - i])
            if i != 0 :
                WC.append(a[i - 1]*lum[L - 1 - i])
            else :
                WC.append(data * lum[L - 1 - i])
            C.append(Wans[i])
            C.append(lum[i])

    return C
       
def main():
    backpropagate()

if __name__ == "__main__":
    main()
