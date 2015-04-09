
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

	finalWC = [None] * (layer)
	finallum = [None] * (layer)

	for j in range(batchsize):
		WC = [None] * (layer)
		lum = [None] * (layer)
		for i in range(layer-1 , -1, -1):
			if i == layer -1:
				lum[i] = funcMultiply(funcSigmoidGrad(z[j][i]), gradC)
				if i != 0 :
					WC[i] = func2MatrixDot( np.matrix(lum[i]).transpose() , np.matrix(a[j][i-1]) )
				else :
					WC[i] = func2MatrixDot( np.matrix(lum[i]).transpose() , np.matrix(featureset[j]) )
				if j == 0:
					finalWC[i] = WC[i]
					finallum[i] = lum[i]
				else:
					finalWC[i] = func2MatrixAdd(finalWC[i], WC[i])
					finallum[i] = funcAdd(finallum[i], lum[i])
			else:
				if i != 0 :
                                        #lum[i] = funcCalLum(z[j][i], WT[i+1], lum[i+1])
                                        lum[i] = func2vectorDot(funcSigmoidGrad(z[j][i]),funcDot(WT[i+1], lum[i+1]))
					WC[i] = func2MatrixDot( np.matrix(lum[i]).transpose() , np.matrix(a[j][i-1]) )
				else :
                                        #lum[i] = funcCalLum(z[j][i], WT[i+1], lum[i+1])
                                        lum[i] = func2vectorDot(funcSigmoidGrad(z[j][i]),funcDot(WT[i+1], lum[i+1]))
					WC[i] = func2MatrixDot( np.matrix(lum[i]).transpose() , np.matrix(featureset[j]) )
				if j == 0 :
					finalWC[i] = WC[i]
					finallum[i] = lum[i]
				else:
					finalWC[i] = func2MatrixAdd(finalWC[i], WC[i])
					finallum[i] = funcAdd(finallum[i], lum[i])


	for i in range(layer ):
		finalWC[i] /= batchsize
		finallum[i]  /= batchsize

	return (finalWC, finallum)


def main():
	backpropagate()

if __name__ == "__main__":
	main()
