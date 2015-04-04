import numpy as np


def save_model(theta, iteration):
    save_model_path = 'model_' + str(iteration)
    np.save(save_model_path, theta)

# def init(layer, neuron):
#     W = []
#     B = []
#     C1 = []
#     C2 = []
#     C = []
#     for l in range(layer):
#         w = np.random.rand(neuron[l], neuron[l+1])
#         c1 = np.random.rand(neuron[layer-l-1], neuron[layer-l])
#         b = np.random.rand(neuron[l+1])
#         c2 = np.random.rand(neuron[layer-l])
#         W.append(w)
#         C1.append(c1)
#         B.append(b)
#         C2.append(c2)
#     C.append(C1)
#     C.append(C2)
#     return W, B, C

def update(learning_rate, W, B, C, iteration):
    layer = len(W)
    for l in range(layer):
        W[l] -= learning_rate*C[0][layer-l-1]
        B[l] -= learning_rate*C[1][layer-l-1]
    theta_log = '''
    ++ W weighting matrix ++
    W:       
    {W}
  
    ++ B matrix ++
    {B}

    =======================================================================
    '''.format(W=W, B=B)
    theta = []
    theta.append(W)
    theta.append(B)
    # print theta for debugging...
    # print theta_log
    save_model(theta, iteration)
    return theta

def main():
    iteration = 10
    learning_rate = 0.01
    batch_size = 10
    layer = 3
    raw = 39
    phonemes = 48
    neuron = [raw, 2, 3, phonemes]
    
    W, B, C = init(layer, neuron)
    update(learning_rate, W, B, C, layer, iteration)

if __name__ == '__main__':
    main()
