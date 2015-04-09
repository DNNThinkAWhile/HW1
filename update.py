import numpy as np


def save_model(theta, epoch, iteration):
    save_model_path = 'model_' + str(epoch) + '_' + str(iteration)
    np.save(save_model_path, theta)

def update(learning_rate, W, B, C):
    layer = len(W)
    for l in range(layer):
        W[l] -= learning_rate*C[0][l]
        B[l] -= learning_rate*C[1][l]
    theta = []
    theta.append(W)
    theta.append(B)
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
    theta = update(learning_rate, W, B, C, layer, iteration)

if __name__ == '__main__':
    main()
