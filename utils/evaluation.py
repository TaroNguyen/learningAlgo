import numpy as np

def accuracy( onehot_prediction, onehot_labels):
    batch_size,  max_class = onehot_prediction.shape
    count=0.
    for i in range(batch_size):
        argmax = np.argmax( onehot_prediction[i])
        if onehot_labels[i][argmax] == 1 :
            count+=1.
    return count/batch_size
