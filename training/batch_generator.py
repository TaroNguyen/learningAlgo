import numpy as np
import pandas as pd
import scipy.stats
import math
# checked with pandas, this method generates no null value
from training.data_transformation import transform


def batch_generator(generator, batch_size, transformation = transform):
    while True:
        batches_X = []
        batches_Y = []
        for i in range( batch_size):
            process = next( generator)
            process = transformation( process)
            one_hot = one_hot_encoder( indice, max_class)
            batches_X.append( process)
            batches_Y.append( one_hot)

        yield (batches_X, batches_Y)
