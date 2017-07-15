'''Keras utilis for data preprocessing and postprocessing
'''

# from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility
from keras.preprocessing import sequence

def padding(input_list, max_len):
  return sequence.pad_sequences(input_list, maxlen=max_len, dtype='int32')

