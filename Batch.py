# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 22:09:31 2021

@author: Jason
"""
import numpy as np

def get_batch(text_data, w2v_model, indices):
    batch_size = len(indices)
    text_length = []
    for idx in indices:
        text_length.append(len(text_data[idx]))
    batch_x = np.zeros((batch_size, max(text_length), args.in_dim), dtype=np.float32)
    for i, idx in enumerate(indices, 0):
        for j, word in enumerate(text_data[idx], 0):
            batch_x[i][j] = w2v_model[word]

    return batch_x