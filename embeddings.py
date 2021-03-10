# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 10:27:57 2021

@author: Jason
"""

import torch
import torch.nn as nn
import pickle
import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
import gensim
from load_data import load_data
from prepro_data import prepro_data_train, prepro_data_dev, prepro_data_test

# 读取tsv文件
train_data, dev_data, test_data = load_data()

# 读取arg_1、arg_2、label
train_arg_1, train_arg_2, train_label = prepro_data_train(train_data)
dev_arg_1, dev_arg_2, dev_label = prepro_data_dev(dev_data)
test_arg_1, test_arg_2, test_label = prepro_data_test(test_data)

w2v_dir = 'D:/_科研/0语料/词向量/GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(w2v_dir, binary=True)

word_list = {}
arg_all = train_arg_1 + train_arg_2 + dev_arg_1 + dev_arg_2 + test_arg_1 + test_arg_2
for text in arg_all:
    for word in text:
        word_list[word] = 0
        


# word2vec
word_vecs = {}
# word_vecs_mat = []
miss_num = 0
for word in word_list:
	try:
		vec = model.wv[word].astype('float')
		word_vecs[word] = vec
		# word_vecs_mat.append(vec)
	except KeyError:
		vec = torch.FloatTensor(300)
		nn.init.uniform(vec, -0.05, 0.05)
		vec = vec.numpy()
		# vec = np.zeros(100, dtype=np.float32)
		word_vecs[word] = vec
		# word_vecs_mat.append(vec)
		miss_num += 1
# word_vecs_mat = np.array(word_vecs_mat, dtype=np.float32)
print('Vocab Length: ', len(word_vecs))
print('Missing Num: ', miss_num)
print('Missing Rate: {:.4f}'.format(miss_num/len(word_vecs)))

# save
with open('word2vec_0.05.pickle', 'wb') as f:
 	pickle.dump(word_vecs, f, protocol=pickle.HIGHEST_PROTOCOL)
# np.save('word2vec.npy', word_vecs_mat)