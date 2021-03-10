# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:56:17 2021

@author: Jason
"""

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='IDRR')
    
    # word2vec
    parser.add_argument('--WORD2VEC_DIR', default='word2vec_0.05.pickle')
    
    # dataset
    parser.add_argument('--train_size', default=17945,  type=int,   help='Train set size')
    parser.add_argument('--dev_size',  default=1653,  type=int,   help='Dev set size')
    parser.add_argument('--test_size',  default=1474,  type=int,   help='Test set size')
    parser.add_argument('--num_class',  default=4,     type=int,   help='Number of classes')

    # # model arguments
    parser.add_argument('--in_dim',     default=300,   type=int,   help='Size of input word vector')
    parser.add_argument('--h_dim',      default=300,   type=int,   help='Size of hidden unit')
    # parser.add_argument('--len_arg',  default=50,    type=int,   help='Argument length')
    # parser.add_argument('--num_topic',  default=30,    type=int,   help='Topic number')
    # parser.add_argument('--en1_units',  default=100,   type=int)
    # parser.add_argument('--en2_units',  default=100,   type=int)
    # parser.add_argument('--init_mult',  default=1.0,   type=float, help='multiplier in initialization of decoder weight')
    # parser.add_argument('--variance',   default=0.995, type=float, help='default variance in prior normal')

    # # training arguments
    parser.add_argument('--num_epoch',  default=50,    type=int,   help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32,     type=int,   help='batchsize for optimizer updates')
    parser.add_argument('--lr',         default=0.001, type=float, help='initial learning rate')
    parser.add_argument('--wd',         default=5e-5,  type=float, help='weight decay')
    parser.add_argument('--momentum',   default=0.99,  type=float)
    
    args = parser.parse_args()
    return args
