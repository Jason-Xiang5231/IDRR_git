# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 23:11:43 2021

@author: Jason
"""

import csv

csv.register_dialect('mydialect',delimiter='\t',quoting=csv.QUOTE_ALL)

def load_data():
    # text
    train_data = []
    dev_data = []
    test_data = []
    with open('./output/train.tsv', 'r', encoding='utf-8') as tsvtrain:
        train_file_list = csv.reader(tsvtrain,'mydialect')
        for line in train_file_list:
            train_data.append(line)
            
    with open('./output/dev.tsv', 'r', encoding='utf-8') as tsvdev:
        dev_train_file_list = csv.reader(tsvdev,'mydialect')
        for line in dev_train_file_list:
            dev_data.append(line)
            
    with open('./output/test.tsv', 'r', encoding='utf-8') as tsvtest:
        test_file_list = csv.reader(tsvtest,'mydialect')
        for line in test_file_list:
            test_data.append(line)


    return train_data, dev_data, test_data