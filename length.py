# -*- coding: utf-8 -*-

# import pandas as pd

# train = pd.read_table('./output/train.tsv')
# dev = pd.read_table('./output/dev.tsv')
# test = pd.read_table('./output/test.tsv')

# print(train)
# print(dev)
# print(test)

import csv
import matplotlib.pyplot as plt
import numpy as np

# import numpy as np

csv.register_dialect('mydialect',delimiter='\t',quoting=csv.QUOTE_ALL)

test_idx = []
test_label_1 = []
test_label_2 = []
test_arg_1 = []
test_arg_2 = []
test_conn_1 = []
test_conn_2 = []

with open('./output/test.tsv',) as CsvTest:
    test_file_list = csv.reader(CsvTest,'mydialect')
    for line in test_file_list:
        test_idx.append(line[0].strip('\n').split(' '))
        test_label_1.append(line[4].strip('\n').split(' '))
        test_label_2.append(line[5].strip('\n').split(' '))
        test_arg_1.append(line[7].strip('\n').split(' '))
        test_arg_2.append(line[8].strip('\n').split(' '))
        test_conn_1.append(line[9].strip('\n').split(' '))
        test_conn_2.append(line[11].strip('\n').split(' '))

train_idx = []
train_label = []
train_arg_1 = []
train_arg_2 = []
train_conn = []

with open('./output/train.tsv',) as CsvTrain:
    train_file_list = csv.reader(CsvTrain,'mydialect')
    for line in train_file_list:
        train_idx.append(line[0].strip('\n').split(' '))
        train_label.append(line[4].strip('\n').split(' '))
        train_arg_1.append(line[6].strip('\n').split(' '))
        train_arg_2.append(line[7].strip('\n').split(' '))
        train_conn.append(line[8].strip('\n').split(' '))
        
dev_idx = []
dev_label_1 = []
dev_label_2 = []
dev_arg_1 = []
dev_arg_2 = []
dev_conn_1 = []
dev_conn_2 = []

with open('./output/dev.tsv',) as CsvDev:
    dev_file_list = csv.reader(CsvDev,'mydialect')
    for line in dev_file_list:
        dev_idx.append(line[0].strip('\n').split(' '))
        dev_label_1.append(line[4].strip('\n').split(' '))
        dev_label_2.append(line[5].strip('\n').split(' '))
        dev_arg_1.append(line[7].strip('\n').split(' '))
        dev_arg_2.append(line[8].strip('\n').split(' '))
        dev_conn_1.append(line[9].strip('\n').split(' '))
        dev_conn_2.append(line[11].strip('\n').split(' '))
        

all_length = [len(i) for i in test_arg_1] + [len(i) for i in test_arg_2] + [len(i) for i in train_arg_1] + [len(i) for i in train_arg_2] + [len(i) for i in dev_arg_1] + [len(i) for i in dev_arg_2]
arg_1_length = [len(i) for i in test_arg_1] + [len(i) for i in train_arg_1] + [len(i) for i in dev_arg_1] 
arg_2_length = [len(i) for i in test_arg_2] + [len(i) for i in train_arg_2] + [len(i) for i in dev_arg_2]

plt.hist(arg_1_length, bins=150)
plt.show()
plt.hist(arg_2_length, bins=150)
plt.show()
plt.hist(all_length, bins=150)
plt.show()

print('Arg_1 length < 50 :  ', np.mean(np.array(arg_1_length) < 50))
print('Arg_2 length < 50 :  ', np.mean(np.array(arg_2_length) < 50))
print('Arg_all length < 50 :  ', np.mean(np.array(all_length) < 50))


