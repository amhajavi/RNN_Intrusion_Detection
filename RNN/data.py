import tensorflow as tf
import numpy as np
import os


def fetch_from_file(FileName):
    temp_data = []
    flag = []
    max_length = 0
    for file_name in os.listdir(FileName):
        for file in os.listdir(os.path.join(FileName, file_name)):
            with open(os.path.join(FileName, file_name, file)) as data_record:
                record = data_record.readlines()[0].strip().split(' ')
                max_length = max([len(record), max_length])
                temp_data.append(record)
                if 'Attack' in file_name:
                    flag.append([1, 0])
                else:
                    flag.append([0, 1])
    data = []
    for i in temp_data:
        temp_list = [[0]] * 2948
        for idx, j in enumerate(i):
            temp_list[idx] = [j]
        data.append(np.array(temp_list))

    return data, flag, max_length


def prepare_train_data():
    return fetch_from_file('./Data/Train')


def prepare_test_data():
    return fetch_from_file('./Data/Test')