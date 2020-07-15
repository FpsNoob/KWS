import os
import numpy as np
import math
import random
import load_data
from tensorflow.python.platform import gfile

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
BACKGROUND_NOISE_DIR_NAME = '_background_noise_'
SILENCE_LABEL = '_silence_'
SILENCE_INDEX = 0
UNKNOWN_WORD_LABEL = '_unknown_'
UNKNOWN_WORD_INDEX = 1
RANDOM_SEED = 59185

def get_label(label_dir):
    with open(label_dir, 'r', encoding='utf-8') as file:
        content_list = file.readlines()
        contentall = [x.strip() for x in content_list]
    return contentall


def partition_data(data, data_path, labels):
    random.seed(RANDOM_SEED)
    labels_index = {}
    data_set = []
    unknown_set = []
    for index, label in enumerate(labels):
        labels_index[label] = index

    for i in range(np.size(data_path, axis=None)):
        _, label = os.path.split(os.path.dirname(data_path[i]))  # dirname 去除文件名
        label = label.lower()
        if label == BACKGROUND_NOISE_DIR_NAME:
            continue
        if label in labels_index:
            data_set.append({'data': data[i], 'label': label})
        else:
            unknown_set.append({'data': data[i], 'label': UNKNOWN_WORD_LABEL})

    # select 10% data be the slience data and add to data_set
    silence_wav_path = data_path[0]
    data_size = len(data_set)
    print('data_set size is %d', data_size)
    silence_size = int(math.ceil(data_size * 10.0 / 100))
    print('silence_size is %d', silence_size)
    for _ in range(silence_size):
        data_set.append({
            'data': data[0],
            'label': SILENCE_LABEL
            #'file': silence_wav_path
        })
    random.shuffle(unknown_set)
    # 10% unknown data add to data_set
    unknown_size = int(math.ceil(data_size * 10.0 / 100))
    print('unknown_size is %d', unknown_size)
    data_set.extend(unknown_set[:unknown_size])
    # rearrange data set
    random.shuffle(data_set)
    data = []
    lables = []
    for i in range(len(data_set)):
        for k, v in data_set[i].items():
            if k == 'data':
                data.append(v)
            else:
                lables.append(v)
    return data, lables


def split_10(data):
    data_x = []
    data_x.append(data[0:int(0.1 * len(data))])
    data_x.append(data[int(0.1 * len(data)):int(0.2 * len(data))])
    data_x.append(data[int(0.2 * len(data)):int(0.3 * len(data))])
    data_x.append(data[int(0.3 * len(data)):int(0.4 * len(data))])
    data_x.append(data[int(0.4 * len(data)):int(0.5 * len(data))])
    data_x.append(data[int(0.5 * len(data)):int(0.6 * len(data))])
    data_x.append(data[int(0.6 * len(data)):int(0.7 * len(data))])
    data_x.append(data[int(0.7 * len(data)):int(0.8 * len(data))])
    data_x.append(data[int(0.8 * len(data)):int(0.9 * len(data))])
    data_x.append(data[int(0.9 * len(data)):int(len(data))])

    test = data_x.pop(0)
    vali = data_x.pop(1)
    for i in range(8):
        if i == 0:
            train = np.array(data_x[i])
        else:
            train = np.concatenate((train, np.array(data_x[i])))

    return train, test, vali

#def MFCC_2D(data):


if __name__ == '__main__':
    data_dir = r'D:\test_data'
    label_dir = r'D:\WorkSpace\Python\KWS\words.txt'
    labels = get_label(label_dir)
    data, data_path = load_data.load_samples(data_dir)
    data, labels = partition_data(data, data_path, labels)
    data_train, data_test, data_vali = split_10(data)
    labels_train, labels_test, labels_vali =split_10(labels)







