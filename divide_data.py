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


def partition_data(data_dir, labels, validation_percentage, testing_percentage):
    random.seed(RANDOM_SEED)
    search_path = os.path.join(data_dir, '*', '*.wav')
    labels_index = {}
    for index, label in enumerate(labels):
        labels_index[label] = index
    data_set = {'validation': [], 'testing': [], 'training': []}
    unknown_set = {'validation': [], 'testing': [], 'training': []}
    for file_path in gfile.Glob(search_path):
        _, label = os.path.split(os.path.dirname(file_path))  # dirname 去除文件名
        label = label.lower()
        if label == BACKGROUND_NOISE_DIR_NAME:
            continue
        set_name = split_10(file_path, validation_percentage, testing_percentage)
        if label in labels_index:
            data_set[set_name].append({'label': label, 'file': file_path})
        else:
            unknown_set[set_name].append({'label': UNKNOWN_WORD_LABEL, 'file': file_path})

    # select 10% data be the slience data and add to data_set
    silence_wav_path = data_set['training'][0]['file']
    for set_index in ['validation', 'testing', 'training']:
        set_size = len(data_set[set_index])
        print(set_size)
        silence_size = int(math.ceil(set_size * 10.0 / 100))
        for _ in range(silence_size):
            data_set[set_index].append({
                'label': SILENCE_LABEL,
                'file': silence_wav_path
            })
        random.shuffle(unknown_set[set_index])
        # 10% unknown data add to data_set
        unknown_size = int(math.ceil(set_size * 10.0 / 100))
        data_set[set_index].extend(unknown_set[set_index][:unknown_size])
    # rearrange data set
    for set_index in ['validation', 'testing', 'training']:
        random.shuffle(data_set[set_index])
    return data_set



if __name__ == '__main__' :
    data_dir = r'D:\WorkSpace\test'
    label_dir = r'D:\WorkSpace\KWS\words.txt'
    data = load_data.load_samples(data_dir)





