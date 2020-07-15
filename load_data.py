import os
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from tensorflow.python.platform import gfile
MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M

def load_wav(data_dir):
    fs, wavsignal = wav.read(data_dir)
    return wavsignal


def display(wavsignal):
    plt.plot(wavsignal)
    plt.show()

def load_samples(dirctory):
    search_path = os.path.join(dirctory, '*', '*.wav')
    data_list = []
    data_path = []
    for filepath in gfile.Glob(search_path):
        print(filepath)
        a = load_wav(filepath)
        data_list.append(a)
        data_path.append(filepath)
    print('Loading Completed.')
    return data_list, data_path


