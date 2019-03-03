from os import listdir
import numpy as np
import re
from os.path import join

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def load_data (datapath):
    Data = {}
    for itm in natural_sort(listdir(datapath)):
        if itm.endswith(".txt"):
            Data[itm] = np.loadtxt(join(datapath, itm), dtype=np.str, skiprows=15, usecols=1, comments='>')
            Data[itm] = np.char.replace(Data[itm], ',', '.').astype(np.float64)
    return Data

def average (data):
    dictlist = []
    Data = {}
    for itm in data :
        dictlist.append(data[itm])
    Data[itm] = np.mean(dictlist, axis=0)
    return Data

def norm_max (data):
    Data = {}
    for itm in data:
        Data[itm] = data[itm] / max(data[itm])
    return Data

# def smooth (Data):
#     for itm in Data:
#         Data[itm] = signal.savgol_filter(Data[itm], 11, 1)
#
# def backgroundless (Data, first, last):
#     for itm in Data:
#         Data[itm] = Data[itm] - np.mean(Data[itm][first:last])

