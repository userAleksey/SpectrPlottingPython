from os import listdir
import numpy as np
import re
from os.path import join
from PIL import Image
from scipy import signal

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def load_data (datapath):
    Data = {}
    #for itm in natural_sort(listdir(datapath)):
    for itm in listdir(datapath):
        if itm.endswith(".txt"):
            Data[itm] = np.loadtxt(join(datapath, itm), dtype=np.str, skiprows=14, usecols=1, comments='>')
            Data[itm] = np.char.replace(Data[itm], ',', '.').astype(np.float64)
        if itm.endswith(".tif"):
            Data[itm] = np.sum(np.array(Image.open(join(datapath, itm)), dtype='float64')/65536.,0)
            Data[itm] = Data[itm][::-1]

    return Data

def load_data_with_sel (datapath, y_val_for_sel):
    Data = {}
    #for itm in natural_sort(listdir(datapath)):
    for itm in listdir(datapath):
        if itm.endswith(".txt"):
            data_itm = np.loadtxt(join(datapath, itm), dtype=np.str, skiprows=17, usecols=1, comments='>')
            data_itm = np.char.replace(data_itm, ',', '.').astype(np.float64)
            if max(data_itm) > y_val_for_sel:
                Data[itm] = data_itm
        if itm.endswith(".tif"):
            Data[itm] = np.sum(np.array(Image.open(join(datapath, itm)), dtype='float64')/65536.,0)
            Data[itm] = Data[itm][::-1]

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

def getivals (data, x, idxs):
    Data = {}
    for itm in data:
        Data[itm] = np.trapz(data[itm][idxs[0]:idxs[-1]],x[idxs[0]:idxs[-1]])
    return Data

def norm_val (data, x, idxs):
    Data = {}
    for itm in data:
        Data[itm] = data[itm] / max(data[itm][idxs[0]:idxs[-1]])
    return Data

def smooth (data):
    Data = {}
    for itm in data:
        Data[itm] = signal.savgol_filter(data[itm], 11, 1)
    return Data
#
# def backgroundless (Data, first, last):
#     for itm in Data:
#         Data[itm] = Data[itm] - np.mean(Data[itm][first:last])


