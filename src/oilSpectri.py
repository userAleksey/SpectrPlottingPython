#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join, isdir, abspath
from os import makedirs
from os import listdir

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# from scipy.optimize import curve_fit
# from statistics import mean
# from matplotlib import style
import datetime
# style.use('ggplot')

import re

from scipy import signal

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

do_save = 0
rad_type = 'sun'
day_of_exp = '28_02_19'
tos = 'clear2'
tos2 = 0
tos2 = 'oil'

norm_max = 1
smooth = 0
backgroundless = 0
legend = 1
average = 1

font = {'family': 'normal',
        'weight': 'bold',
        'size': 24}

matplotlib.rc('font', **font)

plt.rcParams["axes.labelweight"] = "bold"

# if __name__ == '__main__':
#    dir = join('..','data',day_of_exp)
# else:
#    dir = join('data',day_of_exp)

dir = join('..', 'data', day_of_exp, tos)
dir_d1 = dir
Data1 = {}

if tos2 :
    dir2 = join('..', 'data', day_of_exp, tos2)
    dir_d2 = dir2
    Data2 = {}

# wl = np.loadtxt(join(dir_d1,listdir(dir_d1)[0]),skiprows = 14, usecols = 0, comments='>')
wl = np.loadtxt(join(dir_d1, listdir(dir_d1)[0]), dtype=np.str, usecols=0, skiprows=15)
wl = np.char.replace(wl, ',', '.').astype(np.float64)
# wl = np.char.replace(wl, '\'', '')
# wl = np.char.replace(wl, 'b', '')

for itm in natural_sort(listdir(dir_d1)):
    if itm.endswith(".txt"):
        # Data1[itm] = np.loadtxt(join(dir_d1,itm), skiprows=14,usecols = 1, comments='>')
        Data1[itm] = np.loadtxt(join(dir_d1, itm), dtype=np.str, skiprows=15, usecols=1, comments='>')
        Data1[itm] = np.char.replace(Data1[itm], ',', '.').astype(np.float64)

if tos2 :
    for itm in natural_sort(listdir(dir_d2)):
        if itm.endswith(".txt"):
            Data2[itm] = np.loadtxt(join(dir_d2, itm), dtype=np.str, skiprows=15, usecols=1, comments='>')
            Data2[itm] = np.char.replace(Data2[itm], ',', '.').astype(np.float64)

res_dir = join('..', 'results', str(datetime.date.today()), day_of_exp)
sna = join(res_dir, tos)
if tos2 :
    sna = join(res_dir, tos, '_', tos2)


if not isdir(res_dir):
    makedirs(abspath(res_dir))

fig, (ax1) = plt.subplots(1, 1, figsize=(18, 10))
fig.set_tight_layout(True)

if smooth == 1:
    for itm in Data1:
        Data1[itm] = signal.savgol_filter(Data1[itm], 11, 1)

if backgroundless == 1:
    for itm in Data1:
        Data1[itm] = Data1[itm] - np.mean(Data1[itm][100:200])



Data = {}

if average == 1:
    dictlist = []
    for itm in Data1:
        dictlist.append(Data1[itm])
    Data[tos] = np.mean(dictlist, axis=0)
    if tos2 :
        dictlist = []
        for itm in Data2:
            dictlist.append(Data2[itm])
        Data[tos2] = np.mean(dictlist, axis=0)

if norm_max == 1:
    for itm in Data:
        Data[itm] = Data[itm] / max(Data[itm])

for itm in Data :
    ax1.plot(wl, Data [itm])

if legend == 1:
    ax1.legend(list(Data.keys()), loc=0)

ax1.set(xlabel='Wavelength, nm',
        ylabel='I, units',
        title=rad_type + ' ' + tos,
        xlim=[300, 850],
        #       ylim = [0,1]
        )

if do_save == 1:
    fig.savefig(sna + '.png', transparent=False, dpi=300, bbox_inches="tight")

plt.show()