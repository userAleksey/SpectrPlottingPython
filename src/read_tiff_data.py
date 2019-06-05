# !/usr/bin/env python3
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



# import re

from scipy import signal

from somedataproc import managedata



do_save = 1
rad_type = 'LED 340(fluor_pogr)'
day_of_exp = '04_06_2019'

tos = ['С морской водой']

norm_max = 0
smooth = 0
backgroundless = 0
legend = 1
average = 1

font = {'family': 'normal',
        'weight': 'bold',
        'size': 24}

matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"

res_dir = join('..', 'results', day_of_exp, str(datetime.date.today()))
sna = join(res_dir, '-'.join(tos))

if not isdir(res_dir):
    makedirs(abspath(res_dir))

# ----------------------------------------------------

datapath = join('..', 'data', day_of_exp, tos[0])
#wl = np.loadtxt(join(datapath, listdir(datapath)[0]), dtype=np.str, usecols=0, skiprows=15)
#wl = np.char.replace(wl, ',', '.').astype(np.float64)
wl2 = np.loadtxt("../data/04_06_2019/WaveFl.txt")
wl = []
for i in range(len(wl2)):
    if i % 2 == 0:
        wl.append(wl2[i])

data = {}

for itm in tos:
    datapath = join('..', 'data', day_of_exp, itm)
    data[itm] = managedata.load_data(datapath)
    if average:
        data[itm] = managedata.average(data[itm])
    if norm_max:
        data[itm] = managedata.norm_max(data[itm])

# ----------------------------------------------------

fig, (ax1) = plt.subplots(1, 1, figsize=(18, 10))
fig.set_tight_layout(True)

for itm in data:
    for itm2 in data[itm]:
        ax1.plot(wl, data[itm][itm2])

if legend == 1:
    ax1.legend(list(data.keys()), loc=0)

ax1.set(xlabel='Wavelength, nm',
        ylabel='I, units',
        title=rad_type + ' ',
        xlim=[490, 800],
        #       ylim = [0,1]
        )

if do_save == 1:
    fig.savefig(sna + '.png', transparent=False, dpi=300, bbox_inches="tight")

plt.show()