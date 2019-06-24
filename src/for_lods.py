# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join, isdir, abspath
from os import makedirs
from os import listdir

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# from statistics import mean
# from matplotlib import style
import datetime
# style.use('ggplot')
# import re

from scipy import signal

from somedataproc import managedata, processdata

do_save = 0
rad_type = 'LED'
day_of_exp = '11_06 (DT LED)'

datapath = join('..', 'data', day_of_exp)

getivals = 1
norm_max = 0
smooth = 0
backgroundless = 0
legend = 1
average = 1

if getivals:
    ivals = {}
    lim1 = 300
    lim2 = 525

font = {'family': 'normal',
        'weight': 'bold',
        'size': 24}

matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"

res_dir = join('..', 'results', day_of_exp, str(datetime.date.today()))
sna = join(res_dir, '-'.join(day_of_exp))

if not isdir(res_dir):
    makedirs(abspath(res_dir))

# ----------------------------------------------------
for itm in listdir(datapath):
    if itm.endswith(".txt"):
        wl = np.loadtxt(join(datapath, itm ), dtype=np.str, usecols=0, skiprows=17, comments='>')
wl = np.char.replace(wl, ',', '.').astype(np.float64)
#wl2 = np.loadtxt("../data/04_06_2019/WaveFl.txt")
#wl = []
#for i in range(len(wl2)):
#    if i % 2 == 0:
#        wl.append(wl2[i])

if getivals:
    idxs = [i for i, x in enumerate(wl) if x > lim1 and x < lim2]

data = {}

for itm in listdir(datapath):
    if itm.endswith(".txt"):
        continue
    data[itm] = managedata.load_data(join('..', 'data', day_of_exp, itm))
    if getivals:
        ivals[itm] = managedata.getivals(data[itm], wl, idxs)
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
        xlim=[280, 800],
        #       ylim = [0,1]
        )

if do_save == 1:
    fig.savefig(sna + '.png', transparent=False, dpi=300, bbox_inches="tight")

test_std = np.std(list(ivals['Sea Water'].values()))

def func ( x, a, b ) :
    return a*x+b


conc1 = np.mean(list(ivals['0_2'].values()))
conc2 = np.mean(list(ivals['0_5'].values()))
conc3 = np.mean(list(ivals['1 new'].values()))

coefs, pcov = curve_fit(func,[1.,2.,3.], [conc1,conc2,conc3])

fig2, (ax2) = plt.subplots(1, 1, figsize=(18, 10))
fig2.set_tight_layout(True)


ax2.scatter([1.,2.,3.], [conc1,conc2,conc3], color='b')

ax2.plot([1.,2.,3.], func(np.array([1.,2.,3.]), *coefs), 'r-')
plt.show()


plt.show()