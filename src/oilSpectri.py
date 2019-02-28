#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join,isdir, abspath
from os import makedirs
from os import listdir

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from statistics import mean
from matplotlib import style
import datetime
#style.use('ggplot')

import re

from scipy import signal

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

do_save = 1
rad_type = ''
day_of_exp = '28_02_19'
tos = 'led'


norm_max = 0
smooth = 0
backgroundless = 0
legend = 0
average = 1

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)

plt.rcParams["axes.labelweight"] = "bold"

#if __name__ == '__main__':
#    dir = join('..','data',day_of_exp)
#else:
#    dir = join('data',day_of_exp)
dir = join('..','data',day_of_exp,tos)

#----------------------------

dir_d1 = dir
Data1 = {}
#wl = np.loadtxt(join(dir_d1,listdir(dir_d1)[0]),skiprows = 14, usecols = 0, comments='>')
wl= np.loadtxt(join(dir_d1,listdir(dir_d1)[0]), dtype=np.str, usecols = 0, skiprows=15)
wl = np.char.replace(wl, ',', '.').astype(np.float64)
        #wl = np.char.replace(wl, '\'', '')
        #wl = np.char.replace(wl, 'b', '')

for itm in natural_sort(listdir(dir_d1)):
    if itm.endswith(".txt"):
        #Data1[itm] = np.loadtxt(join(dir_d1,itm), skiprows=14,usecols = 1, comments='>')
        Data1[itm] = np.loadtxt(join(dir_d1,itm), dtype=np.str,skiprows=15,usecols = 1, comments='>')
        Data1[itm] = np.char.replace(Data1[itm], ',', '.').astype(np.float64)
    
res_dir = join('..','results',str(datetime.date.today()),day_of_exp)
sna = join(res_dir,tos)

if not isdir(res_dir):
    makedirs(abspath(res_dir))

fig, (ax1) = plt.subplots(1,1,figsize=(18,10))
fig.set_tight_layout(True)

if smooth == 1:
    for itm in Data1:
        Data1[itm] = signal.savgol_filter(Data1[itm], 11, 1)
        

if backgroundless == 1:
    for itm in Data1:
        Data1[itm] = Data1[itm] - np.mean(Data1[itm][100:200])

if norm_max == 1:
    for itm in Data1:
        Data1[itm] = Data1[itm]/max(Data1[itm])

if average == 1 :
    dictlist = []
    for itm in Data1 :
        dictlist.append(Data1[itm])
    Data1={}
    Data1['1'] = np.mean(dictlist[0:9], axis=0)
    Data1['2'] = np.mean(dictlist[10:-1], axis=0)



for itm in Data1:
    ax1.plot(wl,Data1[itm])

if legend == 1:
    ax1.legend(list(Data1.keys()),loc = 0)

ax1.set(xlabel = 'Wavelength, nm', 
       ylabel = 'I, units', 
       title = rad_type+' '+tos,
       xlim = [300,850],
#       ylim = [0,1]
       )    
       
if do_save == 1:
    fig.savefig(sna+'.png',transparent=False,dpi=300,bbox_inches="tight")

plt.show()
