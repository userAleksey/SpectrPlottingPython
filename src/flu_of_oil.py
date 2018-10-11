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

def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

do_save = 1
rad_type = '266 nm'
day_of_exp = '12_10_2018'
tos = 'hfo'

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
wl = np.loadtxt(join(dir_d1,listdir(dir_d1)[0]),usecols = 0, comments='>')
for itm in listdir(dir_d1):
    if itm.endswith(".txt"):
        Data1[itm] = np.loadtxt(join(dir_d1,itm),usecols = 1, comments='>')
    
res_dir = join('..','results',str(datetime.date.today()),day_of_exp)
sna = join(res_dir,tos)

if not isdir(res_dir):
    makedirs(abspath(res_dir))

fig, (ax1) = plt.subplots(1,1,figsize=(18,10))
fig.set_tight_layout(True)

for itm in Data1:
    ax1.plot(wl,Data1[itm])

ax1.legend(list(Data1.keys()),loc = 0)

ax1.set(xlabel = 'Wavelength, nm', 
       ylabel = 'I, units', 
       title = rad_type+' '+tos,
       xlim = [250,500],
 #      ylim = [0,]
       )    
       
if do_save == 1:
    fig.savefig(sna+'.jpg',transparent=False,dpi=300,bbox_inches="tight")       

plt.show()











    

    
















