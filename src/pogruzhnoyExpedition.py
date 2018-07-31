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
rad_type = '527 nm'
tos = 'mwater'
day_of_exp = '2day'

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)

plt.rcParams["axes.labelweight"] = "bold"

#if __name__ == '__main__':
#    dir = join('..','data',day_of_exp)
#else:
#    dir = join('data',day_of_exp)
dir = join('..','data','pogruzhnoyExp2018',day_of_exp)

#----------------------------

if tos == 'mwater':
    dir_f1 = dir
    fData1 = {}
    wl = np.loadtxt(join(dir_f1,listdir(dir_f1)[0]),usecols = 0, comments='>')
    wl = np.linspace(min(wl)+11,max(wl)+11,263)
    for itm in natural_sort(listdir(dir_f1)):
        if itm.endswith("zond.txt"):
            zondVals = np.loadtxt(join(dir_f1,itm), skiprows=1)
            continue

        if itm.endswith(".txt"):
            fData1[itm] = np.loadtxt(join(dir_f1,itm),usecols = 1, comments='>')
    
nfData1 = {}
for itm in fData1:
    nfData1[itm] = fData1[itm]/max(fData1[itm][:100])
    
res_dir = join('..','results',str(datetime.date.today()),day_of_exp)
sna = join(res_dir,tos)

if not isdir(res_dir):
    makedirs(abspath(res_dir))
            
        



fig, (ax1) = plt.subplots(1,1,figsize=(18,10))
fig.set_tight_layout(True)

for itm in nfData1:
    ax1.plot(wl,nfData1[itm])
    
ax1.axvline(x=622, ymax=0.25, ls = '--')

ax1.legend(list(nfData1.keys()),loc = 2)

ax1.set(xlabel = 'Wavelength, nm', 
       ylabel = 'I, Relative units', 
       title = rad_type+' '+tos,
       xlim = [600,800],
       ylim = [0,1.2]
       )

a = plt.axes([.55, .45, .4, .4])

for itm in [1,2,3]:
    plt.plot(zondVals[:,itm],zondVals[:,0])
    
a.legend(['Temp','Fluor','CDOM'], loc=0)

a.set(xlabel = 'units',
      ylabel = 'Depth, m',
      title = 'zond',
      xlim = [0,max(zondVals[:,0])],
      ylim = [17,0]
      )
       
       
if do_save == 1:
    fig.savefig(sna+'.jpg',transparent=False,dpi=300,bbox_inches="tight")       

plt.show()











    

    
















