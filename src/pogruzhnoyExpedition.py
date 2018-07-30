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

do_save = 0
rad_type = '527 nm'
tos = 'mwater'
day_of_exp = 'pogruzhnoyExp2018'

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 24}

matplotlib.rc('font', **font)

plt.rcParams["axes.labelweight"] = "bold"

#if __name__ == '__main__':
#    dir = join('..','data',day_of_exp)
#else:
#    dir = join('data',day_of_exp)
dir = join('..','data',day_of_exp)

#----------------------------

if tos == 'mwater':
    dir_f1 = dir
    fData1 = {}
    wl = np.loadtxt(join(dir_f1,listdir(dir_f1)[0]),usecols = 0, comments='>')
    for itm in listdir(dir_f1):
        if itm.endswith("zond.txt"):
            zondVals = np.loadtxt(join(dir_f1,itm), skiprows=1)
            continue
        if itm.endswith("callib_coefs_part.txt"):
            calibVals = np.loadtxt(join(dir_f1,itm),usecols = 0, comments='>')
            continue
        if itm.endswith(".txt"):
            fData1[itm] = np.loadtxt(join(dir_f1,itm),usecols = 1, comments='>')
    
ncfData1 = {}
for itm in fData1:
    ncfData1[itm] = fData1[itm]*calibVals
    ncfData1[itm] = ncfData1[itm]/max(ncfData1[itm][:100])
    


        
res_dir = join('..','results',str(datetime.date.today()),day_of_exp)
sna = join(res_dir,tos)

if not isdir(res_dir):
    makedirs(abspath(res_dir))
            
        
fData_plot = ncfData1


fig, (ax1) = plt.subplots(1,1,figsize=(18,10))
fig.set_tight_layout(True)

for itm in fData_plot:
    ax1.plot(wl,fData_plot[itm])

ax1.legend(list(fData_plot.keys()))

ax1.set(xlabel = 'Wavelength, nm', 
       ylabel = 'I, Relative units', 
       title = rad_type+' '+tos,
       xlim = [565,790],
       ylim = [0,1.2]
       )
if do_save == 1:
    fig.savefig(sna+'.jpg',transparent=False,dpi=300,bbox_inches="tight")       

plt.show()

fData_plot = zondVals


fig, (ax1) = plt.subplots(1,1,figsize=(18,10))
fig.set_tight_layout(True)

for itm in fData_plot[0][1:]:
    ax1.plot(fData_plot[:][0],fData_plot[itm])

ax1.legend(list(fData_plot.keys()))

ax1.set(xlabel = 'Wavelength, nm', 
       ylabel = 'I, Relative units', 
       title = rad_type+' '+tos,
       xlim = [565,790],
       ylim = [0,1.2]
       )
if do_save == 1:
    fig.savefig(sna+'.jpg',transparent=False,dpi=300,bbox_inches="tight")       

plt.show()
    

    
















