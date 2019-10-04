# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from os.path import join, isdir, abspath, isfile
from os import makedirs
from os import listdir

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score
# from statistics import mean
# from matplotlib import style
import datetime
# style.use('ggplot')
# import re

from somedataproc import managedata, processdata

do_save = 0
str1 = 'crude oil'
rad_type = 'LED_277'
day_of_exp = '04_10_2019'

datapath = join('..', 'data', day_of_exp, 'for plotting')

getivals = 1
norm_max = 0
norm_val = 0
smooth = 0
backgroundless = 0
legend = 1
average = 1
dyax = 0

if getivals:
    ivals = {}
    lim1 = 300
    lim2 = 525

if norm_val:
    nlim1 = 320
    nlim2 = 600

font = {'family': 'normal',
        'weight': 'bold',
        'size': 24}

matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"

res_dir = join('..', 'results', day_of_exp, str(datetime.date.today()))
sna = join(res_dir,rad_type + '_' + str1 + '_0')

if not isdir(res_dir):
    makedirs(abspath(res_dir))

# ----------------------------------------------------
for itm in listdir(datapath):
    if itm.endswith(".txt"):
        wl = np.loadtxt(join(datapath, itm ), dtype=np.str, usecols=0, skiprows=17, comments='>')
wl = np.char.replace(wl, ',', '.').astype(np.float64)

#wl = []
#for i in range(len(wl2)):
#    if i % 2 == 0:
#        wl.append(wl2[i])

if getivals:
    idxs = [i for i, x in enumerate(wl) if x > lim1 and x < lim2]

if norm_val:
    n_idxs = [i for i, x in enumerate(wl) if x > nlim1 and x < nlim2]

data = {}

itemslist = listdir(datapath)
itemslist.sort()
itemslist = managedata.natural_sort(itemslist)

for itm in itemslist:
    if not isdir(join(datapath,itm)):
        continue
    #data[itm] = managedata.load_data(join('..', 'data', day_of_exp, itm))
    data[itm] = managedata.load_data(join(datapath, itm))
    if getivals:
        ivals[itm] = managedata.getivals(data[itm], wl, idxs)
    if average:
        data[itm] = managedata.average(data[itm])
    if norm_max:
        data[itm] = managedata.norm_max(data[itm])
    if norm_val:
        data[itm] = managedata.norm_val(data[itm], wl, n_idxs)
    if smooth:
        data[itm] = managedata.smooth(data[itm])

# ----------------------------------------------------

fig, (ax1) = plt.subplots(1, 1, figsize=(18, 10))
if dyax == 1:
    ax2 = ax1.twinx()

fig.set_tight_layout(True)

for itm in data:
    if itm == '1(19)' or itm == 'RMG180' or itm == 'crude oil':
        continue
    for itm2 in data[itm]:
        ax1.plot(wl, data[itm][itm2])
#if legend == 1:
#    ax1.legend(list(data.keys()), loc=2)

if legend == 1:
#ax1.legend(['RMB80 film (80 mcm)','RMB80 solution (22.2 ppm)','RMB80'], loc=1)
    ax1.legend(list(data.keys()), loc=1)

if dyax == 1:
    for itm in data:
        if itm == '1(19)':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], 'm-')
        if itm == 'RMG380':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], color='maroon')
        if itm == 'RMG180':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], color = '0.75')
        if itm == 'crude oil':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], 'k-')
    if legend == 1:
        # ax2.legend(list(data.keys()), loc=0, bbox_to_anchor=(0.5, 0., 0.5, 0.8))
        #ax2.legend(list(data.keys())[4:8], loc=1)
        ax2.legend(['127.1 mg/l'], loc=1)

    ax2.set(xlabel='Wavelength, nm',
            ylabel='I, rel. un.',
            title=rad_type + ' ',
            xlim=[100, 700],
            #   ylim = [0,5000]
            )
if norm_val == 1 or norm_max == 1:
    ax1.set(xlabel='Wavelength, nm',
            ylabel='I, rel. un.',
            title= ' ' + ' ',
            xlim=[250, 900],
            ylim=[0, 1]
            )
else:
    ax1.set(xlabel='Wavelength, nm',
            ylabel='I, rel. un.',
            title= ' ' + ' ',
            xlim=[250, 700],
            #ylim = [0,4000]
            )



plt.show()

if do_save == 1:
    fig.savefig(sna + '.png', transparent=False, dpi=300, bbox_inches="tight")

#----- for lods
std = np.std(list(ivals['SeaWater'].values()))

def func(x, a, b):
    return a*x+b

# 19 - 127.1 , 130 - 133.9 , diesel - 83.7, rmb80 - 22.2, rmb80 (old) - 96, crude oil - 12.9

max_conc = 12.9
conc_vals = [max_conc/20, max_conc/10, max_conc/5, max_conc/2., max_conc]

sample = "crude oil"
conc1 = np.mean(list(ivals['0.64 ppm (crude oil )'].values()))
conc2 = np.mean(list(ivals['1.29 ppm (crude oil )'].values()))
conc3 = np.mean(list(ivals['2.58 ppm (crude oil )'].values()))
conc4 = np.mean(list(ivals['6.45 ppm (crude oil )'].values()))
conc5 = np.mean(list(ivals['12.9 ppm (crude oil )'].values()))

#conc6 = np.mean(list(ivals['4.8(RMB80)'].values()))
#conc7 = np.mean(list(ivals['9.6(RMB80)'].values()))
#conc8 = np.mean(list(ivals['19.2(RMB80)'].values()))
#conc9 = np.mean(list(ivals['48(RMB80)'].values()))
#conc10 = np.mean(list(ivals['96(RMB80)'].values()))



#max_conc2 = 96
#conc_vals2 = [max_conc2/20, max_conc2/10, max_conc2/5, max_conc2/2., max_conc2]

coefs, pcov = curve_fit(func,conc_vals, [conc1,conc2,conc3,conc4,conc5])

#coefs2, pcov2 = curve_fit(func,conc_vals2, [conc6,conc7,conc8,conc9,conc10])

fig2, (ax2) = plt.subplots(1, 1, figsize=(18, 10))
fig2.set_tight_layout(True)

lod = 3*std/coefs[0]

#lod2 = 3*std/coefs2[0]

r2 = r2_score([conc1,conc2,conc3,conc4,conc5], func(np.array(conc_vals), *coefs))
#r2_2 = r2_score([conc6,conc7,conc8,conc9,conc10], func(np.array(conc_vals2), *coefs2))

y_lod = func(lod, *coefs)
#y_lod2 = func(lod2, *coefs2)

ax2.scatter(conc_vals, [conc1,conc2,conc3,conc4,conc5], color='b')
#ax2.scatter(conc_vals2, [conc6,conc7,conc8,conc9,conc10], color='g')

ax2.plot(conc_vals, func(np.array(conc_vals), *coefs), 'r-')
#ax2.plot(conc_vals2, func(np.array(conc_vals2), *coefs2), 'k-')

if legend == 1:
    ax2.legend([ np.array2string(coefs[0], precision = 3)+ ' * x + ' + np.array2string(coefs[1], precision = 3) ,'data'], loc=0)

ax2.set(xlabel='Concentration, mg/l',
        ylabel='Int.values',
        title=str1
        #       ylim = [0,1]
        )

ax2.text(max_conc - max_conc/8,conc1,r'$R^2$' + ' = ' + np.array2string(r2, precision = 3), fontsize=24)
ax2.text(max_conc - max_conc/8,conc1 + conc1/2,'LOD' + ' = ' + np.array2string(lod, precision = 2), fontsize=24)

if do_save == 1:
    fig2.savefig(sna + '_' +'1' + '.png', transparent=False, dpi=300, bbox_inches="tight")

plt.show()