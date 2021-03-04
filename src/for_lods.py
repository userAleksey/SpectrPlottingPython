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
import datetime
# import re

from scipy import signal, stats


from somedataproc import managedata, processdata

def func(x, a, b):
    return a*x+b

do_save = 0
str1 = 'for_ml'
str2 = '_0'
rad_type = 'rad_type'
day_of_exp = 'films_19-24.02.2021'

xmin = 200
xmax = 1100

average = 0
filtr = 0
outliers = 1
normalize = 1
legend = 1

norm_max = 0
norm_val = 0
smooth = 0
backgroundless = 0

lod_legend = 0

dyax = 0
fitting = 0
something = 0
subrel = 0
edits = 0

select_items = 0

getivals = 0

y_val_for_sel = None
y_val2_for_sel = None

datapath = join('..', 'data', day_of_exp, 'for plotting', str1)

if getivals:
    ivals = {}
    lim1 = 330 #325  # 300 # 310
    lim2 = 370 #435  # 525 # 335

if norm_val or normalize:
    nlim1 = 225
    nlim2 = 600

font = {'family': 'normal',
        'weight': 'bold',
        'size': 24}

matplotlib.rc('font', **font)
plt.rcParams["axes.labelweight"] = "bold"

res_dir = join('..', 'results', day_of_exp, str(datetime.date.today()))
sna = join(res_dir,rad_type + '_' + day_of_exp + str1 + str2)

if not isdir(res_dir):
    makedirs(abspath(res_dir))

# ----------------------------------------------------
for itm in listdir(datapath):
    if not isdir(join(datapath,itm)):
        continue
    for itm2 in listdir(join(datapath,itm)):
        if itm2.endswith(".txt"):
            wl = np.loadtxt(join(datapath, itm, itm2 ), dtype=np.str, usecols=0, skiprows=17, comments='>')
            break
        if itm2.endswith(".tif") or itm2.endswith(".tiff"):
            wl = np.loadtxt(join(datapath, '..', 'wavelengths'), dtype=np.str, usecols=0, comments='>')
            tifs = 1
            break
    if 'wl' in locals():
        break
wl = np.char.replace(wl, ',', '.').astype(np.float64)

if 'tifs' in locals():
    wl2 = []
    for i in range(len(wl)):
        if i % 2 == 0:
            wl2.append(wl[i])
    wl = wl2

xlims_idxs = [i for i, x in enumerate(wl) if x > xmin and x < xmax]

if getivals:
    idxs = [i for i, x in enumerate(wl) if x > lim1 and x < lim2]

if norm_val or normalize:
    n_idxs = [i for i, x in enumerate(wl) if x > nlim1 and x < nlim2]

data = {}

itemslist = listdir(datapath)
itemslist.sort()
itemslist = managedata.natural_sort(itemslist)

datafilepath = join(datapath, 'data.npy')
if isfile(datafilepath):
    data2 = np.load(datafilepath, allow_pickle=True)
    data = data2.item()
else:
    for itm in itemslist:
        if not isdir(join(datapath,itm)):
            continue
        data[itm] = managedata.load_data(join(datapath, itm))
    np.save(datafilepath, data)

if select_items:
    y_val_for_sel = None
    y_val2_for_sel = None
    if itm == 'CleanWater':
        #y_val_for_sel = 600
        y_val_for_sel = None
        y_val2_for_sel = 1400
    if itm == 'FilmInWater(11_07 500Spectra)':
        y_val_for_sel = 700
        y_val2_for_sel = 1100
    if itm == 'Ground':
        #y_val_for_sel = 200
        y_val_for_sel = None
        y_val2_for_sel = 600
    if itm == 'Shell':
        y_val_for_sel = 400
        y_val2_for_sel = None
    if itm == 'CleanWater' and str1 == 'On Surface':
        y_val_for_sel = 15000
        y_val2_for_sel = None
    if itm == 'Film14_07SurfaceOneCascade':
        y_val_for_sel = 15000
        y_val2_for_sel = None
    if itm == 'FilmOnSurface(11_07)OneCascade':
        y_val_for_sel = 6000
        y_val2_for_sel = None
    if itm == 'Ground' and str1 == 'On Surface':
        y_val_for_sel = 5000
        y_val2_for_sel = None
    if itm == 'Shell' and str1 == 'On Surface':
        y_val_for_sel = 4000
        y_val2_for_sel = None
    if y_val2_for_sel:
        data[itm + '_655'] = managedata.load_data_with_sel(join(datapath, itm), y_val2_for_sel)
    if y_val_for_sel:
        data[itm] = managedata.load_data_with_sel(join(datapath, itm), y_val_for_sel)

if getivals:
    ivals[itm] = managedata.getivals(data[itm], wl, idxs)

if average:
    #data[itm] = managedata.average(data[itm])
    #if y_val_for_sel:
    #    data[itm] = managedata.average(data[itm])
    #if y_val2_for_sel:
    #    data[itm + '_655'] = managedata.average(data[itm + '_655'])

    for itm1 in data:
        data[itm1] = managedata.average(data[itm1])


#------!!! check for remove !!!
if smooth:
    if itm == 'RMB30 80 mcm':
        data[itm] = managedata.smooth(data[itm])
        for itm2 in data[itm]:
            data[itm][itm2][0:250] = data[itm][itm2][0:250]*0.5
    if itm == 'RMB30 slick (80 mcm)':
        data[itm] = managedata.smooth(data[itm])
        for itm2 in data[itm]:
            data[itm][itm2][0:250] = data[itm][itm2][0:250]*0.5
    if itm == 'DMA' and str1 == 'fig_12' and day_of_exp == 'statiyav2':
        for itm2 in data[itm]:
            data[itm][itm2][0:-1] = data[itm][itm2][0:-1]*0.1
if norm_max:
    data[itm] = managedata.norm_max(data[itm])
if norm_val:
    data[itm] = managedata.norm_val(data[itm], wl, n_idxs)
#------!!! check for remove !!!

# ----------------------------------------------------

# -------process data---------------------------------------------

if outliers:
    for itm1 in data:
        if itm1 != 'DMA_slick':
            continue
        for itm2 in data[itm1]:
            for count, value in enumerate(data[itm1][itm2]):
                if abs(data[itm1][itm2][count] - data[itm1][itm2][count - 1]) > 250:
                    data[itm1][itm2][count] = data[itm1][itm2][count - 2]
                    print(count)
            print(itm2)


if filtr == 1:
    filtrsignal = None
    filtr_name = 'Filter_Pet'
    for itm in data:
        if itm == filtr_name:
            for itm2 in data[itm]:
                filtrsignal = data[itm][itm2]

    filtrsignal = np.loadtxt(join(datapath, '..', filtr_name + '.txt'), dtype=np.str, skiprows=17, usecols=1, comments='>')
    filtrsignal = np.char.replace(filtrsignal, ',', '.').astype(np.float64)
    filtrsignal = 100.000 / filtrsignal
    for itm in data:
        if itm == filtr_name:
            continue
        else:
            if itm == '1 day (500 mcm)' or itm == 'DystillatePET300ms' or itm == 'DMA_slick':
                for itm2 in data[itm]:
                    #data[itm][itm2] = signal.savgol_filter(data[itm][itm2], 5, 1)
                    data[itm][itm2][265:-1] = data[itm][itm2][265:-1] * filtrsignal[265:-1]
                    #data[itm][itm2] = data[itm][itm2] * filtrsignal

if subrel == 1:
    for itm in data:
        if itm == 'SeaWater' or itm == 'seawater' or itm == 'sea water':
            for itm2 in data[itm]:
                backgroundsignal = data[itm][itm2]
    #for itm in data['seawater']:
    #    backgroundsignal = data['seawater'][itm]
        if itm == 'seawater20':
            for itm2 in data[itm]:
                backgroundsignal2 = data[itm][itm2]
    #for itm in data['seawater20']:
    #        backgroundsignal2 = data['seawater20'][itm]
        if itm == 'DystillatePET300ms':
            for itm2 in data[itm]:
                backgroundsignal3 = data[itm][itm2]

    for itm in data:
        if itm == 'seawater' or itm == 'SeaWater' or itm == 'sea water':
            continue
        for itm2 in data[itm]:
            data[itm][itm2] = data[itm][itm2] - backgroundsignal
        continue
        if itm == '_RMB30 80 mcm':
            data[itm] = managedata.norm_val(data[itm], wl, xlims_idxs)
            continue
        if itm == '20 mcm 3 day' or itm == '20 mcm':
            for itm2 in data[itm]:
                data[itm][itm2] = data[itm][itm2] - backgroundsignal2
        if itm == 'DMA slick (100 mcm) 3 days':
            for itm2 in data[itm]:
                data[itm][itm2] = data[itm][itm2] - backgroundsignal
        if itm != 'DystillatePET300ms' or itm != 'pet':
            for itm2 in data[itm]:
                data[itm][itm2] = data[itm][itm2] - backgroundsignal3

        if smooth:
            if itm == '20 mcm 3 day' or itm == '100 mcm 3 day' or itm == '300 mcm 3 day' or itm == '500 mcm 3 day':
                for itm2 in data[itm]:
                    data[itm][itm2][0:255] = data[itm][itm2][0:255] * 0.1

            if itm == '20 mcm' or itm == '100 mcm' or itm == '300 mcm' or itm == '500 mcm':
                for itm2 in data[itm]:
                    data[itm][itm2][0:255] = data[itm][itm2][0:255] * 0.03

            if itm == 'DIESEL 20 mcm' or itm == 'DMA 20 mcm' or itm == 'DMZ 20 mcm' or itm == 'DMA slick (100 mcm) 3 days':
                for itm2 in data[itm]:
                    value = 0.0025
                    numcel = 0
                    for itm3 in data[itm][itm2][0:250]:
                        data[itm][itm2][numcel] = itm3 * value
                        value = value + 0.000
                        numcel = numcel + 1
                    value = 0.015
                    numcel = 0
                    for itm3 in data[itm][itm2][250:290]:
                        data[itm][itm2][250 + numcel] = itm3 * value
                        value = value + 0.025
                        numcel = numcel + 1




if normalize == 1:
    for itm in data:
        if itm == 'seawater' or itm == 'RMB30 (80 mcm)':
            continue
        data[itm] = managedata.norm_val(data[itm], wl, n_idxs)



#for itm in data:
#    datatxtpath = join(datapath, itm + '.txt')
#    test_dma = data.get(itm).values()
#    np.savetxt(datatxtpath, np.array(list(test_dma)))

datafilepath1 = join(datapath, 'data1.npy')
np.save(datafilepath1, data)

#test1 = np.load(datafilepath1, allow_pickle=True)
#test2 = test1.item()

if edits == 1:
    ### NEED finish
    #if itm == 'DMA slick (100 mcm) 3 days' and str1 == 'fig_5' and day_of_exp == 'statiyav2':
    if itm == '20 mcm' or itm == '300 mcm' or itm == '500 mcm':
        pure = np.linspace(data[itm][itm2][225], data[itm][itm2][290], 65)
        noise = np.random.normal(0, 5.1, pure.shape)
        signal = pure + noise
        arrray = data[itm][itm2][0:226]
        arrray = np.append(arrray,signal)
        arrray = np.append(arrray,data[itm][itm2][290:-1])
    #    ax1.plot(wl, arrray  - 0, linewidth=4)
    #    continue

    if itm == '100 mcm':
        pure = np.linspace(data[itm][itm2][225], data[itm][itm2][290], 65)
        noise = np.random.normal(0, 5.1, pure.shape)
        signal = pure + noise
        pure2 = np.linspace(data[itm][itm2][435], data[itm][itm2][444], 9)
        noise2 = np.random.normal(0, 5.1, pure2.shape)
        signal2 = pure2 + noise2
        arrray = data[itm][itm2][0:226]
        arrray = np.append(arrray,signal)
        arrray = np.append(arrray,data[itm][itm2][290:435])
        arrray = np.append(arrray, signal2)
        arrray = np.append(arrray,data[itm][itm2][444:-1])
    #    ax1.plot(wl, arrray  - 0, linewidth=4)
    #    continue

# -------process data---------------------------------------------

# ------plotting--------------------------------------------------

fig, (ax1) = plt.subplots(1, 1, figsize=(19, 11))
if dyax == 1:
    ax2 = ax1.twinx()

fig.set_tight_layout(True)

for itm in data:
    if itm == '_1(19)' or itm == '_RMG180' or itm == '_crude oil':
        continue
    if filtr == 1:
        if itm == 'pet':
            continue
    for itm2 in data[itm]:
        if itm == 'DMA 6 ppm':
            ax1.plot(wl, data[itm][itm2], 'royalblue', linewidth=4)
            continue
        if itm == 'DMA 6 ppm (5d)':
            ax1.plot(wl, data[itm][itm2], '--', color='royalblue', linewidth=4)
            continue
        if itm == 'DMA 63 ppm':
            ax1.plot(wl, data[itm][itm2], 'r', linewidth=4)
            continue
        if itm == '42 ppm':
            ax1.plot(wl, data[itm][itm2], 'r', linewidth=4)
            continue
        if itm == 'DMA 63 ppm (5d)':
            ax1.plot(wl, data[itm][itm2], '--r', linewidth=4)
            continue
        if itm == 'DMA 127 ppm':
            ax1.plot(wl, data[itm][itm2], 'mediumorchid', linewidth=4)
            continue
        if itm == 'sea water + sun':
            ax1.plot(wl, data[itm][itm2], 'sienna', linewidth=4)
            continue
        if itm == 'DMA 127 ppm (5d)':
            ax1.plot(wl, data[itm][itm2], '--', color='mediumorchid', linewidth=4)
            continue
        if itm == 'Sea Water':
            ax1.plot(wl, data[itm][itm2], 'sienna', linewidth=4)
            continue

        if subrel == 1:
            if itm == 'seawater' or itm == 'seawater20' or itm == 'SeaWater' or itm == 'DystillatePET300ms':
                continue
        if itm == '20 mcm':
            ax1.plot(wl, data[itm][itm2] - 0, linewidth=4)
            continue
        if itm == 'RMB30 80 mcm':
            ax1.plot(wl, data[itm][itm2] - 0, linewidth=4)
            continue

        if itm == 'RMB30' or itm == 'RME380' or itm == 'RMG180' or itm == 'crude oil':
            continue

        data[itm][itm2] = data[itm][itm2] - 0
        ax1.plot(wl, data[itm][itm2], linewidth=4)

#if legend == 1:
#    ax1.legend(list(data.keys()), loc=2)

if legend == 1:
    legendset1 = []
    for itm in list(data.keys()):
        if itm == 'RMB30' or itm == 'RME380' or itm == 'RMG180' or itm == 'crude oil' or itm == 'pet':
            continue
        if itm.find(' 3_days') != -1:
            itm = itm.replace(' 3 days', ' ')
        if itm.find(' 3_day') != -1:
            itm = itm.replace(' 3 day', ' ')
        if itm.find('mcm') != -1:
            test1 = itm.replace('mc', r'$\mu$')
            legendset1.append(test1)
        else:
            legendset1.append(itm)

    ax1.legend(legendset1, loc=0,fancybox=True, facecolor='white', frameon=False)

if dyax == 1:
    for itm in data:
        if itm == '1(19)':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], 'm-')

        if itm == 'RMB30':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], color='C4', linestyle = '--', linewidth=4)
        if itm == 'RME380':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], color='C5', linestyle = '--', linewidth=4)
        if itm == 'RMG180':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], color='C6', linestyle = '--', linewidth=4)
        if itm == 'crude oil':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], color='C7', linestyle = '--', linewidth=4)
        if itm == 'RMB80 slick 80 mcm':
            for itm2 in data[itm]:
                ax2.plot(wl, data[itm][itm2], linewidth=4)

    if legend == 1:
        legendset2 = []
        for itm in list(data.keys()):
            if itm == 'auto diesel oil' or itm == 'DMA' or itm == 'DMZ' or itm == 'kerosene':
                continue
            if itm.find(' 3_days') != -1:
                itm = itm.replace(' 3 days', ' ')
            if itm.find(' 3_day') != -1:
                itm = itm.replace(' 3 day', ' ')
            if itm.find('mcm') != -1:
                test1 = itm.replace('mc', r'$\mu$')
                legendset2.append(test1)
            else:
                legendset2.append(itm)

        ax2.legend(legendset2, loc=1, fancybox=True, facecolor='white', frameon=False)



    ax2.set(xlabel=r'$\lambda$' + ', nm',
            ylabel='I, rel. un.',
            title= ' ' + ' ',
            xlim=[xmin, xmax],
               ylim = [0,5350]
            )
    ax2.ticklabel_format(axis='y', style='sci', scilimits=(3,3), useMathText=True)
    ax2.yaxis.offsetText.set_visible(False)
    #ax1.legend(legendset1, loc=5, fancybox=True)
if norm_val == 1 or norm_max == 1 or normalize:
    ax1.set(xlabel=r'$\lambda$' + ', nm',
            ylabel='I, norm',
            title= ' ' + ' ',
            xlim=[xmin, xmax],
            ylim=[0, 1]
            )

else:
    allymax = []
    for itm1 in data:
        if itm1 == '_DMA' or itm1 == 'seawater' or itm1 == 'seawater20' or itm1 == 'SeaWater':
            continue
        for itm2 in data[itm1]:
            allymax.append(max(data[itm1][itm2][xlims_idxs[0]:xlims_idxs[-1]]))
    ymax1 = max(allymax)
    ax1.set(xlabel=r'$\lambda$' + ', nm',
            ylabel='I, rel. un.',
            title= ' ' + ' ',
            xlim=[xmin,xmax],
            ylim = [-21.13, ymax1]
            #ylim = [0, 1100]
            )

    ax1.ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
    ax1.yaxis.offsetText.set_visible(False)

for tick in ax1.yaxis.get_majorticklabels():
    tick.set_verticalalignment("bottom")

# magic specs -------------------------

#if xmax == 600:
#    ax1.set_xticks([xmin, xmin + 50, xmin + 150, xmin + 250, xmin + 350])
#else:
#    ax1.set_xticks([xmin, xmin + 50, xmin + 150, xmin + 250, xmin + 350, xmin + 450, xmin + 550])

ax1.set_ylabel('I, rel. un.', rotation=0, fontsize=20, labelpad=20)
ax1.yaxis.set_label_coords(0.05,1.00)

if dyax == 1:
    ax1.set_ylabel('I, rel. un.(light fuel)', rotation=0, fontsize=20, labelpad=20)
    ax2.set_ylabel('I, rel. un.(heavy fuel)', rotation=0, fontsize=20, labelpad=20)
    ax2.yaxis.set_label_coords(0.9, 1.03)

    for tick in ax2.yaxis.get_majorticklabels():
        tick.set_verticalalignment("bottom")

#ax1.axvline(x=345.895,linestyle='--',color='black',ymax=0.85)
#ax1.text(345.9,400,'346', fontsize=24)

# magic specs -------------------------

plt.show()

if do_save == 1:
    fig.savefig(sna + '.png', transparent=False, dpi=300, bbox_inches="tight")
#------------something1
if something:
    n_str1 = '6.69 ppm'
    n_str2 = '6.69 ppm'
    for itm in data[n_str1]:
        y = data[n_str1][itm]

    bnds_lim1 = 340
    bnds_lim2 = 355
    bnds_idxs = [i for i, x in enumerate(wl) if x > bnds_lim1 and x < bnds_lim2]
    abs_max = max(y[bnds_idxs[0]:bnds_idxs[-1]])
    abs_min = min(y[bnds_idxs[0]:bnds_idxs[-1]])
    print('abs_min: ' + str(abs_min))
    min_idx = [i for i, x in enumerate(y) if x == abs_min]
    print('min_idx: ' + str(min_idx))
    abs_val = abs_max - abs_min
    print('abs_val = ' + str(abs_val))
    #val1 = (abs_val - 526.6857) / 20.6157
    val1 = (abs_val - 434.2891) / 27.758

    print('val1 = ' + str(val1))

    d_abs = 3093.9285714285725 / 1194.2666666666673

    # dma : -19.2904291389572
    # dmz : 2.2743317241876073
    # dma + dmz: -5.529544635780678

    for itm in ivals[n_str2]:
        yi = ivals[n_str2][itm]
    #val2 = (yi - 404553.151) / 28215.851
    val2 = (yi - 402769.829) / 42837.715
    print(yi)
    print(val2)

    for itm in ivals[n_str1]:
        alpha1 = ivals[n_str1][itm]
    for itm in ivals[n_str2]:
        alpha2 = ivals[n_str2][itm]
    d_alpha = alpha1 / alpha2
    print('d_alpha = ' + str(d_alpha))
    print('d_abs = ' + str(d_abs))

    # dma: -9.392502533416414
    # dmz: 0.8360391671217748
    # dma + dmz: -3.0903133154978044
#-----------------------------------

#----- fitting
if fitting:
    for itm in data[n_str1]:
        y = data[n_str1][itm]

    #import pylab as plb
    from scipy.optimize import curve_fit
    from scipy import asarray as ar,exp

    n = len(wl)
    #max = max(y)#the number of data
    minA = 3800#the number of data
    maxA = 3900
    #mean = sum(wl*y)/n                   #note this correction
    mean = 335                 #note this correction
    #sigma = sum(y*(wl-mean)**2)/n        #note this correction
    sigma = 20      #note this correction

    def gaus(x,a,x0,sigma):
        return a*exp(-(x-x0)**2/(2*sigma**2))

    def _1gaussian(x, amp1, cen1, sigma1):
        return amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2)))

    def _2gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2):
        return amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2))) + \
               amp2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen2) / sigma2) ** 2)))

    def _3gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3):
        return amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2))) + \
               amp2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen2) / sigma2) ** 2))) + \
               amp3 * (1 / (sigma3 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen3) / sigma3) ** 2)))

    def _4gaussian(x, amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3, amp4, cen4, sigma4):
        return amp1 * (1 / (sigma1 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen1) / sigma1) ** 2))) + \
               amp2 * (1 / (sigma2 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen2) / sigma2) ** 2))) + \
               amp3 * (1 / (sigma3 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen3) / sigma3) ** 2))) + \
               amp4 * (1 / (sigma4 * (np.sqrt(2 * np.pi)))) * (np.exp((-1.0 / 2.0) * (((x - cen4) / sigma4) ** 2)))

    amp1 = 20000
    sigma1 = 5
    cen1 = 200

    amp2 = 20000
    sigma2 = 5
    cen2 = 200

    amp3 = 20000
    sigma3 = 5
    cen3 = 200

    amp4 = 20000
    sigma4 = 5
    cen4 = 200

    #popt_1gauss, pcov_1gauss = curve_fit(_1gaussian, wl, y, p0=[amp1, cen1, sigma1])
    #popt_3gauss, pcov_3gauss = curve_fit(_3gaussian, wl, y, p0=[amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3])
    popt_2gauss, pcov_2gauss = curve_fit(_2gaussian, wl, y, p0=[amp1, cen1, sigma1, amp2, cen2, sigma2],bounds = ([0, 200, 1,0, 200, 1], \
                                                  [5000000,800,250,5000000,800,250]),maxfev=10000)
    #popt_4gauss, pcov_4gauss = curve_fit(_4gaussian, wl, y, p0=[amp1, cen1, sigma1, amp2, cen2, sigma2, amp3, cen3, sigma3, amp4, cen4, sigma4], \
    #                                     bounds = ([0, 200, 1,0, 200, 1,0, 200, 1,0, 200, 1], \
    #                                               [500000,800,25,500000,800,25,500000,800,25,500000,800,25]),maxfev=10000)
    #perr_3gauss = np.sqrt(np.diag(pcov_3gauss))
    perr_2gauss = np.sqrt(np.diag(pcov_2gauss))
    #perr_4gauss = np.sqrt(np.diag(pcov_4gauss))
    #perr_1gauss = np.sqrt(np.diag(pcov_1gauss))

    print("amplitude1 = %0.2f (+/-) %0.2f" % (popt_2gauss[0], perr_2gauss[0]))
    print("center1 = %0.2f (+/-) %0.2f" % (popt_2gauss[1], perr_2gauss[1]))
    print("sigma1 = %0.2f (+/-) %0.2f" % (popt_2gauss[2], perr_2gauss[2]))

    print("amplitude2 = %0.2f (+/-) %0.2f" % (popt_2gauss[3], perr_2gauss[3]))
    print("center2 = %0.2f (+/-) %0.2f" % (popt_2gauss[4], perr_2gauss[4]))
    print("sigma2 = %0.2f (+/-) %0.2f" % (popt_2gauss[5], perr_2gauss[5]))

    pars_1 = popt_2gauss[0:3]
    pars_2 = popt_2gauss[3:6]
    #pars_3 = popt_4gauss[6:9]
    #pars_4 = popt_4gauss[9:12]
    gauss_peak_1 = _1gaussian(wl, *pars_1)
    gauss_peak_2 = _1gaussian(wl, *pars_2)
    #gauss_peak_3 = _1gaussian(wl, *pars_3)
    #gauss_peak_4 = _1gaussian(wl, *pars_4)

    #ax1.plot(wl, _3gaussian(wl, *popt_3gauss), 'k--')
    #ax1.plot(wl, _1gaussian(wl, *popt_1gauss), 'k--')
    ax1.plot(wl, _2gaussian(wl, *popt_2gauss), 'k--')

    ax1.plot(wl, gauss_peak_1, "g")
    ax1.fill_between(wl, gauss_peak_1.min(), gauss_peak_1, facecolor="green", alpha=0.5)

    ax1.plot(wl, gauss_peak_2, "y")
    ax1.fill_between(wl, gauss_peak_2.min(), gauss_peak_2, facecolor="yellow", alpha=0.5)

    #ax1.plot(wl, gauss_peak_3, "b")
    #ax1.fill_between(wl, gauss_peak_3.min(), gauss_peak_3, facecolor="blue", alpha=0.5)

    #ax1.plot(wl, gauss_peak_4, "r")
    #ax1.fill_between(wl, gauss_peak_4.min(), gauss_peak_4, facecolor="red", alpha=0.5)

    popt,pcov = curve_fit(gaus,wl,y,p0=[minA,mean,sigma],bounds = ([minA, 335, 20],[maxA,341,21]))

    plt.plot(wl,y,'b+:',label='data')
    plt.plot(wl,gaus(wl,*popt),'ro:',label='fit')
    plt.legend()
    plt.title('Fig. 3 - Fit ')
    plt.xlabel('Wavelength, nm')
    plt.ylabel('I, rel. un.')
    plt.xlim(300, 400)

    plt.show()

    #abs_max1 = gaus(wl[min_idx],*popt)
    abs_max1 = _2gaussian(wl[min_idx],*popt_2gauss)
    print('abs_max1: ' + str(abs_max1))

    #conc5 = 27663.15286988 - 20358.72642857143
    conc5 = 23220.03910999 - 20358.72642857143
    #conc4 = 17415.26739332 - 12444.878666666667
    conc4 = 14326.37876527 - 12444.878666666667
    #conc3 = 11610.17826222 - 8311.06185185185
    conc3 = 9518.28733114 - 8311.06185185185
    #conc2 = 6772.60398629 - 4522.166551724138
    conc2 = 5169.37227701 - 4522.166551724138
    #conc1 = 3795.98106328 - 2926.1040740740746
    conc1 = 3344.02459665 - 2926.1040740740746


    max_conc = 127.1
    conc_vals = [max_conc / 20, max_conc / 10, max_conc / 5, max_conc / 2., max_conc]
    coefs, pcov = curve_fit(func, conc_vals, [conc1, conc2, conc3, conc4, conc5])

    fig2, (ax2) = plt.subplots(1, 1, figsize=(18, 10))
    fig2.set_tight_layout(True)

    r2 = r2_score([conc1, conc2, conc3, conc4, conc5], func(np.array(conc_vals), *coefs))

    ax2.scatter(conc_vals, [conc1, conc2, conc3, conc4, conc5], color='b')

    ax2.plot(conc_vals, func(np.array(conc_vals), *coefs), 'r-')

    ax2.set(xlabel='Concentration, mg/l',
            ylabel=r'$\alpha$',
            title=str1,
            #       ylim = [0,1]
            )

    ax2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    ax2.text(max_conc - max_conc / 8, conc1, r'$R^2$' + ' = ' + np.array2string(r2, precision=3), fontsize=24)

    plt.show()
#--------------

#----- for lods

# 19(dma) - 127.1 , 130(dmz) - 133.9 , diesel - 83.7, rmb80 - 22.2, rmb80 (old) - 96, crude oil - 12.9, ...
# rmg380 - 3.4, kerosin - 142.2, rmb180 - 2.4, dma - 3.3 (-->42)

max_conc = 42
conc_vals = [2.1, 4.2, 21, 42]
#conc_vals = [max_conc/20, max_conc/10, max_conc/5, max_conc/2., max_conc]
#conc_vals = [max_conc*0.04, max_conc*0.05, max_conc*0.2, max_conc*0.3, max_conc*0.4, max_conc*0.5, max_conc*0.7, max_conc*0.85,max_conc]
#conc_vals = [max_conc*0.04, max_conc*0.05, max_conc*0.1, max_conc*0.2, max_conc*0.3, max_conc*0.4, max_conc*0.5, max_conc*0.7, max_conc*0.85,max_conc]
#conc_vals = [max_conc/10, max_conc/5, max_conc/2., max_conc]

std = np.std(list(ivals['SeaWater'].values()))

#conc1 = 238.26666666666665
#conc2 = 339.1666666666665
#conc3 = 803.4666666666672
#conc4 = 1478.9000000000015
#conc5 = 3093.166666666666

conc1 = np.mean(list(ivals['2.1 ppm'].values()))
conc2 = np.mean(list(ivals['4.2 ppm'].values()))
conc3 = np.mean(list(ivals['21 ppm'].values()))
conc4 = np.mean(list(ivals['42 ppm'].values()))
#conc5 = np.mean(list(ivals['16.8 ppm'].values()))
#conc6 = np.mean(list(ivals['21 ppm'].values()))
#conc7 = np.mean(list(ivals['29.4 ppm'].values()))
#conc8 = np.mean(list(ivals['35.7 ppm'].values()))
#conc9 = np.mean(list(ivals['42 ppm'].values()))
#conc10 = np.mean(list(ivals['3.3 ppm'].values()))

#all_concs = [conc1,conc2,conc3,conc4,conc5, conc6, conc7, conc8,conc9,conc10]
all_concs = [conc1,conc2,conc3,conc4]
#all_concs = [conc1,conc2,conc3,conc4,conc5]

#conc6 = np.mean(list(ivals['4.8(RMB80)'].values()))
#conc7 = np.mean(list(ivals['9.6(RMB80)'].values()))
#conc8 = np.mean(list(ivals['19.2(RMB80)'].values()))
#conc9 = np.mean(list(ivals['48(RMB80)'].values()))
#conc10 = np.mean(list(ivals['96(RMB80)'].values()))

#max_conc2 = 96
#conc_vals2 = [max_conc2/20, max_conc2/10, max_conc2/5, max_conc2/2., max_conc2]

coefs, pcov = curve_fit(func,conc_vals, all_concs)

#coefs2, pcov2 = curve_fit(func,conc_vals2, [conc6,conc7,conc8,conc9,conc10])

fig2, (ax2) = plt.subplots(1, 1, figsize=(19, 11))
fig2.set_tight_layout(True)

lod = 3*std/coefs[0]

#lod2 = 3*std/coefs2[0]

r2 = r2_score(all_concs, func(np.array(conc_vals), *coefs))
#r2_2 = r2_score([conc6,conc7,conc8,conc9,conc10], func(np.array(conc_vals2), *coefs2))

y_lod = func(lod, *coefs)
#y_lod2 = func(lod2, *coefs2)

ax2.scatter(conc_vals, all_concs, color='b')
#ax2.scatter(conc_vals2, [conc6,conc7,conc8,conc9,conc10], color='g')

ax2.plot(conc_vals, func(np.array(conc_vals), *coefs), 'r-', linewidth=2.0)
#ax2.plot(conc_vals2, func(np.array(conc_vals2), *coefs2), 'k-')

if lod_legend == 1:
    ax2.legend([ np.array2string(coefs[0], precision = 3)+ ' * x + ' + np.array2string(coefs[1], precision = 3) ,'data'], loc=0)

ax2.set(xlabel='Concentration, mg/l',
        ylabel=r'$\alpha$',
        #ylabel=r'$\epsilon$',
        title= '' ,
        #       ylim = [0,1]
        )

ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.text(max_conc - max_conc/8,conc1,r'$R^2$' + ' = ' + np.array2string(r2, precision = 2), fontsize=24)
#ax2.text(max_conc - max_conc/8,all_concs[0],r'$R^2$' + ' = ' + '0.99', fontsize=24)
ax2.text(max_conc - max_conc/8,(all_concs[-1] - all_concs[0])/10 + all_concs[0],'LoD' + ' = ' + np.array2string(lod, precision = 2), fontsize=24)

ax2.ticklabel_format(axis='y', style='sci', scilimits=(3, 3), useMathText=True)
ax2.yaxis.offsetText.set_visible(False)

for tick in ax2.yaxis.get_majorticklabels():
    tick.set_verticalalignment("bottom")

plt.show()

if do_save == 1:
    fig2.savefig(sna + '_' +'1' + '.png', transparent=False, dpi=300, bbox_inches="tight")