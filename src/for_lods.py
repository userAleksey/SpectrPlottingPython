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

do_save = 1
str1 = 'dma'
rad_type = 'LED_278'
day_of_exp = '20_11_2019'

datapath = join('..', 'data', day_of_exp, 'for plotting')

getivals = 1
norm_max = 0
norm_val = 0
smooth = 0
backgroundless = 0
legend = 1
lod_legend = 0
average = 1
dyax = 0
fitting = 0
something = 0

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
        #wl = np.loadtxt(join(datapath, itm ), dtype=np.str, usecols=0, comments='>')
wl = np.char.replace(wl, ',', '.').astype(np.float64)

# for tifs
#wl2 = []
#for i in range(len(wl)):
#    if i % 2 == 0:
#        wl2.append(wl[i])
#wl = wl2

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
    if smooth:
        if itm == 'RMB80 slick (80 mcm)':
            data[itm] = managedata.smooth(data[itm])
    if norm_max:
        data[itm] = managedata.norm_max(data[itm])
    if norm_val:
        data[itm] = managedata.norm_val(data[itm], wl, n_idxs)


# ----------------------------------------------------

fig, (ax1) = plt.subplots(1, 1, figsize=(18, 10))
if dyax == 1:
    ax2 = ax1.twinx()

fig.set_tight_layout(True)

for itm in data:
    if itm == '_1(19)' or itm == '_RMG180' or itm == '_crude oil':
        continue
    for itm2 in data[itm]:
        if itm == 'DMA 6 ppm':
            ax1.plot(wl, data[itm][itm2], 'r')
            continue
        if itm == 'DMA 6 ppm (5d)':
            ax1.plot(wl, data[itm][itm2], '--r')
            continue
        if itm == 'DMA 63 ppm':
            ax1.plot(wl, data[itm][itm2], 'y')
            continue
        if itm == 'DMA 63 ppm (5d)':
            ax1.plot(wl, data[itm][itm2], '--y')
            continue
        if itm == 'DMA 127 ppm':
            ax1.plot(wl, data[itm][itm2], 'g')
            continue
        if itm == 'DMA 127 ppm (5d)':
            ax1.plot(wl, data[itm][itm2], '--g')
            continue
        ax1.plot(wl, data[itm][itm2])

#if legend == 1:
#    ax1.legend(list(data.keys()), loc=2)

if legend == 1:
    #ax1.legend(['DMA','DMA film (80 '+ r'$\mu$' + 'm)','DMA solution (127.1 ppm)'], loc=1)
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
            xlim=[250, 600],
            ylim=[0, 1]
            )
else:
    ax1.set(xlabel='Wavelength, nm',
            ylabel='I, rel. un.',
            title= ' ' + ' ',
            xlim=[200, 700],
            #ylim = [0,4000]
            )


ax1.ticklabel_format(axis='y', style='sci', scilimits=(3,3))
plt.show()

#------------something1
if something:
    n_str1 = '0_05(19)'
    n_str2 = '1(19)'
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
    print(abs_val)
    #val1 = (abs_val - 526.6857) / 20.6157
    val1 = (abs_val - 434.2891) / 27.758

    print(val1)

    # dma : -19.2904291389572
    # dmz : 2.2743317241876073
    # dma + dmz: -5.529544635780678

    for itm in ivals[n_str2]:
        yi = ivals[n_str2][itm]
    #val2 = (yi - 404553.151) / 28215.851
    val2 = (yi - 402769.829) / 42837.715
    print(yi)
    print(val2)

    # dma: -9.392502533416414
    # dmz: 0.8360391671217748
    # dma + dmz: -3.0903133154978044
#-----------------------------------

if do_save == 1:
    fig.savefig(sna + '.png', transparent=False, dpi=300, bbox_inches="tight")

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


    def func(x, a, b):
        return a * x + b

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
std = np.std(list(ivals['SeaWater'].values()))

def func(x, a, b):
    return a*x+b

# 19 - 127.1 , 130 - 133.9 , diesel - 83.7, rmb80 - 22.2, rmb80 (old) - 96, crude oil - 12.9, ...
# rmg380 - 3.4, kerosin - 142.2, rmb180 - 2.4

max_conc = 2.4
conc_vals = [max_conc/20, max_conc/10, max_conc/5, max_conc/2., max_conc]


conc1 = np.mean(list(ivals['0.12 ppm'].values()))
conc2 = np.mean(list(ivals['0.24 ppm'].values()))
conc3 = np.mean(list(ivals['0.48 ppm'].values()))
conc4 = np.mean(list(ivals['1.2 ppm'].values()))
conc5 = np.mean(list(ivals['2.4 ppm'].values()))

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

if lod_legend == 1:
    ax2.legend([ np.array2string(coefs[0], precision = 3)+ ' * x + ' + np.array2string(coefs[1], precision = 3) ,'data'], loc=0)

ax2.set(xlabel='Concentration, mg/l',
        ylabel=r'$\alpha$',
        title=str1,
        #       ylim = [0,1]
        )

ax2.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
ax2.text(max_conc - max_conc/8,conc1,r'$R^2$' + ' = ' + np.array2string(r2, precision = 3), fontsize=24)
ax2.text(max_conc - max_conc/8,conc1 + conc1/2,'LOD' + ' = ' + np.array2string(lod, precision = 2), fontsize=24)

if do_save == 1:
    fig2.savefig(sna + '_' +'1' + '.png', transparent=False, dpi=300, bbox_inches="tight")

plt.show()