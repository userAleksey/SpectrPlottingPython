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
rad_type = 'Femto'
tos = 'TCM'
day_of_exp = '20_06 Oil Convert'
dir_f0_name = 'Sea1'

f0_names = []
# f1_names = []

font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

matplotlib.rc('font', **font)

if __name__ == '__main__':
    dir = join('..','data',day_of_exp)
else:
    dir = join('data',day_of_exp)

#-------calibration_kuvet
try:
    kalib_values = np.ones(2068)
    initial = np.loadtxt(join('..','data','Initial.txt'), skiprows=17,usecols = 1, comments='>')  
    kuvete_calib = np.loadtxt(join('..','data','kuvete calib.txt'), skiprows=17,usecols = 1, comments='>')
    kalib_values[195:] = 100.0/kuvete_calib[195:]
except OSError:
    pass

#kalib_values = np.ones(2068)
#----------------------------
    
if day_of_exp == '19_6 Oil Convert':  
    KR_Film = {}
    for file in listdir(join(dir,'KR Film')):
        if file.endswith(".txt"):
            KR_Film[file] = np.loadtxt(join(dir,'KR Film',file), skiprows=17,usecols = 1, comments='>')
        
    KR_Film_nd = {}
    for file in listdir(join(dir,'KR Film next day')):
        if file.endswith(".txt"):
            KR_Film_nd[file] = np.loadtxt(join(dir,'KR Film next day',file), skiprows=17,usecols = 1, comments='>')

if day_of_exp == '21_06 Led':
    Led = {}
    wl2 = np.loadtxt(join(dir,'Led266.txt'), skiprows=17,usecols = 0, comments='>')
    for file in listdir(join(dir)):
        if file.endswith("6.txt"):
            Led[file] = np.loadtxt(join(dir,file), skiprows=17,usecols = 1, comments='>')

dir_f0 = join(dir, dir_f0_name)
#if not day_of_exp == '21_06 Led':
#    for file in listdir(dir_f0):
#        if file.endswith(".txt"):
#            f0_names.append(join(dir_f0,file))

wl = np.loadtxt(listdir(dir_f0)[0], skiprows=17,usecols = 0, comments='>')

norm_peak_indx = (wl>750)&(wl<850)
flu_indx = (wl>300)&(wl<600)
bad_peak_indx = (wl>390)&(wl<405)

if tos == 'MFO'or tos == 'TCM':
    conc = np.array([0.05,0.1,0.2,0.5,1.0])
    conc = np.array([0.001,0.002,0.004,0.01,0.02])
    dir_f1 = join(dir,tos)
    fData1 = {}
    dict_for_dyn = {}
    for dir in listdir(dir_f1):
        test = np.array([])
        test2 = np.array([])
        for itm in listdir(join(dir_f1,dir)):
            if itm.endswith(".txt"):
                tmpVar = np.loadtxt(join(dir_f1,dir,itm), skiprows=17,usecols = 1, comments='>')
                if not test.any():
                    test = tmpVar
                else:
                    test = np.vstack((test,tmpVar))
            if  isdir(join(dir_f1,dir,itm)):
                for file in listdir(join(dir_f1,dir,itm)):
                    if file.endswith(".txt"):
                        tmpVar2 = np.loadtxt(join(dir_f1,dir,itm,file), skiprows=17,usecols = 1, comments='>')
                        if not test2.any():
                            test2 = tmpVar2
                        else:
                            test2 = np.vstack((test2,tmpVar2))
                        dict_for_dyn[dir] = test2
        fData1[dir] = test[0:4]
        
res_dir = join('..','results',str(datetime.date.today()),day_of_exp)
sna = join(res_dir,tos)

if not isdir(res_dir):
    makedirs(abspath(res_dir))
    
if not day_of_exp == '21_06 Led':
    tmpVar = np.loadtxt(f0_names[0], skiprows=17,usecols = 1, comments='>')

fData0 = np.zeros((len(tmpVar),len(f0_names)))
fIngl = np.zeros((1,len(f0_names)))
i = 0
for itm in f0_names:
    fData0[:,i] = np.loadtxt(itm, skiprows=17,usecols = 1,comments='>')
    fData0[:,i][bad_peak_indx] = np.random.normal(np.linspace(fData0[:,i][bad_peak_indx][0:1].item(),fData0[:,i][bad_peak_indx][-2:-1].item(),np.count_nonzero(bad_peak_indx)),50)
    fIngl[:,i] = np.trapz(fData0[flu_indx,i],wl[flu_indx])/max(fData0[norm_peak_indx,i])
#    if i== 0:
 #       break
    i+=1

fData0_m = np.mean(fData0, axis = 1, dtype=np.float64)

fData1_mn = {}

for itm in fData1_m:
    fData1_mn[itm] = fData1_m[itm]/max(fData1_m[itm][norm_peak_indx])
    
    
#-----------fault_0_05--------------------!!!----------------------------------
if tos == 'TCM' and day_of_exp == '20_06 Oil Convert':
    max_vls = {}
    edt_fData1_m  = {}
    edt_fData1_m = fData1_m
    for itm in edt_fData1_m:
        if itm == '0_05':
            continue
        max_vls[itm] = max(edt_fData1_m[itm][norm_peak_indx])
    
    mOfMaxs = mean(list(max_vls.values()))
    fData1_mn['0_05'] = fData1_m['0_05']/mOfMaxs
    
    mxs_t = []
    for time in list(range(6)):
        mxss = []
        for c in dict_for_dyn:
            if c == "0_05":
                continue
            mxss.append(max(dict_for_dyn[c][time][norm_peak_indx]))
        mxs_t.append(mean(mxss))
           
        
#------------------------------!!!--------------------------------------

fData0_mn = fData0_m/max(fData0_m[norm_peak_indx])

fData_plot = fData1_mn
fData_plot[dir_f0_name] = fData0_mn

#fData_plot = Led
    
#plt.style.use('mystyle')
fig, (ax1) = plt.subplots(1,1,figsize=(18,10))
fig.set_tight_layout(True)

for itm in fData_plot:
    ax1.plot(wl,fData_plot[itm])


ax1.legend(np.hstack([conc, 'Sea']))

ax1.set(xlabel = 'Wavelength, nm', 
       ylabel = 'I, Relative units', 
       title = rad_type+' '+tos,
       xlim = [290,610],
       ylim = 0
       )
if do_save == 1:
    fig.savefig(sna+'.jpg',transparent=False,dpi=300,bbox_inches="tight")       

plt.show()

f_sn = {}
for key in fData1_mn:
    f_sn[key] = np.trapz(fData1_mn[key][flu_indx],wl[flu_indx])-np.trapz(fData0_mn[flu_indx],wl[flu_indx])
del f_sn[dir_f0_name]

f_sn = list(f_sn.values())

def func_line(x, a, b):
    return a*x+b
    
def func_pow1(x, a, b):
    return a*x**b   
    
def func_pow2(x, a, b, c):
    return a*x**b+c   
    
def func_pow3(x, a, b):
    return a*x**b+0.5
    
func1 = func_line

popt, pcov = curve_fit(func1, conc, f_sn)

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))
    
def coefficient_of_determination(ys_orig,ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

#np.trapz(fData0_mn[flu_indx],wl[flu_indx])
LoD = 3.0*np.std(fIngl[0:4])/popt[0]

r_square = coefficient_of_determination(np.array(f_sn),np.array(func1(conc, *popt)))

fig, (ax1) = plt.subplots(1,1,figsize=(18,10))
fig.set_tight_layout(True)


ax1.scatter(conc, f_sn)
ax1.plot(conc, func1(conc, *popt),'r-')
ax1.errorbar(conc, f_sn,list(stds_v2.values()))



ax1.set(xlabel = 'Conc, %', 
       ylabel = 'SN, Relative units', 
       title = rad_type+' '+tos,
       xlim = [0,conc[-1]],
       ylim = 0
       )

ax1.axvline(LoD, ls = '--', color = 'g')
ax1.text(LoD,250,'LOD', color = 'g')
ax1.text(LoD,0,"{:.4f}".format(LoD), color = 'g')
ax1.text(ax1.get_xlim()[1]/2,ax1.get_ylim()[1]/1.5,'$R^2$ = '+"{:.4f}".format(r_square))

if do_save == 1:
    fig.savefig(sna+'_LOD'+'.jpg',transparent=False,dpi=300,bbox_inches="tight")       



plt.show()

if bool(dict_for_dyn):

    dyn_conc = '1'
    
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.collections import PolyCollection
    from matplotlib import colors as mcolors
    
    fig = plt.figure(figsize=(18,10))
    ax = fig.gca(projection='3d')
    fig.set_tight_layout(True)
    fig.tight_layout(pad=10.4, w_pad=10.5, h_pad=11.0)
    
    def cc(arg):
        return mcolors.to_rgba(arg, alpha=0.6)
    
    xs = wl
    verts = []
    zs = [0.0, 1.0, 2.0, 3.0, 4.0]
    for z in zs:
        ys = dict_for_dyn[dyn_conc][int(z)][1:1500]
        ys[0], ys[-1] = 0, 0
        verts.append(list(zip(xs, ys)))
    
    poly = PolyCollection(verts, facecolors=[cc('r'), cc('g'), cc('b'),
                                             cc('y'), cc('r')])
    poly.set_alpha(0.4)
    ax.add_collection3d(poly, zs=zs, zdir='y')
    
    ax.set_xlabel('Wavelength, nm',labelpad=20)
    ax.set_xlim3d(200, 900)
    ax.set_ylabel('T',labelpad=20)
    ax.set_ylim3d(0, 4)
    ax.set_zlabel('I, Relative units',labelpad=20)
    ax.set_zlim3d(0, max(ys))
    
    if do_save == 1:
        fig.savefig(sna+'_Dynamic'+dyn_conc+'.jpg',transparent=False,dpi=300,bbox_inches="tight")    
    
    plt.show()
    
    

    
    fig, (ax1) = plt.subplots(1,1,figsize=(18,10))
    fig.set_tight_layout(True)
    
    itm = 0
    for conc in dict_for_dyn:
        for t in [0,1,2,3,4]:
            if conc == '0_05' and tos == 'TCM' and day_of_exp == '20_06 Oil Convert':

                
                dict_for_dyn[conc][t] = np.trapz(dict_for_dyn[conc][t][flu_indx]/mxs_t[t],wl[flu_indx])-np.trapz(fData0_mn[flu_indx],wl[flu_indx])
            else:
                dict_for_dyn[conc][t] = np.trapz(dict_for_dyn[conc][t][flu_indx]/max(dict_for_dyn[conc][t][norm_peak_indx]),wl[flu_indx])-np.trapz(fData0_mn[flu_indx],wl[flu_indx])
        if conc == '0_05' and tos == 'TCM' and day_of_exp == '20_06 Oil Convert':
            ax1.plot([5,10,20,30,60], dict_for_dyn[conc][0:-1,0],'.-', markersize=15, alpha = 0)
            
        else:
            ax1.plot([1,5,10,20,30,60], np.hstack((tcm_1st_day[itm],dict_for_dyn[conc][0:-1,0])),'.-', markersize=15)  
            ax1.errorbar([1,5,10,20,30,60], np.hstack((tcm_1st_day[itm],dict_for_dyn[conc][0:-1,0])),stds_v2[conc],barsabove = True, c = 'b')
            
        itm+=1
   
    
    for itm in [0,1,2,3,4]:
        ax1.errorbar(1, tcm_1st_day[itm],list(stds_tcm_1st_day.values())[itm],barsabove = True, c = 'r')


    conc = np.array([0.001,0.002,0.004,0.01,0.02])
    ax1.legend(conc)
    
    ax1.set(xlabel = 't, s', 
           ylabel = 'I, Relative units', 
           title = rad_type+' '+tos,
           xlim = 0,
           ylim = 0     
           )
    if do_save == 1:
        fig.savefig(sna+'dyn.jpg',transparent=False,dpi=300,bbox_inches="tight")       
    
    plt.show()









