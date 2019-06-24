from scipy.optimize import curve_fit

def func(x, a, b):
    return a * x + b

def fitting(xdata, ydata):
    popt, pcov = curve_fit(func, xdata, ydata)
    return popt