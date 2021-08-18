#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 20:21:12 2021

@author: shehnazislam
"""




# import numpy as np
# import matplotlib.pyplot as plt


# x = np.linspace(0,2*np.pi,100)
# y = np.sin(x) + np.random.random(100) * 0.2

# from scipy.signal import savgol_filter
# yhat = savgol_filter(y, 1-g ;hyii7uk, 3) # window size 51, polynomial order 3

# plt.plot(x,y)
# plt.plot(x,yhat, color='red')
# plt.show()



# import numpy as np
# import numpy as np
# from scipy.interpolate import make_interp_spline
# import matplotlib.pyplot as plt
 
# # Dataset
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
# y = np.array([20, 30, 5, 12, 39, 48, 50, 3])
# # plt.plot(x, y)
 
# X_Y_Spline = make_interp_spline(x, y)
 
# # Returns evenly spaced numbers
# # over a specified interval.
# X_ = np.linspace(x.min(), x.max(), 500)
# Y_ = X_Y_Spline(X_)
 
# # Plotting the Graph
# plt.plot(X_, Y_)
# plt.title("Plot Smooth Curve Using the scipy.interpolate.make_interp_spline() Class")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.show()

# print(x)
# print(y)
# print(X_)
# print(Y_)



import numpy as np
import scipy.fftpack

N = 100
x = np.linspace(0,2*np.pi,N)
y = np.sin(x) + np.random.random(N) * 0.2

w = scipy.fftpack.rfft(y)
f = scipy.fftpack.rfftfreq(N, x[1]-x[0])
spectrum = w**2

cutoff_idx = spectrum < (spectrum.max()/5)
w2 = w.copy()
w2[cutoff_idx] = 0

y2 = scipy.fftpack.irfft(w2)





