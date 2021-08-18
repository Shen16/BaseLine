#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 01:03:47 2021

@author: shehnazislam
"""
import json  # Serialization: process of encoding data into JSON format (like converting a Python list to JSON). Deserialization: process of decoding JSON data back into native objects you can work with (like reading JSON data into a Python list)

import math  # To use mathematical functions
import re  # Regular Expression, The functions in this module let you check if a particular string matches a given regular expression
import random  # random number generation. random() function, generates random numbers between 0 and 1.
from random import randint  # randint() is an inbuilt function of the random module in Python3
from statistics import mean, median, stdev  # mean() function can be used to calculate mean/average of a given list of numbers.
from operator import itemgetter  # operator is a built-in module providing a set of convenient operators #operator. itemgetter(n) assumes an iterable object (e.g. list, tuple, set) as input, and fetches the n-th element out of it. If multiple items are specified, returns a tuple of lookup values.
from scipy.stats import linregress  # Calculate a linear least-squares regression for two sets of measurements. Parameters x, yarray_like.
from sklearn import preprocessing  # The sklearn. preprocessing package provides several functions that transform your data before feeding it to the algorithm.
import pandas as pd  # presents a diverse range of utilities, ranging from parsing multiple file formats to converting an entire data table into a NumPy matrix array.
import numpy as np  # NumPy i


def stringToFloat(str):
    list= []
    for i in str:
        extractNums= re.findall(r"[-+]?\d*\.\d+|\d+", i)
        num= extractNums[0]
        list.append(num) 
    return list

xVal= ['2018', '2017', '2016', '2015' ,'2014', '2013', '2012' ,'2011', '2010', '2009',
 '2008', '2007', '2006' ,'2005']
# print(xVal)

yVals= [['55968.7', '45622.2', '48752.7', '71404.6' ,'43256.4' ,'91886.1', '55652.9' ,'65749.5', '48278.3' ,'29452.8' ,'42525.4' ,'38762.0' ,'28678.0' ,'24045.7'],
 ['108775.3', '94101.9', '78531.7' ,'88653.6', '69174.6', '103475.2', '72623.0' ,'75092.9', '57152.3' ,'44580.8' ,'62685.1' ,'59961.2' ,'42992.7', '37427.3']]
# print(yVals[0])

yVals_float= yVals
# print(len(yVals))
for i in range(0, len(yVals)):
    yVals_float[i] = stringToFloat(yVals[i])
# print(yVals_float)


yVals= np.array(yVals).astype(np.float) # yVal is now in float type




coordinates = dict(zip(xVal, zip(*yVals)))
print(coordinates)

sorted_coordinates= dict(sorted(coordinates.items()))
# for key, value in sorted(coordinates.items()): # Note the () after items!
#     print(key, value)
print("sorted_coordinates")
print(sorted_coordinates)

keys, values = zip(*sorted_coordinates.items())

print(keys)
print(values)

arr=[]
for j in range(0, len(values[0])):
    array=[]
    for i in range(0, len(values)):
        array.append(values[i][j])
    arr.append(array)

print("xVal")
print(keys)
print("yVals")
print(arr)

xVal_sorted = np.array(keys)
yVals_sorted= np.array(arr)



## Correct Algorithm:
    
xVal= valueArrMatrix[0,:]
# print(xVal)

yVals= valueArrMatrix[1:,:]
# print(yVals)

yVals_float= yVals
# print(len(yVals))
for i in range(0, len(yVals)):
    yVals_float[i] = stringToFloat(yVals[i])
# print(yVals_float)


yVals= np.array(yVals_float).astype(np.float) # yVal is now in float type
# print(yVals)




coordinates = dict(zip(xVal, zip(*yVals)))
# print(coordinates)

sorted_coordinates= dict(sorted(coordinates.items()))
# for key, value in sorted(coordinates.items()): # Note the () after items!
#     print(key, value)
# print("sorted_coordinates")
# print(sorted_coordinates)

keys, values = zip(*sorted_coordinates.items())

# print(keys)
# print(values)

arr=[]
for j in range(0, len(values[0])):
    array=[]
    for i in range(0, len(values)):
        array.append(values[i][j])
    arr.append(array)

# print("keys== xVal")
# print(keys)
# print("arr== yVals")
# print(arr)



# xVal_sorted = xVal[len(xVal)::-1]  
  
# yVals_sorted= yVals
# for i in range(0, len(yVals)):
#     yVals_sorted[i] = yVals[i][len(yVals[i])::-1]  ## Ordered correctly this time


xVal_sorted = np.array(keys)
yVals_sorted= np.array(arr)


print("Sorted X vals")
print(xVal_sorted)
print("Sorted Y vals")
print(yVals_sorted)
            
        
    
        



    
    
    
