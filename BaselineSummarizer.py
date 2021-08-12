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
import numpy as np  # NumPy is a general-purpose array-processing package. It provides a high-performance multidimensional array object, and tools for working with these arrays.

dataPath = 'Data/test/testData.txt'
titlePath = 'Data/test/testTitle.txt'

# websitePath = 'results/generated_baseline'
websitePath = 'static/generated'  # Folder where the json file is created as the final output
# websitePath = '../TourDeChart/generated'

summaryList = []


def globalTrendBarChart(yValueArr):
    reversed_yValueArr = yValueArr[::-1]  # reversing

    globalDifference = float(reversed_yValueArr[0]) - float(reversed_yValueArr[len(reversed_yValueArr) - 1])
    if reversed_yValueArr[len(reversed_yValueArr) - 1] == 0:
        reversed_yValueArr[len(reversed_yValueArr) - 1] = 1
    globalPercentChange = (globalDifference / float(reversed_yValueArr[len(reversed_yValueArr) - 1])) * 100

    bar_trend = ""

    up_trend = ["increased", "grew", "climbed", "risen"]
    down_trend = ["decreased", "declined", "reduced", "lowered"]
    constant_trend = ["stable", "constant", "unchanged", "unvaried"]

    if globalPercentChange > 0:
        bar_trend = up_trend[random.randint(0, len(up_trend) - 1)]
    elif globalPercentChange < 0:
        bar_trend = down_trend[random.randint(0, len(down_trend) - 1)]
    else:
        bar_trend = "remained " + constant_trend[random.randint(0, len(constant_trend) - 1)]

    return bar_trend


def match_trend(trend1, trend2):
    if trend1 in ["increased", "grew", "climbed", "risen"] and trend2 in ["increased", "grew", "climbed", "risen"]:
        return 1
    elif trend1 in ["decreased", "declined", "reduced", "lowered"] and trend2 in ["decreased", "declined", "reduced", "lowered"]:
        return 1
    elif trend1 in ["stable", "constant", "unchanged", "unvaried"] and trend2 in ["stable", "constant", "unchanged", "unvaried"]:
        return 1
    else:
        return 0


def checkIfDuplicates(listOfElems):
    # Check if given list contains any duplicates
    setOfElems = set()
    for elem in listOfElems:
        if elem in setOfElems:
            return True
        else:
            setOfElems.add(elem)
    return False


def most_frequent(List):
    # to find most frequent
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i
    return num


def getChartType(x):
    if x.lower() == 'year':
        return 'line_chart'
    else:
        return 'bar_chart'


def openCaption(captionPath):
    with open(captionPath, 'r', encoding='utf-8') as captionFile:
        caption = captionFile.read()
    return caption


def openData(dataPath):
    df = pd.read_csv(dataPath)
    cols = df.columns
    size = df.shape[0]
    xAxis = cols[0]
    yAxis = cols[1]
    chartType = getChartType(xAxis)
    return df, cols, size, xAxis, yAxis, chartType


def cleanAxisLabel(label):
    cleanLabel = re.sub('\s', '_', label)
    cleanLabel = cleanLabel.replace('%', '').replace('*', '')
    return cleanLabel


def cleanAxisValue(value):
    # print(value)
    if value == '-' or value == 'nan':
        return '0'
    cleanValue = re.sub('\s', '_', value)
    cleanValue = cleanValue.replace('|', '').replace(',', '').replace('%', '').replace('*', '')
    return cleanValue


def getMagnitude(normalizedSlope):
    magnitude = "slightly"
    # print(normalizedSlope)
    if (abs(normalizedSlope) > 0.75):
        magnitude = "extremely"
    elif (abs(normalizedSlope) > 0.25 and abs(normalizedSlope) <= 0.75):
        magnitude = "moderately"
    else:
        mangitude = "slightly"
    return magnitude







## shehnaz-- The functions created by me

# Initilizing constant values for the fucntions below
# mean_percentArray= 0
# sd_percentArray= 0


# constant_rate = 3.45# avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
# significant_rate = 6.906 # avg(% chnage)*0.1 # Meaning any chnage >constant rate and less than this rate is considered not significant and so it's trend direction is chnaged to the trend of the succesive interval # Determines the start and end of the trend
# rapidly_rate= 57.55
# gradually_rate= 28.77

# constant_rate = mean_percentArray- 1*(sd_percentArray) # avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
# significant_rate = mean_percentArray# avg(% chnage)*0.1 # Meaning any chnage >constant rate and less than this rate is considered not significant and so it's trend direction is chnaged to the trend of the succesive interval # Determines the start and end of the trend
# gradually_rate= mean_percentArray+ 1*(sd_percentArray)
# rapidly_rate= mean_percentArray+ 2*(sd_percentArray)

# meanRefinedSlope= 0
# sdRefinedSlope= 0

# constant_rate = 20# avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
# significant_rate= 40 # avg(% chnage)*0.1 # Meaning any chnage >constant rate and less than this rate is considered not significant and so it's trend direction is chnaged to the trend of the succesive interval # Determines the start and end of the trend
# gradually_rate= 50
# rapidly_rate= 70

#These rate stay constant
constant= 5
sig= 10
gradual=20
rapid= 70

## These rate chnages dynamically with c_rate and mean(percentChnageArr)
constant_rate = constant
significant_rate = 0
gradually_rate= gradual
rapidly_rate= rapid


c_rate = 0.6 #0.6 avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
s_rate = 1.2 #1.2
g_rate = 2 #2
r_rate = 3 #3

## These rate chnages dynamically with c_rate and mean(percentChnageArr)
constant_rate = constant
significant_rate = 0
gradually_rate= gradual
rapidly_rate= rapid


c_rate = 0.6 #0.6 avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
s_rate = 1.2 #1.2
g_rate = 2 #2
r_rate = 3 #3






def directionTrend(new, old, constant_rate):
    difference = new - old
    if (old != 0):
        percentageChange = ((new - old) / old) * 100
    else:
        old = 0.00000000001
        percentageChange = ((new - old) / old) * 100

    absChnage = abs(percentageChange)
    if (difference > 0 and absChnage > constant_rate):  # if change is significant >5%
        return "increasing"
    elif (difference < 0 and absChnage > constant_rate):
        return "decreasing"
    else:
        return "constant"


def rateOfChnage(refinedPercentChnageArr, direction, c, g, r):
    # new_x= float(new_x)
    # old_x= float(old_x)

    # percentageChange = ((new_y - old_y) / new_x-old_x)

    # # min_val= 0
    # # max_val= 100

    # if (max_val-min_val != 0):
    #     normalized_percentChange= (100*(percentageChange- min_val))/(max_val-min_val)
    # else:
    #     normalized_percentChange= (100*(percentageChange- min_val))/0.00000000001

    constant_rate = c
    gradually_rate = g
    rapidly_rate = r

    absChnage = abs(refinedPercentChnageArr)
    if (direction == "constant"):
        return "roughly"
    elif (absChnage > rapidly_rate):
        return "rapidly"
    elif (absChnage > gradually_rate):
        return "gradually"
    elif (absChnage > constant_rate):
        return "slightly"
    else:
        return "roughly"


def globalDirectionTrend(percent, constant_rate):
    absChnage = abs(percent)
    if (percent > 0 and absChnage > constant_rate):  # if change is significant >5%
        return "increasing"
    elif (percent < 0 and absChnage > constant_rate):
        return "decreasing"
    else:
        return "constant"


def globalRateOfChange(percentChange, c, g, r):
    # new_x= float(new_x)
    # old_x= float(old_x)

    # percentageChange = ((new_y - old_y) / new_x-old_x)

    # # min_val= 0
    # # max_val= 100

    # if (max_val-min_val != 0):
    #     normalized_percentChange= (100*(percentageChange- min_val))/(max_val-min_val)
    # else:
    #     normalized_percentChange= (100*(percentageChange- min_val))/0.00000000001

    constant_rate = c
    gradually_rate = g
    rapidly_rate = r

    absChnage = abs(percentChange)
    if (absChnage > rapidly_rate):
        return "rapidly"
    elif (absChnage > gradually_rate):
        return "gradually"
    elif (absChnage > constant_rate):
        return "slightly"
    else:
        old = 0.00000000001
        percentageChange = ((new - old) / (maximum-0)) * 100
    return percentageChange

    
# def rateOfChnageVal(new_y, old_y, direction, new_x, old_x,  max_val, min_val):
#     new_x= float(new_x)
#     old_x= float(old_x)
   
#     percentageChange = ((new_y - old_y) / new_x-old_x) 

#     # min_val= 0
#     # max_val= 100
#     print(max_val)
#     print(min_val)
    
#     if (max_val-min_val != 0):
#         normalized_percentChange= (100*(percentageChange- min_val))/(max_val-min_val)
#     else:
#         normalized_percentChange= (100*(percentageChange- min_val))/0.00000000001
   
#     return normalized_percentChange

def percentChnageFunc(new, old):
    difference = new - old
    if (old != 0):
        percentageChange = ((new - old) / old) * 100
    else:
        old = 0.00000000001
        percentageChange = ((new - old) / old) * 100
    return percentageChange



def percentChnageRangeFunc(new, old, maximum):
    difference = new - old
    if (old != 0):
        percentageChange = ((new - old) / (maximum-0)) * 100
    else:
        old = 0.00000000001
        percentageChange = ((new - old) / (maximum-0)) * 100
    return percentageChange


# def rateOfChnageVal(new_y, old_y, direction, new_x, old_x,  max_val, min_val):
#     new_x= float(new_x)
#     old_x= float(old_x)

#     percentageChange = ((new_y - old_y) / new_x-old_x)

#     # min_val= 0
#     # max_val= 100
#     print(max_val)
#     print(min_val)

#     if (max_val-min_val != 0):
#         normalized_percentChange= (100*(percentageChange- min_val))/(max_val-min_val)
#     else:
#         normalized_percentChange= (100*(percentageChange- min_val))/0.00000000001

#     return normalized_percentChange

def increaseDecrease(x):
    if (x == "increasing"):
        return "increase"
    elif (x == "decreasing"):
        return "decrease"
    else:
        return "stays the same"
    


def get_indexes_max_value(l):
    max_value = max(l) # key=lambda x:float(x))
    return [i for i, x in enumerate(l) if x == max(l)]


def get_indexes_min_value(l):
    min_value = min(l) # key=lambda x:float(x))
    return [i for i, x in enumerate(l) if x == min(l)]

def stringToFloat(str):
    list= []
    for i in str:
        extractNums= re.findall(r"[-+]?\d*\.\d+|\d+", i)
        num= extractNums[0]
        list.append(num) 
    return list
        
def commaAnd(arr):
    if (len(arr)<2):
        arr=arr[0]
    else:
        slice1= arr[:len(arr)-1]
        # print(slice1)
        slice2=  ", ".join(slice1)
        slice2+= ", and " +  arr[-1]
        # print(slice2)
        arr= slice2
    return arr

def stringToFloat(str):
    list= []
    for i in str:
        extractNums= re.findall(r"[-+]?\d*\.\d+|\d+", i)
        num= extractNums[0]
        list.append(num)
    return list


def commaAnd(arr):
    if (len(arr)<2):
        arr=arr[0]
    else:
        slice1= arr[:len(arr)-1]
        # print(slice1)
        slice2=  ", ".join(slice1)
        slice2+= ", and " +  arr[-1]
        # print(slice2)
        arr= slice2
    return arr


# scaler = preprocessing.MinMaxScaler()
count = 0


# with open(dataPath, 'r', encoding='utf-8') as dataFile, \
#         open(titlePath, 'r', encoding='utf-8') as titleFile:
#
#     fileIterators = zip(dataFile.readlines(), titleFile.readlines())
#     for data, title in fileIterators:

def summarize(data, all_y_label, name, title):
    # scaler = preprocessing.MinMaxScaler()
    # count += 1
    datum = data.split()  # Splits data where space is found. So datum[0] is groups of data with no space. e.g. Country|Singapore|x|bar_chart                 `
    # check if data is multi column
    columnType = datum[0].split('|')[2].isnumeric()  # e.g. Country|Singapore|x|bar_chart, ...  x means single, numeric means multiline

    # print("Column Type -> " + str(columnType) + " this is -> " + str(datum[0].split('|')[2]))

    # fp = open("all_Y_labels.txt", "a")

    if columnType:  # If MULTI

        # fp.write(str(name) + "\t\n")
        # fp.close()

        y_label = all_y_label

        labelArr = []
        chartType = datum[0].split('|')[3].split('_')[0]

        values = [value.split('|')[1] for value in datum]  # for every datum take the 2nd element

        # print("VALUES")
        # for a in values:
        #     print(a)

        # find number of columns:
        columnCount = max([int(data.split('|')[2]) for data in
                           datum]) + 1  # The number of categories #for every datum take the 3rd element
        # Get labels
        for i in range(columnCount):
            label = datum[i].split('|')[0].split('_')
            labelArr.append(
                label)  # e.g. "Year|2018|0|line_chart Export|55968.7|1|line_chart Import|108775.3|2|line_chart Year|2017|0|line_chart ==> [['Year'], ['Export'], ['Import']]

        # print(labelArr)

        stringLabels = [' '.join(label) for label in labelArr]  # e.g. stringLabels = ['Year', 'Export', 'Import']

        # print("stringLabels -> ")
        # for a in stringLabels:
        #     print(a)

        # Get values
        valueArr = [[] for i in range(columnCount)]
        cleanValArr = [[] for i in range(columnCount)]

        # print("columnCount -> " + str(columnCount))

        # columnCount : how many grouped bars
        # stringLabels : label of X-axis and the individual groups

        groupedLabels = []

        for i in range(len(stringLabels)):
            groupedLabels.append(str(stringLabels[i]).replace('_', ' '))

        # print("groupedLabels")
        # for a in groupedLabels:
        #     print(a)

        a = 0
        b = 0

        groupedCol = int(len(values) / len(stringLabels))

        row = groupedCol
        col = columnCount
        arr = np.empty((row, col),
                       dtype=object)  # creates a martic with rows representing each distinct x value and cols representing y values for different categories/lines (2 in this case)
        # arr[0, 0] = stringLabels[0]

        m = 0
        n = 0

        for b in range(len(values)):
            if n == col:
                m += 1
                n = 0
            if a == len(stringLabels):
                a = 0
            if (b % columnCount) == 0:
                arr[m][b % columnCount] = str(values[b]).replace('_', ' ')
            else:
                num = ""
                for c in values[b]:  # Done for error: could not convert string to float: '290$'
                    if c.isdigit():
                        num = num + c
                arr[m][b % columnCount] = float(num)

            n += 1
            a += 1

        max_row = []
        max_row_val = []
        min_row = []
        min_row_val = []

        number_of_group = len(groupedLabels) - 1

        for i in range(len(groupedLabels) - 1):
            arr1 = arr[arr[:, (i + 1)].argsort()]
            min_row.append(arr1[0][0])
            min_row_val.append(arr1[0][i + 1])
            arr2 = arr[arr[:, (i + 1)].argsort()[::-1]]
            max_row.append(arr2[0][0])
            max_row_val.append(arr2[0][i + 1])

        # print(max_row) # x values at which max occured for each category (e.g. ['2013', '2018'] ==> Export max occured at 2013 and Import at 2018)
        # print(max_row_val) # y values at which max occured for each category (e.g. [91886.1, 108775.3] ==> Export max occured at 91886.1 and Import at 108775.3)
        # print(min_row)
        # print(min_row_val)

        global_max = max(max_row_val)
        global_max_index = get_indexes_max_value(max_row_val)
        global_max_category_label = groupedLabels[global_max_index[0] + 1]
        global_max_category_xlabel = str(max_row[global_max_index[0]])

        if len(groupedLabels) > 3:
            global_2nd_max = sorted(max_row_val)[1]
            global_2nd_max_index = get_indexes_max_value(max_row_val)
            global_2nd_max_category_label = groupedLabels[global_2nd_max_index[0] + 1]
            global_2nd_max_category_xlabel = str(max_row[global_2nd_max_index[0]])

        global_min = min(min_row_val)
        global_min_index = get_indexes_min_value(min_row_val)
        global_min_category_label = groupedLabels[global_min_index[0] + 1]
        global_min_category_xlabel = str(min_row[global_min_index[0]])

        # print("Global Max")
        # print(global_max)
        # print(global_max_index)
        # print(global_max_category_label)
        # print(global_max_category_xlabel)
        #
        # print("Global 2nd Max")
        # print(global_2nd_max)
        # print(global_2nd_max_index)
        # print(global_2nd_max_category_label)
        # print(global_2nd_max_category_xlabel)
        # #
        # print("Global Min")
        # print(global_min)
        # print(global_min_index)
        # print(global_min_category_label)
        # print(global_min_category_xlabel)


        rowCount = round(
            len(datum) / columnCount)  # same as groupedCols or row, with rows representing each distinct x value

        categoricalValueArr = [[] for i in range(rowCount)]

        i = 0
        for n in range(rowCount):
            for m in range(columnCount):
                value = values[i]
                cleanVal = datum[i].split('|')[1].replace('_', ' ')
                valueArr[m].append(value)
                cleanValArr[m].append(cleanVal)
                if m == 0:
                    categoricalValueArr[n].append(cleanVal)
                else:
                    categoricalValueArr[n].append(float(re.sub("[^\d\.]", "", cleanVal)))
                i += 1
        titleArr = title.split()
        # calculate top two largest categories
        summaryArray = []
        dataJson = []
        # iterate over index of a value
        for i in range(len(cleanValArr[0])):
            # iterate over each value
            dico = {}
            for value, label in zip(cleanValArr, labelArr):
                cleanLabel = ' '.join(label)
                dico[cleanLabel] = value[i]
            dataJson.append(dico)

        # HERE

        # print(json.dumps(dataJson, indent=4, sort_keys=True))

        if (chartType == "bar"):

            meanCategoricalDict = {}
            stringLabels.insert(len(stringLabels) - 1, 'and')
            categories = ', '.join(stringLabels[1:-1]) + f' {stringLabels[-1]}'
            # if rowCount > 2:
            for category in categoricalValueArr:
                meanCategoricalDict[category[0]] = mean(category[1:])
            sortedCategories = sorted(meanCategoricalDict.items(), key=lambda x: x[1])

            # print("sortedCategories")
            # print(sortedCategories)

            numerator = abs(sortedCategories[-1][1] - sortedCategories[-2][1])
            denominator = (sortedCategories[-1][1] + sortedCategories[-2][1]) / 2
            topTwoDelta = round((numerator / denominator) * 100, 1)

            numerator1 = abs(sortedCategories[-1][1] - sortedCategories[0][1])
            denominator1 = (sortedCategories[-1][1] + sortedCategories[0][1]) / 2
            minMaxDelta = round((numerator1 / denominator1) * 100, 1)

            group_names = groupedLabels[1:]
            group_names_text = ""

            for a in range(len(group_names)):
                if a == len(group_names) - 1:
                    group_names_text += "and " + group_names[a]
                else:
                    group_names_text += group_names[a] + ", "

            rand_category_index = random.randint(0, number_of_group - 1)
            global_max_min_categorical = []
            global_max_min_categorical.append(" In " + str(groupedLabels[0]) + " " + str(max_row[rand_category_index]) + ", " + str(groupedLabels[rand_category_index+1]) + " had the highest " + y_label + " among all " + str(rowCount) + " " + str(groupedLabels[0]) + "s and it has the lowest " + y_label + " in " + str(groupedLabels[0]) + " " + str(min_row[rand_category_index]) + ". ")
            global_max_min_categorical.append(" In " + str(groupedLabels[0]) + " " + str(max_row[rand_category_index]) + ", " + str(groupedLabels[rand_category_index+1]) + " had the maximum " + y_label + " and it saw the lowest in " + str(groupedLabels[0]) + " " + str(min_row[rand_category_index]) + " out of all " + str(rowCount) + " " + str(groupedLabels[0]) + "s. ")
            global_max_min_categorical.append(" Among all the " + str(groupedLabels[0]) + "s, " + str(groupedLabels[rand_category_index+1]) + " had the highest " + y_label + " in " + str(groupedLabels[0]) + " " + str(max_row[rand_category_index]) + " and lowest " + y_label + " in " + str(groupedLabels[0]) + " " + str(min_row[rand_category_index]) + ". ")
            global_max_min_categorical.append(" Among all the " + str(groupedLabels[0]) + "s, " + str(groupedLabels[rand_category_index+1]) + " had the highest " + y_label + " " + str(max_row_val[rand_category_index]) + " in " + str(groupedLabels[0]) + " " + str(max_row[rand_category_index]) + " and lowest value " + str(min_row_val[rand_category_index]) + " in " + str(groupedLabels[0]) + " " + str(min_row[rand_category_index]) + ". ")

            extrema_categorical = global_max_min_categorical[random.randint(0, len(global_max_min_categorical) - 1)]
            print("Extrema [min/max][categorical] : " + global_max_min_categorical[random.randint(0, len(global_max_min_categorical) - 1)])

            trend_global = None

            if groupedLabels[0].lower() == "year" or groupedLabels[0].lower() == "years" or groupedLabels[0].lower() == "month" or groupedLabels[0].lower() == "months" or groupedLabels[0].lower() == "quarter" or groupedLabels[0].lower() == "quarters":
                category_trend = []

                for a in range(1, len(arr[0])):
                    # print(arr[:, a])
                    category_trend.append(globalTrendBarChart(arr[:, a]))

                # print(category_trend)
                categorical_global_trend = []

                if match_trend(category_trend[rand_category_index], category_trend[rand_category_index-1]):
                    categorical_global_trend.append(" Over the " + str(rowCount) + " " + groupedLabels[0] + "s, the " + y_label + " for both " + str(groupedLabels[rand_category_index + 1]) + " and " + str(groupedLabels[rand_category_index]) + " have " + category_trend[rand_category_index] + ". ")
                    categorical_global_trend.append(" All through the " + groupedLabels[0] + "s, similar trend was observed for " + str(groupedLabels[rand_category_index + 1]) + " and " + str(groupedLabels[rand_category_index]) + ". In both cases, the " + y_label + " have " + category_trend[rand_category_index] + ". ")
                else:
                    categorical_global_trend.append(
                        " Over the " + str(rowCount) + " " + groupedLabels[0] + "s, the " + y_label + " for " + str(groupedLabels[rand_category_index + 1]) + " have been " + category_trend[rand_category_index] + " whereas " + category_trend[rand_category_index - 1] + " for " + str(groupedLabels[rand_category_index]) + ". ")
                    categorical_global_trend.append(
                        " All through the " + groupedLabels[0] + "s, the " + y_label + " for " + str(groupedLabels[rand_category_index + 1]) + " have " + category_trend[rand_category_index] + ". On the other hand, for " + str(groupedLabels[rand_category_index]) + " the " + y_label + " have " + category_trend[rand_category_index - 1] + ". ")

                trend_global = categorical_global_trend[random.randint(0, len(categorical_global_trend) - 1)]
                print("Trend [global] : " + categorical_global_trend[random.randint(0, len(categorical_global_trend) - 1)])


            # sorted_max_row = sorted(max_row_val)
            # print("sorted_max_row")
            # print(sorted_max_row)

            max_gap_abs = 0
            max_gap_rel = 0
            max_gap_index = 0
            for i in range(number_of_group - 1):
                if max_row_val[i] - min_row_val[i] > max_gap_abs:
                    max_gap_abs = max_row_val[i] - min_row_val[i]
                    if min_row_val[i] == 0:
                        min_row_val[i] = 1
                    max_gap_rel = round((max_row_val[i] / min_row_val[i]), 2)
                    max_gap_index = i

            max_diff_all_cat = []
            max_diff_all_cat.append(" Out of all " + str(number_of_group) + " groups, the highest gap between the maximum and minimum " + y_label + " was found in case of " + str(groupedLabels[max_gap_index+1]) + ". ")
            max_diff_all_cat.append(" Among the groups, " + str(groupedLabels[max_gap_index+1]) + " had the biggest difference in " + y_label + ". Where the maximum " + y_label + " was " + str(max_gap_rel) + " times larger than the minimum " + y_label + ". ")
            max_diff_all_cat.append(" Among all " + str(number_of_group) + " groups, " + str(groupedLabels[max_gap_index+1]) + " had the gap of " + str(max_gap_abs) + " between the maximum and minimum " + y_label + " observed in " + str(groupedLabels[0]) + " " + max_row[max_gap_index] + " and " + min_row[max_gap_index] + " respectively. ")

            extrema_max_diff_in_cat = max_diff_all_cat[random.randint(0, len(max_diff_all_cat) - 1)]
            print("Extrema [difference in a category] : " + max_diff_all_cat[random.randint(0, len(max_diff_all_cat) - 1)])

            max_min_difference_abs = max_row_val[rand_category_index] - min_row_val[rand_category_index]
            if min_row_val[rand_category_index] != 0:
                max_min_difference_rel = round((max_row_val[rand_category_index] / min_row_val[rand_category_index]), 2)
            else:
                max_min_difference_rel = 0

            diff_in_category = []
            diff_in_category.append(" The maximum " + y_label + " for " + str(groupedLabels[rand_category_index+1]) + " that was found in " + str(groupedLabels[0]) + " " + str(max_row[rand_category_index]) + " was " + str(max_min_difference_rel) + " times larger than the minimum " + y_label + " observed in " + str(groupedLabels[0]) + " " + str(min_row[rand_category_index]) + ". ")
            diff_in_category.append(" There is a gap of " + str(max_min_difference_abs) + " between the highest and lowest " + y_label + " found for " + str(groupedLabels[rand_category_index+1]) + " in " + str(groupedLabels[0]) + " " + str(max_row[rand_category_index]) + " and " + str(min_row[rand_category_index]) + ". ")
            diff_in_category.append(str(groupedLabels[0]) + " " + str(max_row[rand_category_index]) + " and " + str(min_row[rand_category_index]) + " had the biggest gap of " + str(max_min_difference_abs) + " between the highest and lowest " + y_label + " found for " + str(groupedLabels[rand_category_index+1]) + ". ")

            comparison_categorical = diff_in_category[random.randint(0, len(diff_in_category) - 1)]
            print("Comparison [categorical] : " + diff_in_category[random.randint(0, len(diff_in_category) - 1)])

            average_stat = []
            average_stat.append("On average, the " + str(groupedLabels[0]) + " " + sortedCategories[-1][0] + " had the highest " + y_label + " for all " + str(number_of_group) + " groups " + group_names_text + ". Whereas " + sortedCategories[0][0] + " had the lowest average " + y_label + ". ")
            average_stat.append("Averaging all " + str(number_of_group) + " groups " + group_names_text + ", the " + str(groupedLabels[0]) + " " + sortedCategories[-1][0] + " is the maximum " + y_label + " and " + sortedCategories[0][0] + " is the minimum " + y_label + ". ")

            compute_der_val_avg = average_stat[random.randint(0, len(average_stat) - 1)]
            print("Compute derived val [avg] : " + average_stat[random.randint(0, len(average_stat) - 1)])

            global_extrema = []
            global_extrema.append(
                " In " + str(groupedLabels[0]) + " " + str(global_max_category_xlabel) + ", " + str(
                    global_max_category_label) + " had the highest " + y_label + " " + str(global_max) + " among the " + str(
                    number_of_group) + " groups and in " + str(global_min_category_xlabel) + ", " + str(
                    global_min_category_label) + " had the lowest " + y_label + " " + str(global_min) + ". ")
            global_extrema.append(" Out of all " + str(number_of_group) + " groups, " + str(
                global_max_category_label) + " had the highest " + y_label + " in " + str(groupedLabels[0]) + " " + str(
                global_max_category_xlabel) + " and " + str(
                global_min_category_label) + " had the lowest " + y_label + " in " + str(groupedLabels[0]) + " " + str(
                global_min_category_xlabel) + ". ")
            global_extrema.append(" " + str(groupedLabels[0]) + " " + str(
                global_max_category_xlabel) + " had the maximum " + y_label + " among all " + str(
                number_of_group) + " groups, and it was for " + str(
                global_max_category_label) + ". The minimum " + y_label + " was observed in " + str(
                groupedLabels[0]) + " " + str(global_min_category_xlabel) + " for " + str(
                global_min_category_label) + ". ")

            extrema_global = global_extrema[random.randint(0, len(global_extrema) - 1)]
            print("Extrema [global] : " + global_extrema[random.randint(0, len(global_extrema) - 1)])
            order_global = []
            if len(groupedLabels) > 3:
                order_global.append(" In " + str(groupedLabels[0]) + " " + str(global_max_category_xlabel) + ", " + str(global_max_category_label) + " had the highest " + y_label + " " + str(global_max) + " among the " + str(number_of_group) + " groups and in " + str(global_min_category_xlabel) + ", " + str(global_min_category_label) + " had the lowest " + y_label + " " + str(global_min) + ". The second highest " + y_label + " " + str(global_2nd_max) + " was observed for " + str(global_2nd_max_category_label) + " in " + str(groupedLabels[0]) + " " + str(global_2nd_max_category_xlabel) + ". ")
                order_global.append(" " + str(global_max_category_label) + " had the maximum " + y_label + " out of all " + str(number_of_group) + " groups in " + str(groupedLabels[0]) + " " + str(global_max_category_xlabel) + " followed by " + str(global_2nd_max_category_label) + " in " + str(global_2nd_max_category_xlabel) + ", and the minimum " + y_label + " is found for " + str(global_min_category_label) + " in " + str(global_min_category_xlabel) + ". ")

                order_extrema = order_global[random.randint(0, len(order_global) - 1)]
                print("Order [Extrema(max/min)] : " + order_global[random.randint(0, len(order_global) - 1)])


            x_label = str(stringLabels[0])

            intro = []

            if x_label.lower() == "month" or x_label.lower() == "year" or x_label.lower() == "months" or x_label.lower() == "years":
                intro.append("This is a grouped bar chart showing " + y_label + " on the Y-axis throughout " + str(rowCount) + " " + x_label + "s for " + categories + " on the X-axis. ")
                intro.append("This grouped bar chart represents " + y_label + " on the Y-axis. And, its value throughout " + str(rowCount) + " " + x_label + "s for " + categories + ". ")
                intro.append("This grouped bar chart represents " + y_label + " on the Y-axis. And, how the value changed throughout " + str(rowCount) + " " + x_label + "s for " + categories + ". ")
            else:
                intro.append("This grouped bar chart represents " + str(rowCount) + " different " + x_label + "s on X-axis for " + str(number_of_group) + " groups " + categories + ". On the Y-axis it shows their corresponding " + y_label + ". ")
                intro.append("This grouped bar chart shows " + y_label + " on the Y-axis for " + str(rowCount) + " different " + x_label + "s for " + str(number_of_group) + " groups " + categories + " that are presented on the X-axis. ")

            intro_summary = intro[random.randint(0, len(intro) - 1)]

            summary1 = f"This grouped bar chart has {rowCount} categories of {stringLabels[0]} on the x axis representing {str(number_of_group)} groups: {categories}."


            min_summary = []
            mid_summary = []
            max_summary = []

            min_summary.append(random.choice(intro))

            if trend_global is not None:
                min_summary.append(random.choice(categorical_global_trend))
            else:
                min_summary.append(random.choice(global_extrema))

            mid_summary.append(random.choice(intro))
            if trend_global is not None:
                mid_summary.append(random.choice(categorical_global_trend))
                mid_summary.append(random.choice(global_extrema))
                mid_summary.append(random.choice(global_max_min_categorical))

            else:
                mid_summary.append(random.choice(global_extrema))
                mid_summary.append(random.choice(global_max_min_categorical))
                mid_summary.append(random.choice(diff_in_category))

            max_summary.append(random.choice(intro))
            if trend_global is not None:
                max_summary.append(random.choice(categorical_global_trend))
                max_summary.append(random.choice(global_extrema))
                max_summary.append(random.choice(global_max_min_categorical))
                max_summary.append(random.choice(diff_in_category))
                max_summary.append(random.choice(max_diff_all_cat))
                if len(order_global) != 0:
                    max_summary.append(random.choice(order_global))
                max_summary.append(random.choice(average_stat))

            else:
                max_summary.append(random.choice(global_extrema))
                max_summary.append(random.choice(global_max_min_categorical))
                max_summary.append(random.choice(diff_in_category))
                max_summary.append(random.choice(max_diff_all_cat))
                if len(order_global) != 0:
                    max_summary.append(random.choice(order_global))
                max_summary.append(random.choice(average_stat))

            # print("SMALLEST SUMMARY")
            # print(min_summary)
            # print("MID SUMMARY")
            # print(mid_summary)
            # print("LARGEST SUMMARY")
            # print(max_summary)


            summary2 = f" Averaging these {str(number_of_group)} groups, the highest category is found for {str(groupedLabels[0])} {sortedCategories[-1][0]} with a mean value of {round(sortedCategories[-1][1], 2)}."

            summaryArray = mid_summary

            maxValueIndex = cleanValArr[0].index(sortedCategories[-1][0])
            secondValueIndex = cleanValArr[0].index(sortedCategories[-2][0])

            # if float(random.uniform(0, 1)) < 0.60:
            #     summary3 = f" Followed by {sortedCategories[-2][0]} with a mean value of {round(sortedCategories[-2][1], 2)}."
            #     summaryArray.append(summary3)

            # if topTwoDelta >= 3:
            #     summary4 = f" This represents a difference of {topTwoDelta}%."
            #     summaryArray.append(summary4)

            # summary5 = f" The lowest category is observed at {sortedCategories[0][0]} that has a mean value of {round(sortedCategories[0][1], 2)}"
            # summaryArray.append(summary5)
            #
            # if minMaxDelta >= 2:
            #     summary6 = f" showing a difference of {minMaxDelta}% with the maximum value."
            #     summaryArray.append(summary6)

            trendsArray = [
                {}, {"2": ["0", str(maxValueIndex)], "13": [str(columnCount - 1), str(maxValueIndex)]},
                {"2": ["0", str(secondValueIndex)], "14": [str(columnCount - 1), str(secondValueIndex)]}, {}
            ]
            # elif rowCount == 2:
            #     for category in categoricalValueArr:
            #         meanCategoricalDict[category[0]] = mean(category[1:])
            #     sortedCategories = sorted(meanCategoricalDict.items(), key=lambda x: x[1])
            #     numerator = abs(sortedCategories[-1][1] - sortedCategories[-2][1])
            #     denominator = (sortedCategories[-1][1] + sortedCategories[-2][1]) / 2
            #     topTwoDelta = round((numerator / denominator) * 100, 1)
            #
            #     summary1 = f"This grouped bar chart has {rowCount} categories of {stringLabels[0]} on the x axis representing {str(number_of_group)} groups: {categories}."
            #     summary2 = f" Averaging the {str(number_of_group)} groups, the highest category is found for {str(groupedLabels[0])} {sortedCategories[-1][0]} with a mean value of {round(sortedCategories[-1][1], 2)}."
            #     summaryArray.append(summary1)
            #     summaryArray.append(summary2)
            #     maxValueIndex = cleanValArr[0].index(sortedCategories[-1][0])
            #     secondValueIndex = cleanValArr[0].index(sortedCategories[-2][0])
            #     summary3 = f" The minimum category is found at {sortedCategories[-2][0]} with a mean value of {round(sortedCategories[-2][1], 2)}."
            #     summaryArray.append(summary3)
            #
            #     if topTwoDelta >= 5:
            #         summary4 = f" This represents a difference of {topTwoDelta}%."
            #         summaryArray.append(summary4)
            #
            #     summaryArray.append(chosen_summary)
            #     trendsArray = [
            #         {}, {"2": ["0", str(maxValueIndex)], "13": [str(columnCount - 1), str(maxValueIndex)]},
            #         {"2": ["0", str(secondValueIndex)], "14": [str(columnCount - 1), str(secondValueIndex)]}, {}
            #     ]
            # else:
            #     summary1 = f"This grouped bar chart has 1 category for the x axis of {stringLabels[0]}."
            #     summary2 = f" This category is {stringLabels[1]}, with a mean value of {round(mean(categoricalValueArr[1]), 2)}."
            #     summaryArray.append(summary1)
            #     summaryArray.append(summary2)
            #     summaryArray.append(chosen_summary)
            #     trendsArray = [{}, {"3": ["0", "0"], "9": ["0", "0"]}]
            websiteInput = {"title": title.strip(),
                            "labels": [' '.join(label) for label in labelArr],
                            "columnType": "multi",
                            "graphType": chartType, "summaryType": "baseline", "summary": summaryArray,
                            "xAxis": x_label,
                            "yAxis": y_label,
                            "min_summary": min_summary,
                            "mid_summary": mid_summary,
                            "max_summary": max_summary,
                            "trends": trendsArray,
                            "data": dataJson}
            with open(f'{websitePath}/{name}.json', 'w', encoding='utf-8') as websiteFile:
                json.dump(websiteInput, websiteFile, indent=3)
            # oneFile.writelines(''.join(summaryArray)+'\n')

        ## for Multi Line charts
        elif (chartType == "line"): # MULTI LINE
            # clean data
            intData = []
            # print(valueArr)
            # print(valueArr[1:])
            for line in valueArr[1:]:  # take 2nd to end elements in valueArr array
                cleanLine = []
                for data in line:

                    if data.isnumeric():
                        cleanLine.append(float(data))
                    else:
                        cleanData = re.sub("[^\d\.]", "",
                                           data)  # Delete pattern [^\d\.] from data  where [^\d\.] probably denotes digits
                        if len(cleanData) > 0:
                            cleanLine.append(
                                float(cleanData[:4]))  # character from the beginning to position 4 (excluded)
                        else:
                            cleanLine.append(float(cleanData))
                intData.append(cleanLine)
                # print(len(intData))
            # calculate mean for each line
            meanLineVals = []
            

            # print("stringLabels")
            # print(stringLabels[1:])
            # print("intData")
            # print(intData)

            assert len(stringLabels[1:]) == len(
                intData)  # tests if a condition is true. If a condition is false, the program will stop with an optional message
            for label, data in zip(stringLabels[1:],
                                   intData):  # zip output: \(('Export', [5596.0, 4562.0, 4875.0, 7140.0, 4325.0, 9188.0, 5565.0, 6574.0, 4827.0, 2945.0, 4252.0, 3876.0, 2867.0, 2404.0]), ('Import', [1087.0, 9410.0, 7853.0, 8865.0, 6917.0, 1034.0, 7262.0, 7509.0, 5715.0, 4458.0, 6268.0, 5996.0, 4299.0, 3742.0]))
                x = (label, round(mean(data), 1))  # round to 1 d.p
                # print(x)
                meanLineVals.append(x)
            sortedLines = sorted(meanLineVals, key=itemgetter(1))
            print(sortedLines)  # Ranks all the lines from bottomost to topmost using mean values
            # if more than 2 lines
            lineCount = len(labelArr) - 1  # no of categories

            # The line with higest overall mean
            maxLine = sortedLines[-1]  # the category with highest overall mean
            index1 = stringLabels.index(maxLine[0]) - 1  # index for line with max mean
            maxLineData = round(max(intData[index1]), 2)  # the max data point (y axis value) of the line with max mean
            maxXValue = valueArr[0][
                intData[index1].index(maxLineData)]  # the corrsponding x value for the above y value

            # The line with second higest overall mean
            secondLine = sortedLines[-2]  # line with second highest overall mean value
            rowIndex1 = intData[index1].index(
                maxLineData)  # the index for the max y value data point of the line with max mean
            index2 = stringLabels.index(secondLine[0]) - 1  # index for line with second max mean
            secondLineData = round(max(intData[index2]),
                                   2)  # the max data point (y axis value) of the line with max mean
            secondXValue = valueArr[0][
                intData[index2].index(secondLineData)]  ## the corrsponding x value for the above y value
            rowIndex2 = intData[index2].index(
                secondLineData)  # the index for the max y value data point of the line with second max mean

            # The line with the smallest overall mean
            minLine = sortedLines[0]
            index_min = stringLabels.index(minLine[0]) - 1
            minLineData = round(max(intData[index_min]), 2)
            minXValue = valueArr[0][intData[index_min].index(minLineData)]

            line_names = ""
            for i in range(len(stringLabels) - 1):
                if i < len(stringLabels) - 2:
                    line_names += stringLabels[i + 1] + ", "
                else:
                    line_names += "and " + stringLabels[i + 1]

            ## New Summary Template-shehnaz
            valueArrMatrix = np.array(valueArr)
            # print(valueArrMatrix)
            # valueArr_CorrectOrder= np.flip(valueArrMatrix, axis=1)

            xVal = valueArrMatrix[0, :]
            # print(xVal)

            yVals = valueArrMatrix[1:, :]
            # print(yVals)

            yVals_float = yVals
            # print(len(yVals))
            for i in range(0, len(yVals)):
                yVals_float[i] = stringToFloat(yVals[i])
            # print(yVals_float)

            yVals = np.array(yVals_float).astype(np.float)  # yVal is now in float type
            # print(yVals)

            coordinates = dict(zip(xVal, zip(*yVals)))
            # print(coordinates)

            sorted_coordinates = dict(sorted(coordinates.items()))
            # for key, value in sorted(coordinates.items()): # Note the () after items!
            #     print(key, value)
            # print("sorted_coordinates")
            # print(sorted_coordinates)

            keys, values = zip(*sorted_coordinates.items())

            # print(keys)
            # print(values)

            arr = []
            for j in range(0, len(values[0])):
                array = []
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
            yVals_sorted = np.array(arr)

            print("Sorted X vals")
            print(xVal_sorted)
            print("Sorted Y vals")
            print(yVals_sorted)

            # yVals= valueArrMatrix[1:,:].astype(np.float) # Must convert all y values to float otherwise wrong min max shows up
            # print("Y values")
            # print(yVals)

            # ##### Sort x and y values
            # xVal_sorted = np.sort(xVal)#sorted from small to big
            # print("Sorted X vals")
            # print(xVal_sorted)

            # yVals_sorted = yVals
            # for i in range(0, len(yVals)):
            #     yVals_sorted[i] = np.sort(yVals[i])
            # print("Sorted Y vals")
            # print(yVals_sorted)  # yVals_sorted is to be used from now
            # ##### End of sorting

            # yVals_sorted = yVals_float
            # print(len(yVals_float))
            # for i in range(0, len(yVals_float)):
            #     yVals_sorted[i] = np.sort(yVals_float[i])
            # print("Sorted Y vals")
            # print(yVals_sorted)

            # yValueArrCorrectOrder = yValueArr[len(yValueArr)::-1]

            # valueArr_CorrectOrder= []
            # for i in range(0, len(valueArr)):
            #     valueArr_CorrectOrder[i] = valueArr[i][len(valueArr[i])::-1]  ## Ordered correctly this time

            # yVal_array = []
            # for i in range(1, len(valueArr)):
            #     for item in valueArr[i]:
            #         yVal_array.append(float(item))

            ###### Order/Rank of all lines

            print(sortedLines)

            sortedLines_descending = sortedLines[len(sortedLines)::-1]
            print(sortedLines_descending)

            ###### Topmost Line
            # print(maxLine[0])
            # print(stringLabels.index(maxLine[0]))
            topmostLineIndex = stringLabels.index(maxLine[0]) - 1
            max_yVal_ofTopmost = max(yVals_sorted[topmostLineIndex])
            max_yValIndx_ofTopmost = get_indexes_max_value(yVals_sorted[topmostLineIndex])
            max_xVal_ofTopmost = xVal_sorted[max_yValIndx_ofTopmost]  # Is an array of xVals

            ## To concatenate commas and "and" in max_xVal_ofTopmost
            if (len(max_xVal_ofTopmost) < 2):
                max_xVal_ofTopmost = max_xVal_ofTopmost[0]
            else:
                slice1 = max_xVal_ofTopmost[:len(max_xVal_ofTopmost) - 1]
                # print(slice1)
                slice2 = ", ".join(slice1)
                slice2 += ", and " + max_xVal_ofTopmost[-1]
                # print(slice2)
                max_xVal_ofTopmost = slice2

            # print("Mean line max")
            # print(max_yVal_ofTopmost)
            # print(max_xVal_ofTopmost)

            meanOfTopmost = mean(yVals_sorted[topmostLineIndex]).round(2)
            # print(meanOfTopmost)

            ###### Bottommost Line
            # print(minLine[0])
            # print(stringLabels.index(minLine[0]))
            bottomostLineIndex = stringLabels.index(minLine[0]) - 1
            max_yVal_ofBotommost = max(yVals_sorted[bottomostLineIndex])
            max_yValIndx_ofBotommost = get_indexes_max_value(yVals_sorted[bottomostLineIndex])
            max_xVal_ofBotommost = xVal_sorted[max_yValIndx_ofBotommost]  # Is an array of xVals

            ## To concatenate commas and "and" in max_xVal_ofTopmost
            if (len(max_xVal_ofBotommost) < 2):
                max_xVal_ofBotommost = max_xVal_ofBotommost[0]
            else:
                slice1 = max_xVal_ofBotommost[:len(max_xVal_ofBotommost) - 1]
                # print(slice1)
                slice2 = ", ".join(slice1)
                slice2 += ", and " + max_xVal_ofBotommost[-1]
                # print(slice2)
                max_xVal_ofBotommost = slice2

            # print("Mean line min")
            # print(max_yVal_ofBotommost)
            # print(max_xVal_ofBotommost)

            meanOfBotommost = mean(yVals[bottomostLineIndex]).round(2)
            # print(meanOfBotommost)

            # Extrema [max, absolute, allLines]
            ## To find max of all the categories
            maxLocal_array = []
            maxLineNames = []
            maxLine_xVals = []
            num_of_xVals_max = []  # number of x values listed for each line (e.g. Suppose same max val occurred at two lines and one of those lines reached the max val twice. Then maxLine_xVals = [2010, 2013, 2016]) where 2010 and 2013 are for line 1 and 2016 for line 2. So n for line 1 is: 2 and for line 2 is: 1. So num_of_xVals will be [2,1]
            for i in range(0, len(yVals_sorted)):
                max_local = max(yVals_sorted[i])  # key=lambda x:float(x)
                maxLocal_array.append(max_local)

            # max_global= max(maxLocal_array, key=lambda x:float(x))
            # print(max_global)
            # print(maxLocal_array)
            maxLineIndex = get_indexes_max_value(maxLocal_array)  # Line which has the max value

            # print("maxLineIndex")
            # print(maxLineIndex)
            for i in range(0, len(maxLineIndex)):
                maxLineName = stringLabels[maxLineIndex[i] + 1]
                maxLineNames.append(maxLineName)
                # print(valueArr[maxLineIndex[i]+1])

                maxValIndex = get_indexes_max_value(
                    yVals_sorted[maxLineIndex[i]])  # Index at which the max value occurred for that line
                n = 0
                for j in range(0, len(maxValIndex)):
                    maxLine_xVal = xVal_sorted[maxValIndex[j]]
                    maxLine_xVals.append(maxLine_xVal)
                    n = n + 1
                num_of_xVals_max.append(n)
            # print(valueArr)

            maxLineNames = commaAnd(maxLineNames)
            maxLine_xVals = commaAnd(maxLine_xVals)

            # print("Global Max:")
            # print(maxLocal_array)
            # print(maxLineNames)
            # print(maxLine_xVals)
            # print(num_of_xVals_max)

            # Extrema [min, absolute, allLines]

            minLocal_array = []
            minLineNames = []
            minLine_xVals = []
            num_of_xVals_min = []  # number of x values listed for each line (e.g. Suppose same max val occurred at two lines and one of those lines reached the max val twice. Then maxLine_xVals = [2010, 2013, 2016]) where 2010 and 2013 are for line 1 and 2016 for line 2. So n for line 1 is: 2 and for line 2 is: 1. So num_of_xVals will be [2,1]
            for i in range(0, len(yVals_sorted)):
                min_local = min(yVals_sorted[i])  # key=lambda x:float(x)
                minLocal_array.append(min_local)

            # max_global= max(maxLocal_array, key=lambda x:float(x))
            # print(max_global)
            # print(maxLocal_array)
            minLineIndex = get_indexes_min_value(minLocal_array)  # Line which has the max value

            # print("maxLineIndex")
            # print(maxLineIndex)
            for i in range(0, len(minLineIndex)):
                minLineName = stringLabels[minLineIndex[i] + 1]
                minLineNames.append(minLineName)
                # print(valueArr[maxLineIndex[i]+1])

                minValIndex = get_indexes_min_value(
                    yVals_sorted[minLineIndex[i]])  # Index at which the max value occurred for that line
                n = 0
                for j in range(0, len(minValIndex)):
                    minLine_xVal = xVal_sorted[minValIndex[j]]
                    minLine_xVals.append(minLine_xVal)
                    n = n + 1
                num_of_xVals_min.append(n)
            # print(valueArr)

            minLineNames = commaAnd(minLineNames)
            minLine_xVals = commaAnd(minLine_xVals)

            # print("Global Max:")
            # print(minLocal_array)
            # print(minLineNames)
            # print(minLine_xVals)
            # print(num_of_xVals_min)

            ############# GlobalTrend ##############
            direction = []
            rate = []

            for i in range(0, len(yVals_sorted)):
                n = float(yVals_sorted[i][len(yVals_sorted[i]) - 1])
                o = float(yVals_sorted[i][0])
                m = max(maxLocal_array)
                globalPercentChange = percentChnageRangeFunc(n, o, m)
                rate.append(globalPercentChange)

                d = globalDirectionTrend(globalPercentChange, constant)

                # if (globalPercentChange > 0):
                #     direction.append("increased")
                # elif (globalPercentChange < 0):
                #     direction.append("decreased")
                # else:
                #     direction.append("constant")
                direction.append(d)

            lineNames = stringLabels[1:]
            # print(lineNames)
            # print(direction)
            # print(rate)

            lineNames_increasing = []
            lineNames_decreasing = []
            lineNames_constant = []
            for i in range(0, len(direction)):
                if (direction[i] == "increased"):
                    lineNames_increasing.append(lineNames[i])
                elif (direction[i] == "decreased"):
                    lineNames_decreasing.append(lineNames[i])
                else:
                    lineNames_constant.append(lineNames[i])

            # print(lineNames_increasing)
            # print(lineNames_decreasing)
            # print(lineNames_constant)

            if (len(lineNames) == 2):

                difference_arr = []
                if (len(yVals_sorted) == 2):
                    for i in range(0, len(xVal_sorted)):
                        diff = yVals_sorted[0][i] - yVals_sorted[1][i]
                        difference_arr.append(diff)
                # print(difference_arr)

                abs_difference_arr = []
                for i in range(0, len(difference_arr)):
                    abs_difference_arr.append(abs(difference_arr[i]))
                # print(abs_difference_arr)

                constant_rate = 5
                diffPercentChange = percentChnageFunc(abs_difference_arr[-1], abs_difference_arr[0])
                diff_direction = directionTrend(abs_difference_arr[-1], abs_difference_arr[0], constant_rate)
                # print(diffPercentChange)
                # print(diff_direction)

                if (diff_direction == "increasing"):
                    diff_direction = "greater"
                elif (diff_direction == "decreasing"):
                    diff_direction = "smaller"
                else:
                    diff_direction = "roughly same"

                # Find and report the max and the min gap between two Lines
                max_diff = max(abs_difference_arr)
                max_diff_indx = get_indexes_max_value(abs_difference_arr)

                min_diff = min(abs_difference_arr)
                min_diff_indx = get_indexes_min_value(abs_difference_arr)

            # Global Trends with rate of change

            globalTrends = []
            # print(constant)
            # print(gradual)
            # print(rapid)
            for i in rate:
                rate = globalRateOfChange(i, constant, gradual, rapid)
                globalTrends.append(rate)
            # print(globalTrends)

            lineNames = stringLabels[1:]
            # print(lineNames)
            # print(direction)
            # print(rate)
            # print(globalTrends)

            lineNames_increasing_r = []
            lineNames_increasing_g = []

            lineNames_decreasing_r = []
            lineNames_decreasing_g = []

            lineNames_constant_c = []

            for i in range(0, len(direction)):
                if (direction[i] == "increasing"):
                    if (globalTrends[i] == "rapidly"):
                        lineNames_increasing_r.append(lineNames[i])
                    else:
                        lineNames_increasing_g.append(lineNames[i])
                elif (direction[i] == "decreasing"):
                    if (globalTrends[i] == "rapidly"):
                        lineNames_decreasing_r.append(lineNames[i])
                    else:
                        lineNames_decreasing_g.append(lineNames[i])
                else:
                    lineNames_constant_c.append(lineNames[i])

            print(direction)

            print(lineNames_increasing_r)
            print(lineNames_increasing_g)
            print(lineNames_decreasing_r)
            print(lineNames_decreasing_g)
            print(lineNames_constant_c)

            # For rapidly incresing lines report percentage increase or factor of increase
            percentChng_in = []
            factorChng_in = []

            if (len(lineNames_increasing_r) != 0):
                for i in range(0, len(lineNames_increasing_r)):
                    indx = lineNames.index(lineNames_increasing_r[i])
                    n = float(yVals_sorted[indx][len(yVals_sorted[indx]) - 1])
                    o = float(yVals_sorted[indx][0])

                    p = abs(percentChnageFunc(n, o))
                    f = round(n / o, 1)
                    percentChng_in.append(p)
                    factorChng_in.append(f)

            print(percentChng_in)
            print(factorChng_in)

            # For rapidly decreasing lines report percentage decrease or factor of decrease
            percentChng_de = []
            factorChng_de = []

            if (len(lineNames_decreasing_r) != 0):
                for i in range(0, len(lineNames_decreasing_r)):
                    indx = lineNames.index(lineNames_decreasing_r[i])
                    n = float(yVals_sorted[indx][len(yVals_sorted[indx]) - 1])
                    o = float(yVals_sorted[indx][0])

                    p = abs(percentChnageFunc(n, o))
                    f = round(n / o, 1)
                    percentChng_de.append(p)
                    factorChng_de.append(f)

            print(percentChng_de)
            print(factorChng_de)

            percentChngSumm = ""
            factorChngSumm = ""

            # Line that are rapidly increasing
            if (len(lineNames_increasing_r) > 1):
                percentChngSumm += commaAnd(lineNames_increasing_r) + " has increased by " + str(
                    commaAnd(percentChng_in)) + " percent respectively."
                factorChngSumm += commaAnd(lineNames_increasing_r) + " has increased by " + str(
                    commaAnd(factorChng_in)) + " times respectively."
                # globalTrendRate_summary.append(summary_increasing_r)
            elif (len(lineNames_increasing_r) == 1):
                percentChngSumm += commaAnd(lineNames_increasing_r) + " has increased by " + str(
                    commaAnd(percentChng_in)) + " percent."
                factorChngSumm += commaAnd(lineNames_increasing_r) + " has increased by " + str(
                    commaAnd(factorChng_in)) + " times."
                # globalTrendRate_summary.append(summary_increasing_r)

            # Line that are rapidly decreasing
            if (len(lineNames_decreasing_r) > 1):
                percentChngSumm += commaAnd(lineNames_decreasing_r) + " has increased by " + str(
                    commaAnd(percentChng_de)) + " percent respectively."
                factorChngSumm += commaAnd(lineNames_decreasing_r) + " has increased by " + str(
                    commaAnd(factorChng_de)) + " times respectively."
                # globalTrendRate_summary.append(summary_increasing_r)
            elif (len(lineNames_decreasing_r) == 1):
                percentChngSumm += commaAnd(lineNames_decreasing_r) + " has increased by " + str(
                    commaAnd(percentChng_de)) + " percent."
                factorChngSumm += commaAnd(lineNames_decreasing_r) + " has increased by " + str(
                    commaAnd(factorChng_de)) + " times."
                # globalTrendRate_summary.append(summary_increasing_r)

            print(percentChngSumm)
            print(factorChngSumm)

            chnageFactor = [percentChngSumm, factorChngSumm]
            selectedChange = random.choice(chnageFactor)
            # print(selectedChange)

            # Done by jason
            summaryArr = []

            summary1 = "This is a multi-line chart with " + str(lineCount) + " lines representing " + line_names + ". "
            # summary2 = "The line for " + str(maxLine[0]) + " has the highest values across " + str(
            #     stringLabels[0]) + " with a mean value of " + str(maxLine[1]) + ", "
            summaryArr.append(summary1)

            #### Order/Ranking of all lines given total no of lines is < 5
            if (len(sortedLines_descending) < 5):  # Given there are no more than 5 lines
                summary_rank = "The ranking of the lines from topmost to botommmost is as follows: "
                for i in range(0, len(sortedLines_descending) - 1):
                    summary_rank += str(i + 1) + ". " + sortedLines_descending[i][0] + ", "
                summary_rank += "and lastly, " + str(len(sortedLines_descending)) + ". " + \
                                sortedLines_descending[len(sortedLines_descending) - 1][0] + ". "
                summaryArr.append(summary_rank)

            ## Talks about the topmost line
            summary2 = "Throughout the period, " + str(maxLine[
                                                           0]) + " mostly maintained the highest " + "y-axis name" + " when compared to others" + " with a mean value of " + str(
                meanOfTopmost) + ", "
            # summary3 = "and it peaked at " + str(maxXValue) + " with a value of " + str(maxLineData) + "."
            summary3 = "and it peaked at " + str(max_xVal_ofTopmost) + " with a value of " + str(
                max_yVal_ofTopmost) + "."  # revised

            summaryArr.append(summary2)
            summaryArr.append(summary3)

            ## Talks about the second topmost line
            if lineCount > 2:
                summary4 = "After " + str(maxLine[0]) + ", " + str(
                    secondLine[0]) + " overall has the second highest values " + ", with a mean value of " + str(
                    secondLine[1]) + ", "
                summary5 = "peaking at " + str(secondXValue) + " with a value of " + str(secondLineData) + ". "
                summaryArr.append(summary4)
                summaryArr.append(summary5)

            ## Talks about the bottomost line
            summary6 = str(minLine[0]) + " mostly had the least" + " y-axis name" + " with a mean value of " + str(
                meanOfBotommost) + ", "
            summary7 = "which peaked at " + str(max_xVal_ofBotommost) + " with a value of " + str(
                max_yVal_ofBotommost) + ". "
            summaryArr.append(summary6)
            summaryArr.append(summary7)
            
            #Additional summaries -shehnaz
            
            #Global Max
            if (max_yVal_ofTopmost != max(maxLocal_array) and len(maxLine_xVals)<5):
                summary8 = maxLineNames + " reported the highest " + "y-axis name " + " about " + str(max(maxLocal_array)) + " in " + stringLabels[0] + " " + maxLine_xVals
                summaryArr.append(summary8)
             
            #Global Min
            if (len(minLine_xVals)<5): #given no more than 5 x values are reported
                summary9 = minLineNames + " reported the lowest " + "y-axis name " + " about " + str(min(minLocal_array)) + " in " + stringLabels[0] + " " + minLine_xVals
                summaryArr.append(summary9)
            
            
            #### Global Trend without rate
            
            # #Lines that increase
            # summary_increasing= "Overall "
            # if (len(lineNames_increasing)>1):
            #     summary_increasing+= commaAnd(lineNames_increasing) + " are increasing throughout the " + stringLabels[0]
            #     summaryArr.append(summary_increasing)
            # elif(len(lineNames_increasing)==1):
            #     summary_increasing+= commaAnd(lineNames_increasing) + "is increasing throughout the " + stringLabels[0]
            #     summaryArr.append(summary_increasing)
            
            
            # #Lines that decrease
            # summary_decreasing= "Overall "
            # if (len(lineNames_decreasing)>1):
            #     summary_decreasing+= commaAnd(lineNames_decreasing) + " are decreasing throughout the " + stringLabels[0]
            #     summaryArr.append(summary_decreasing)
            # elif(len(lineNames_decreasing)==1):
            #     summary_decreasing+= commaAnd(lineNames_decreasing) + "is decreasing throughout the " + stringLabels[0]
            #     summaryArr.append(summary_decreasing)

            
            # # Lines that stay constant
            # summary_constant= "Overall "
            # if (len(lineNames_constant)>1):
            #     summary_constant+= commaAnd(lineNames_constant) + " are roughly constant throughout the " + stringLabels[0]
            #     summaryArr.append(summary_constant)
            # elif(len(lineNames_constant)==1):
            #     summary_constant+= commaAnd(lineNames_constant) + "is roughly constant throughout the " + stringLabels[0]
            #     summaryArr.append(summary_constant)
            
            
            
            
            ###### Global Trends with Rate of chnage
            
            globalTrendRate_summary= "Overall "

            
            #Lines that rapidly increase
            # summary_increasing_r= ""
            if (len(lineNames_increasing_r)>1):
                globalTrendRate_summary+= commaAnd(lineNames_increasing_r) + " are rapidly increasing, "
                # globalTrendRate_summary.append(summary_increasing_r)
            elif(len(lineNames_increasing_r)==1):
                globalTrendRate_summary+= commaAnd(lineNames_increasing_r) + " is rapidly increasing, "
                # globalTrendRate_summary.append(summary_increasing_r)
            
            #Lines that gradually increase
            # summary_increasing_g= ""
            if (len(lineNames_increasing_g)>1):
                globalTrendRate_summary+= commaAnd(lineNames_increasing_g) + " are gradually increasing, "
                # globalTrendRate_summary.append(summary_increasing_g)
            elif(len(lineNames_increasing_g)==1):
                globalTrendRate_summary+= commaAnd(lineNames_increasing_g) + " is gradually increasing, "
                # globalTrendRate_summary.append(summary_increasing_g)
            
            #Lines that rapidly decrease
            # summary_decreasing_r= ""
            if (len(lineNames_decreasing_r)>1):
                globalTrendRate_summary+= commaAnd(lineNames_decreasing_r) + " are rapidly decreasing, "
                # globalTrendRate_summary.append(lineNames_decreasing_r)
            elif(len(lineNames_decreasing_r)==1):
                globalTrendRate_summary+= commaAnd(lineNames_decreasing_r) + " is rapidly decreasing, "
                # globalTrendRate_summary.append(lineNames_decreasing_r)
            
            #Lines that gradually decrease
            # summary_decreasing_g= ""
            if (len(lineNames_decreasing_g)>1):
                globalTrendRate_summary+= commaAnd(lineNames_decreasing_g) + " are gradually decreasing, "
                # globalTrendRate_summary.append(lineNames_decreasing_g)
            elif(len(lineNames_decreasing_g)==1):
                globalTrendRate_summary+= commaAnd(lineNames_decreasing_g) + " is gradually decreasing, "
                # globalTrendRate_summary.append(lineNames_decreasing_g)

            # Lines that stay constant
            # summary_constant_c= ""
            if (len(lineNames_constant_c)>1):
                globalTrendRate_summary+= commaAnd(lineNames_constant_c) + " are roughly constant, " 
                # globalTrendRate_summary.append(summary_constant_c)
            elif(len(lineNames_constant_c)==1):
                globalTrendRate_summary+= commaAnd(lineNames_constant_c) + " is roughly constant, " 
                # globalTrendRate_summary.append(summary_constant_c)
                
            globalTrendRate_summary+= " throughout the " + stringLabels[0]
            summaryArr.append(globalTrendRate_summary)
            
            
            # Append randomly the factor of chnage given the chnage was rapid
            if (len(lineNames_increasing_r)!=0 or len(lineNames_decreasing_r)!=0):
                summaryArr.append(selectedChange)
            
            
            
            ###### The gap between two lines    
            if (len(lineNames)==2):
                summary10 = "The difference of "+ "y_axis name"+ " between "  + lineNames[0] + " and " + lineNames[1] + " is " + diff_direction + " at "+ stringLabels[0]+" " + xVal_sorted[-1]+ " compared to the " + stringLabels[0] +" "  +xVal_sorted[0] + "."
                summaryArr.append(summary10)
                
                summary11 = "The greatest difference of "+ "y_axis name"+ " between "  + lineNames[0] + " and " + lineNames[1] + " occurs at "+ stringLabels[0]+" " +  str(xVal_sorted[max_diff_indx[0]]) + " and the smallest difference occurs at " + str(xVal_sorted[min_diff_indx[0]]) + "." # Assumes there is only one max and min gap or difference
                summaryArr.append(summary11)
                
            
            
            
            
                
                
            
            
            
            
            
            
            
            
            
            

            # Additional summaries -shehnaz

            # Global Max
            if (max_yVal_ofTopmost != max(maxLocal_array) and len(maxLine_xVals) < 5):
                summary8 = maxLineNames + " reported the highest " + "y-axis name " + " about " + str(
                    max(maxLocal_array)) + " in " + stringLabels[0] + " " + maxLine_xVals
                summaryArr.append(summary8)

            # Global Min
            if (len(minLine_xVals) < 5):  # given no more than 5 x values are reported
                summary9 = minLineNames + " reported the lowest " + "y-axis name " + " about " + str(
                    min(minLocal_array)) + " in " + stringLabels[0] + " " + minLine_xVals
                summaryArr.append(summary9)

            #### Global Trend without rate

            # #Lines that increase
            # summary_increasing= "Overall "
            # if (len(lineNames_increasing)>1):
            #     summary_increasing+= commaAnd(lineNames_increasing) + " are increasing throughout the " + stringLabels[0]
            #     summaryArr.append(summary_increasing)
            # elif(len(lineNames_increasing)==1):
            #     summary_increasing+= commaAnd(lineNames_increasing) + "is increasing throughout the " + stringLabels[0]
            #     summaryArr.append(summary_increasing)

            # #Lines that decrease
            # summary_decreasing= "Overall "
            # if (len(lineNames_decreasing)>1):
            #     summary_decreasing+= commaAnd(lineNames_decreasing) + " are decreasing throughout the " + stringLabels[0]
            #     summaryArr.append(summary_decreasing)
            # elif(len(lineNames_decreasing)==1):
            #     summary_decreasing+= commaAnd(lineNames_decreasing) + "is decreasing throughout the " + stringLabels[0]
            #     summaryArr.append(summary_decreasing)

            # # Lines that stay constant
            # summary_constant= "Overall "
            # if (len(lineNames_constant)>1):
            #     summary_constant+= commaAnd(lineNames_constant) + " are roughly constant throughout the " + stringLabels[0]
            #     summaryArr.append(summary_constant)
            # elif(len(lineNames_constant)==1):
            #     summary_constant+= commaAnd(lineNames_constant) + "is roughly constant throughout the " + stringLabels[0]
            #     summaryArr.append(summary_constant)

            ###### Global Trends with Rate of chnage

            globalTrendRate_summary = "Overall "

            # Lines that rapidly increase
            # summary_increasing_r= ""
            if (len(lineNames_increasing_r) > 1):
                globalTrendRate_summary += commaAnd(lineNames_increasing_r) + " are rapidly increasing, "
                # globalTrendRate_summary.append(summary_increasing_r)
            elif (len(lineNames_increasing_r) == 1):
                globalTrendRate_summary += commaAnd(lineNames_increasing_r) + " is rapidly increasing, "
                # globalTrendRate_summary.append(summary_increasing_r)

            # Lines that gradually increase
            # summary_increasing_g= ""
            if (len(lineNames_increasing_g) > 1):
                globalTrendRate_summary += commaAnd(lineNames_increasing_g) + " are gradually increasing, "
                # globalTrendRate_summary.append(summary_increasing_g)
            elif (len(lineNames_increasing_g) == 1):
                globalTrendRate_summary += commaAnd(lineNames_increasing_g) + " is gradually increasing, "
                # globalTrendRate_summary.append(summary_increasing_g)

            # Lines that rapidly decrease
            # summary_decreasing_r= ""
            if (len(lineNames_decreasing_r) > 1):
                globalTrendRate_summary += commaAnd(lineNames_decreasing_r) + " are rapidly decreasing, "
                # globalTrendRate_summary.append(lineNames_decreasing_r)
            elif (len(lineNames_decreasing_r) == 1):
                globalTrendRate_summary += commaAnd(lineNames_decreasing_r) + " is rapidly decreasing, "
                # globalTrendRate_summary.append(lineNames_decreasing_r)

            # Lines that gradually decrease
            # summary_decreasing_g= ""
            if (len(lineNames_decreasing_g) > 1):
                globalTrendRate_summary += commaAnd(lineNames_decreasing_g) + " are gradually decreasing, "
                # globalTrendRate_summary.append(lineNames_decreasing_g)
            elif (len(lineNames_decreasing_g) == 1):
                globalTrendRate_summary += commaAnd(lineNames_decreasing_g) + " is gradually decreasing, "
                # globalTrendRate_summary.append(lineNames_decreasing_g)

            # Lines that stay constant
            # summary_constant_c= ""
            if (len(lineNames_constant_c) > 1):
                globalTrendRate_summary += commaAnd(lineNames_constant_c) + " are roughly constant, "
                # globalTrendRate_summary.append(summary_constant_c)
            elif (len(lineNames_constant_c) == 1):
                globalTrendRate_summary += commaAnd(lineNames_constant_c) + " is roughly constant, "
                # globalTrendRate_summary.append(summary_constant_c)

            globalTrendRate_summary += " throughout the " + stringLabels[0]
            summaryArr.append(globalTrendRate_summary)

            # Append randomly the factor of chnage given the chnage was rapid
            if (len(lineNames_increasing_r) != 0 or len(lineNames_decreasing_r) != 0):
                summaryArr.append(selectedChange)

            ###### The gap between two lines
            if (len(lineNames) == 2):
                summary10 = "The difference of " + "y_axis name" + " between " + lineNames[0] + " and " + lineNames[
                    1] + " is " + diff_direction + " at " + stringLabels[0] + " " + xVal_sorted[
                                -1] + " compared to the " + stringLabels[0] + " " + xVal_sorted[0] + "."
                summaryArr.append(summary10)

                summary11 = "The greatest difference of " + "y_axis name" + " between " + lineNames[0] + " and " + \
                            lineNames[1] + " occurs at " + stringLabels[0] + " " + str(
                    xVal_sorted[max_diff_indx[0]]) + " and the smallest difference occurs at " + str(
                    xVal_sorted[min_diff_indx[0]]) + "."  # Assumes there is only one max and min gap or difference
                summaryArr.append(summary11)

            trendsArray = [{},
                           {"2": ["0", str(index1)], "16": [str(rowCount - 1), str(index1)]},
                           {"1": [str(rowIndex1), str(index1)], "9": [str(rowIndex1), str(index1)]},
                           {"2": ["0", str(index2)], "15": [str(rowCount - 1), str(index2)]},
                           {"1": [str(rowIndex2), str(index2)], "10": [str(rowIndex2), str(index2)]}
                           ]
            websiteInput = {"title": title.strip(), "labels": [' '.join(label) for label in labelArr],
                            "columnType": "multi",
                            "graphType": chartType, "summaryType": "baseline", "summary": summaryArr,
                            "trends": trendsArray,
                            "data": dataJson}

            print(summaryArr)

            with open(f'{websitePath}/{name}.json', 'w', encoding='utf-8') as websiteFile:
                json.dump(websiteInput, websiteFile, indent=3)

            # oneFile.writelines(''.join(summaryArr)+'\n')

    else:
        xValueArr = []
        yValueArr = []
        cleanXArr = []
        cleanYArr = []
        xLabel = ' '.join(datum[0].split('|')[0].split('_'))
        yLabel = ' '.join(datum[1].split('|')[0].split('_'))
        chartType = datum[0].split('|')[3].split('_')[0]

        # fp.write(str(name) + "\t" + str(yLabel) + "\n")
        # fp.close()

        print(xLabel)
        print(yLabel)
        print(chartType)

        for i in range(0, len(datum)):
            if i % 2 == 0:
                xValueArr.append((datum[i].split('|')[1]))
                cleanXArr.append((datum[i].split('|')[1].replace('_', ' ')))
            else:
                yValueArr.append(float(re.sub("[^\d\.]", "", datum[i].split('|')[1])))
                cleanYArr.append(float(re.sub("[^\d\.]", "", datum[i].split('|')[1])))

        titleArr = title.split()
        maxValue = str(max(yValueArr))
        minValue = str(min(yValueArr))
        maxValueIndex = pd.Series(yValueArr).idxmax()
        minValueIndex = pd.Series(yValueArr).idxmin()
        summaryArray = []

        totalValue = sum(yValueArr)
        avgValueOfAllBars = totalValue / len(yValueArr)

        # print("totalValue -> " + str(totalValue))
        # print("avgValueOfAllBars -> " + str(avgValueOfAllBars))

        maxPercentage = int(math.ceil((max(yValueArr) / totalValue) * 100.00))
        minPercentage = int(math.ceil((min(yValueArr) / totalValue) * 100.00))

        if len(xValueArr) > 2:

            sortedDataY = sorted(yValueArr, reverse=True)
            secondMaxPercentage = int(math.ceil((int(sortedDataY[1]) / totalValue) * 100))

            secondMaxIndex = 0
            thirdMaxIndex = 0

            for a in range(len(yValueArr)):
                if yValueArr[a] == sortedDataY[1]:
                    secondMaxIndex = a
                if yValueArr[a] == sortedDataY[2]:
                    thirdMaxIndex = a

            position_in_X_axis_for_second_max_value = str(xValueArr[secondMaxIndex])
            position_in_X_axis_for_second_max_value = position_in_X_axis_for_second_max_value.replace("_", " ")
            y_axis_for_second_max_value = str(yValueArr[secondMaxIndex])

            position_in_X_axis_for_third_max_value = str(xValueArr[thirdMaxIndex]).replace("_", " ")
            y_axis_for_third_max_value = str(yValueArr[thirdMaxIndex])
            
        position_in_X_axis_for_second_max_value = ""  # Added to deal with following error: UnboundLocalError: local variable 'secondMaxIndex' referenced before assignment
        num_of_category = str(len(xValueArr))
        position_in_X_axis_for_max_value = str(xValueArr[maxValueIndex])
        position_in_X_axis_for_max_value = position_in_X_axis_for_max_value.replace("_", " ")
        y_axis_for_max_value = str(yValueArr[maxValueIndex])

        position_in_X_axis_for_min_value = str(xValueArr[minValueIndex])
        position_in_X_axis_for_min_value = position_in_X_axis_for_min_value.replace("_", " ")
        y_axis_for_min_value = str(yValueArr[minValueIndex])

        if (chartType == "pie" or chartType == "bar"):
            if type(yValueArr[maxValueIndex]) == int or type(yValueArr[maxValueIndex]) == float:

                # proportion = int(math.ceil(yValueArr[maxValueIndex] / yValueArr[minValueIndex]))
                # proportion = round((yValueArr[maxValueIndex] / yValueArr[minValueIndex]), 2)
                try:
                    proportion = round((yValueArr[maxValueIndex] / yValueArr[minValueIndex]), 2)
                except ZeroDivisionError:
                    proportion = round((yValueArr[maxValueIndex] / 0.00000000001), 2)  # To avoid x/0 math error

                max_avg_diff_rel = round((yValueArr[maxValueIndex] / avgValueOfAllBars), 2)
                max_min_diff = (yValueArr[maxValueIndex] - yValueArr[minValueIndex])
                max_avg_diff_abs = (yValueArr[maxValueIndex] - avgValueOfAllBars)
                median_val = median(yValueArr)

                # print("proportion -> " + str(proportion))
                # print("max_min_diff -> " + str(max_min_diff))
                # print("max_avg_diff_rel -> " + str(max_avg_diff_rel))
                # print("max_avg_diff -> " + str(max_avg_diff_abs))
            else:
                print('The variable is not a number')

        # run pie
        if (chartType == "pie"):

            summary1 = "This is a pie chart showing the distribution of " + str(
                len(xValueArr)) + " different " + xLabel + "."
            summary2 = xValueArr[maxValueIndex] + " " + xLabel + " has the highest proportion with " + str(
                maxPercentage) + "% of the pie chart area"
            summary3 = "followed by " + xLabel + " " + xValueArr[secondMaxIndex] + ", with a proportion of " + str(
                secondMaxPercentage) + "%. "
            summary4 = "Finally, " + xLabel + " " + xValueArr[
                minValueIndex] + " has the minimum contribution of " + str(minPercentage) + "%."

            summaryArray.append(summary1)
            summaryArray.append(summary2)
            summaryArray.append(summary3)
            summaryArray.append(summary4)

            dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
            trendsArray = [{}]
            websiteInput = {"title": title, "name": xLabel, "percent": yLabel,
                            "columnType": "two",
                            "graphType": chartType, "summaryType": "baseline", "summary": summaryArray,
                            "trends": trendsArray,
                            "data": dataJson}
            with open(f'{websitePath}/{name}.json', 'w', encoding='utf-8') as websiteFile:
                json.dump(websiteInput, websiteFile, indent=3)

        # run bar
        elif (chartType == "bar"):
            secondMaxIndex= 0  # to deal with error: local variable 'secondMaxIndex' referenced before assignment

            intro = []
            intro.append("This is a bar chart representing " + xLabel + " in the x axis and " + yLabel + " in the y axis. ")
            intro.append("This bar chart has " + str(len(xValueArr)) + " columns on the x axis representing " + xLabel + ", and " + yLabel + " in each " + xLabel + " on the y axis. ")
            intro.append("This is a bar chart. It shows " + yLabel + " for " + str(len(xValueArr)) + " number of " + xLabel + "s. ")
            print("INTRO : " + intro[random.randint(0, len(intro) -1)])

            summaryArray.append(intro[random.randint(0, len(intro) -1)])

            # Extrema [max/min]

            summary2_extrema_max_min = []
            summary2_extrema_max_min.append(
                "The maximum " + yLabel + " " + str(yValueArr[maxValueIndex]) + " is found at " + xLabel + " " + position_in_X_axis_for_max_value + " and the minimum is found at " + position_in_X_axis_for_min_value + " where " + xLabel + " is " + str(yValueArr[minValueIndex]) + ". ")
            summary2_extrema_max_min.append(
                "The " + yLabel + " is highest at " + xLabel + " " + position_in_X_axis_for_max_value + " and lowest at " + xLabel + " " + position_in_X_axis_for_min_value + ". ")
            summary2_extrema_max_min.append(
                xLabel + " " + position_in_X_axis_for_max_value + " has the highest " + yLabel + " and " + position_in_X_axis_for_min_value + " has the lowest " + yLabel + ". ")
            summary2_extrema_max_min.append(
                "The " + yLabel + " is appeared to be the highest at " + xLabel + " " + position_in_X_axis_for_max_value + " and lowest at " + xLabel + " " + position_in_X_axis_for_min_value + ". ")

            print(
                "Extrema [max/min] : " + summary2_extrema_max_min[random.randint(0, len(summary2_extrema_max_min) - 1)])

            summaryArray.append(summary2_extrema_max_min[random.randint(0, len(summary2_extrema_max_min) - 1)])

            global_trend_text = []
            # Trend [Pos/Neg]
            if xLabel.lower() == "year" or xLabel.lower() == "years" or xLabel.lower() == "month" or xLabel.lower() == "months" or xLabel.lower() == "quarter" or xLabel.lower() == "quarters":
                single_bar_trend = globalTrendBarChart(yValueArr)

                global_trend_text.append(
                    "Overall " + yLabel + " has " + single_bar_trend + " over the " + xLabel + "s. ")
                global_trend_text.append("The " + yLabel + " has " + single_bar_trend + " over the past " + str(
                    len(yValueArr)) + " " + xLabel + "s. ")
                global_trend_text.append("Over the past " + str(
                    len(yValueArr)) + " " + xLabel + "s, the " + yLabel + " has " + single_bar_trend + ". ")

                print("Trend [Pos/Neg] : " + global_trend_text[random.randint(0, len(global_trend_text) - 1)])
                summaryArray.append(global_trend_text[random.randint(0, len(global_trend_text) - 1)])

            # Order [position]
            summary3_order_2nd_max = []
            if len(xValueArr) > 2:
                summary3_order_2nd_max.append("The second highest " + yLabel + " is appeared to be the " + xLabel + " " + position_in_X_axis_for_second_max_value + ". ")
                summary3_order_2nd_max.append("Second maximum " + yLabel + " is found at " + xLabel + " " + position_in_X_axis_for_second_max_value + ". ")
                summary3_order_2nd_max.append(xLabel + " " + position_in_X_axis_for_second_max_value + " has the second highest value for " + yLabel + ". ")
                print("Order [position] : " + summary3_order_2nd_max[random.randint(0, len(summary3_order_2nd_max) - 1)])

            # Order [rank]
            summary_order_rank = []
            if len(xValueArr) > 3:
                summary_order_rank.append("The " + xLabel + " " + position_in_X_axis_for_max_value + " has the highest " + yLabel + ", followed by " + position_in_X_axis_for_second_max_value + ", and " + position_in_X_axis_for_third_max_value + ". Down to the " + xLabel + " " + position_in_X_axis_for_min_value + " which is the lowest. ")
                summary_order_rank.append(xLabel + " " + position_in_X_axis_for_max_value + " is higher than any other " + xLabel + "s with value " + str(yValueArr[maxValueIndex]) + ", followed by " + position_in_X_axis_for_second_max_value + ", and " + position_in_X_axis_for_third_max_value + ". Down to " + xLabel + " " + position_in_X_axis_for_min_value + " with the lowest value " + str(yValueArr[minValueIndex]) + ". ")
                summary_order_rank.append(yLabel + " at " + xLabel + " " + position_in_X_axis_for_max_value + " is " + str(yValueArr[maxValueIndex]) + " , second place is " + position_in_X_axis_for_second_max_value + " at " + str(yValueArr[secondMaxIndex]) + ", and thirdly is " + position_in_X_axis_for_third_max_value + " at " + str(yValueArr[thirdMaxIndex]) + ".")

                print("Order [rank] : " + summary_order_rank[random.randint(0, len(summary_order_rank) - 1)])
                summaryArray.append(summary_order_rank[random.randint(0, len(summary_order_rank) - 1)])


            # Comparison [Absolute]
            comparison_abs = []
            comparison_abs.append("There is a difference of " + str(round(max_min_diff, 2)) + " between the maximum " + xLabel + " " + position_in_X_axis_for_max_value + " and minimum " + xLabel + " " + position_in_X_axis_for_min_value + ". ")
            comparison_abs.append("The difference of " + yLabel + " between the highest and lowest " + xLabel + " is " + str(round(max_min_diff, 2)) + ". ")
            comparison_abs.append("The highest " + xLabel + " " + position_in_X_axis_for_max_value + " has " + str(round(max_min_diff, 2)) + " more " + yLabel + " than the lowest " + xLabel + " " + position_in_X_axis_for_min_value + ". ")

            print("Comparison [Absolute] : " + comparison_abs[random.randint(0, len(comparison_abs) - 1)])

            # Comparison [Relative]
            comparison_rel = []
            comparison_rel.append(xLabel + " " + position_in_X_axis_for_max_value + " has " + str(proportion) + " times more " + yLabel + " than " + xLabel + " " + position_in_X_axis_for_min_value + " which is has the lowest. ")
            comparison_rel.append(xLabel + " " + position_in_X_axis_for_min_value + " has " + str(proportion) + " times less " + yLabel + " than " + xLabel + " " + position_in_X_axis_for_max_value + " which is the highest. ")
            comparison_rel.append("The highest value at " + xLabel + " " + position_in_X_axis_for_max_value + " is " + str(proportion) + "x times more than the lowest value at " + position_in_X_axis_for_min_value + ". ")
            comparison_rel.append("The lowest value at " + xLabel + " " + position_in_X_axis_for_min_value + " is " + str(proportion) + "x times less than the highest value at " + position_in_X_axis_for_max_value + ". ")
            comparison_rel.append("The " + yLabel + " of " + xLabel + " " + position_in_X_axis_for_max_value + " is " + str(proportion) + "% larger than the minimum value at " + position_in_X_axis_for_min_value + ". ")
            comparison_rel.append("The " + yLabel + " of " + xLabel + " " + position_in_X_axis_for_min_value + " is " + str(proportion) + "% smaller than the maximum value at " + position_in_X_axis_for_max_value + ". ")
            comparison_rel.append("The maximum " + xLabel + " " + position_in_X_axis_for_max_value + " has got " + str(proportion) + " times higher " + yLabel + " than the minimum " + xLabel + " " + position_in_X_axis_for_min_value + ". ")
            comparison_rel.append("The minimum " + xLabel + " " + position_in_X_axis_for_min_value + " has got " + str(proportion) + " times less " + yLabel + " than the maximum " + xLabel + " " + position_in_X_axis_for_max_value + ". ")

            print("Comparison [Relative] : " + comparison_rel[random.randint(0, len(comparison_rel) - 1)])

            if float(random.uniform(0, 1)) > 0.75:
                summaryArray.append(comparison_rel[random.randint(0, len(comparison_rel) - 1)])
            else:
                summaryArray.append(comparison_abs[random.randint(0, len(comparison_abs) - 1)])

            # Compute derived val [avg]
            derived_val_avg = []
            derived_val_avg.append(
                "The average " + yLabel + " in all " + str(len(yValueArr)) + " " + xLabel + "s is " + str(round(avgValueOfAllBars , 2)) + ". ")
            derived_val_avg.append("Considered together, the average " + yLabel + " in all " + str(
                len(yValueArr)) + " " + xLabel + "s is roughly " + str(round(avgValueOfAllBars , 2)) + ". ")

            print("Compute derived val [avg] : " + derived_val_avg[random.randint(0, len(derived_val_avg) - 1)])

            # Comparison [Relative, vs Avg]
            comparison_rel_with_avg = []
            comparison_rel_with_avg.append("The highest value " + str(yValueArr[maxValueIndex]) + " at " + position_in_X_axis_for_max_value + " is almost " + str(max_avg_diff_rel) + " times larger than the average value " + str(round(avgValueOfAllBars, 2)) + ". ")
            comparison_rel_with_avg.append("The lowest value " + str(yValueArr[minValueIndex]) + " at " + position_in_X_axis_for_min_value + " is almost " + str(max_avg_diff_rel) + " times smaller than the average value " + str(round(avgValueOfAllBars, 2)) + ". ")
            comparison_rel_with_avg.append("The " + xLabel + " " + position_in_X_axis_for_max_value + " has " + str(max_avg_diff_rel) + " times more " + yLabel + " than average. ")
            comparison_rel_with_avg.append("The " + xLabel + " " + position_in_X_axis_for_min_value + " has " + str(max_avg_diff_rel) + " times less " + yLabel + " than average. ")
            comparison_rel_with_avg.append("The " + xLabel + " " + position_in_X_axis_for_max_value + " tends to be " + str(max_avg_diff_rel) + " percent higher than average. ")
            comparison_rel_with_avg.append("The " + xLabel + " " + position_in_X_axis_for_min_value + " tends to be " + str(max_avg_diff_rel) + " percent lower than average. ")

            print("Comparison [Relative, vs Avg] : " + comparison_rel_with_avg[random.randint(0, len(comparison_rel_with_avg) - 1)])

            if float(random.uniform(0, 1)) > 0.75:
                summaryArray.append(comparison_rel_with_avg[random.randint(0, len(comparison_rel_with_avg) - 1)])
            else:
                summaryArray.append(derived_val_avg[random.randint(0, len(derived_val_avg) - 1)])

            # Compute derived val [sum]
            sum_text = []
            sum_text.append("The " + yLabel + " is " + str(round(totalValue, 2)) + " if we add up values of all " + xLabel + "s. ")
            sum_text.append("Summing up the values of all " + xLabel + "s, we get total " + str(round(totalValue, 2)) + ". ")

            print("Compute derived val [sum] : " + sum_text[random.randint(0, len(sum_text) - 1)])
            summaryArray.append(sum_text[random.randint(0, len(sum_text) - 1)])

            # Compute derived val [shared value]
            shared_value = []

            res = checkIfDuplicates(yValueArr)
            if res:
                # print('Yes, list contains duplicates')

                most_freq_value = most_frequent(yValueArr)
                most_freq_pos = []
                most_freq_x_label = []
                for i in range(len(yValueArr)):
                    if yValueArr[i] == most_freq_value:
                        most_freq_pos.append(i)
                        most_freq_x_label.append(xValueArr[i])

                shared_value_labels = ""
                for a in range(len(most_freq_x_label)):
                    if a == len(most_freq_x_label) - 1:
                        shared_value_labels += "and " + str(most_freq_x_label[a]).replace('_', ' ')
                    else:
                        shared_value_labels += str(most_freq_x_label[a]).replace('_', ' ') + ", "

                shared_value.append(xLabel + " " + shared_value_labels + " have a similar " + yLabel + " that is " + str(most_freq_value) + ". ")
                shared_value.append(xLabel + " " + shared_value_labels + " share the same value " + str(most_freq_value) + ". ")
                shared_value.append(xLabel + " " + shared_value_labels + " have the same " + yLabel + ". ")
                shared_value.append("Similar " + yLabel + " is found in " + xLabel + " " + shared_value_labels + ". ")

                print("Compute derived val [shared value] : " + shared_value[random.randint(0, len(shared_value) -1)])
                summaryArray.append(shared_value[random.randint(0, len(shared_value) - 1)])


            min_summary = []
            mid_summary = []
            max_summary = []

            min_summary.append(random.choice(intro))
            min_summary.append(random.choice(summary2_extrema_max_min))

            mid_summary.append(random.choice(intro))
            mid_summary.append(random.choice(summary2_extrema_max_min))
            if float(random.uniform(0, 1)) > 0.75:
                mid_summary.append(random.choice(comparison_rel))
            else:
                mid_summary.append(random.choice(comparison_abs))

            max_summary.append(random.choice(intro))
            max_summary.append(random.choice(summary2_extrema_max_min))
            if float(random.uniform(0, 1)) > 0.35 and len(summary3_order_2nd_max) > 0:
                max_summary.append(random.choice(summary3_order_2nd_max))
            if float(random.uniform(0, 1)) > 0.75:
                max_summary.append(random.choice(comparison_rel))
            else:
                max_summary.append(random.choice(comparison_abs))
            if len(summary_order_rank) > 0:
                max_summary.append(random.choice(summary_order_rank))
            if len(global_trend_text) > 0:
                max_summary.append(random.choice(global_trend_text))
            if len(shared_value) > 0:
                max_summary.append(random.choice(shared_value))
            if float(random.uniform(0, 1)) > 0.75:
                max_summary.append(random.choice(derived_val_avg))
            else:
                max_summary.append(random.choice(comparison_rel_with_avg))
            if float(random.uniform(0, 1)) > 0.35:
                max_summary.append(random.choice(sum_text))


            trendsArray = [{}, {"7": maxValueIndex, "12": maxValueIndex},
                           {"7": minValueIndex, "12": minValueIndex}, {}]
            dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
            websiteInput = {"title": title, "xAxis": xLabel, "yAxis": yLabel,
                            "columnType": "two",
                            "graphType": chartType, "summaryType": "baseline", "summary": mid_summary,
                            "min_summary": min_summary,
                            "mid_summary": mid_summary,
                            "max_summary": max_summary,
                            "trends": trendsArray,
                            "data": dataJson}
            with open(f'{websitePath}/{name}.json', 'w', encoding='utf-8') as websiteFile:
                json.dump(websiteInput, websiteFile, indent=3)
            # oneFile.writelines(''.join(summaryArray)+'\n')

        ## for single line charts
        # run line  
        elif (chartType == "line"):
            trendArray = []
            numericXValueArr = []
            for xVal, index in zip(xValueArr, range(
                    len(xValueArr))):  # Every x value is assigned an index from 0 to 11 (e.g. xval1: 0, xval2: 1)
                if xVal.isnumeric():
                    numericXValueArr.append(float(xVal))
                else:
                    # see if regex works better
                    cleanxVal = re.sub("[^\d\.]", "", xVal)
                    if len(cleanxVal) > 0:
                        numericXValueArr.append(float(cleanxVal[:4]))
                    else:
                        numericXValueArr.append(float(index))
            # determine local trends
            graphTrendArray = []
            i = 1
            # calculate variance between each adjacent y values
            # print(xValueArr)
            # print(yValueArr)

            ##For json's smoothing
            while i < (len(yValueArr)):
                variance1 = float(yValueArr[i]) - float(yValueArr[
                                                            i - 1])  # 2nd yVal- Prev yVal # Note that xValueArr and yValueArr are ordered such that the start values are written at the end of the array
                if (variance1 > 0):
                    type1 = "decreasing"  # Drop/ falls/ goes down
                elif (variance1 < 0):
                    type1 = "increasing"  # Rise/ goes up
                else:
                    type1 = "constant"  # Stays the same
                trendArray.append(type1)
                i = i + 1
            #####

            ##Finding the direction of trend -shehnaz

            yVals_float = yValueArr  # yVals_float= stringToFloat(yValueArr)
            yVal = np.array(yVals_float).astype(np.float)  # yVal is now in float type

            # print(xValueArr)
            # print(yVal)

            coordinates = dict(zip(xValueArr, yVal))

            # print(coordinates)

            sorted_coordinates = dict(sorted(coordinates.items()))

            # print(sorted_coordinates)

            keys, values = zip(*sorted_coordinates.items())  # keys, values = zip(sorted_coordinates.items())
            print(keys)
            print(values)

            yValueArrCorrectOrder = np.array(values)  # yValueArr[len(yValueArr)::-1]  ## Ordered correctly this time
            xValueArrCorrectOrder = np.array(keys)  # xValueArr[len(xValueArr)::-1]  ## Ordered correctly this time

            ##### Smooth the entire chart first

            # import numpy as np
            # import numpy as np
            # from scipy.interpolate import make_interp_spline
            # import matplotlib.pyplot as plt

            # # Dataset
            # yValueArrCorrectOrder = yValueArr[len(yValueArr)::-1]  ## Ordered correctly this time
            # xValueArrCorrectOrder = xValueArr[len(xValueArr)::-1]  ## Ordered correctly this time

            # x = np.array(xValueArrCorrectOrder).astype(np.float)
            # y = np.array(yValueArrCorrectOrder).astype(np.float)
            # # plt.plot(x, y)

            # n=2
            # i=0
            # while i < n:
            #     x = np.array(xValueArrCorrectOrder).astype(np.float)
            #     y = np.array(yValueArrCorrectOrder).astype(np.float)

            #     X_Y_Spline = make_interp_spline(x, y)

            #     # Returns evenly spaced numbers
            #     # over a specified interval.
            #     X_ = np.linspace(x.min(), x.max(), 500)
            #     Y_ = X_Y_Spline(X_)

            #     # Plotting the Graph
            #     plt.plot(X_, Y_)
            #     plt.title("Plot Smooth Curve Using the scipy.interpolate.make_interp_spline() Class")
            #     plt.xlabel("X")
            #     plt.ylabel("Y")
            #     plt.show()

            #     # print(x)
            #     # print(y)
            #     # print(X_)
            #     # print(Y_)
            #     yValueArrCorrectOrder= Y_
            #     xValueArrCorrectOrder= X_

            #     i= i+1

            ## End of smoothing

            # yValueArrCorrectOrder = yValueArr[len(yValueArr)::-1]  ## Ordered correctly this time
            # print(yValueArrCorrectOrder)

            ############# GlobalTrend ##############
            globalDifference = float(yValueArrCorrectOrder[len(yValueArrCorrectOrder) - 1]) - float(
                yValueArrCorrectOrder[0])
            globalPercentChange = (globalDifference / float(yValueArr[len(yValueArr) - 1])) * 100

            # print(yValueArr)
            # print(globalDifference)
            # print(globalPercentChange)
            
            
            ## percentChnage divided by max-0
            # n= float(yValueArrCorrectOrder[len(yValueArrCorrectOrder) - 1])
            # o=float(yValueArrCorrectOrder[0])
            # m= max(yValueArrCorrectOrder)
            # globalPercentChange= percentChnageRangeFunc(n, o, m)
            # direction= globalDirectionTrend(globalPercentChange, constant)
            # print(globalPercentChange)
            # print(direction)
            # end
            

            ## percentChnage divided by max-0
            # n= float(yValueArrCorrectOrder[len(yValueArrCorrectOrder) - 1])
            # o=float(yValueArrCorrectOrder[0])
            # m= max(yValueArrCorrectOrder)
            # globalPercentChange= percentChnageRangeFunc(n, o, m)
            # direction= globalDirectionTrend(globalPercentChange, constant)
            # print(globalPercentChange)
            # print(direction)
            # end

            localTrendSentence1 = "This line chart has an x axis representing " + xLabel + " and a y axis representing " + yLabel + ", with a total of " + str(
                len(yValueArrCorrectOrder)) \
                                  + " data points."
            summaryArray.append(localTrendSentence1)

            summary2 = " Overall " + yLabel + " has "

            if globalPercentChange > 0:
                summary2 += "increased"
            elif globalPercentChange < 0:
                summary2 += "decreased"
            else:
                summary2 += "constant"
            
            #summary2 +=direction

            # summary2 +=direction

            summary2 += " over the " + xLabel + "."
            summaryArray.append(summary2)

            ############# LocalTrend ##############
            varianceArray = []

            ### Percentage change appraoch
            percentArray = []
            # directionArray = []
            i = 1
            while i < (len(yValueArrCorrectOrder)):
                old = yValueArrCorrectOrder[i - 1]
                if (old == 0 or old == 0.0):
                    old = 0.00000000001
                variance1 = float(yValueArrCorrectOrder[i]) - float(
                    old)  # 2nd yVal- Prev yVal # Note that xValueArr and yValueArr are ordered such that the start values are written at the end of the array
                localPercentChange = (variance1 / float(old)) * 100

                # o= yValueArrCorrectOrder[i - 1]
                # n= yValueArrCorrectOrder[i]
                # m=max(yValueArrCorrectOrder)
                # variance1 = n-o
                # localPercentChange= percentChnageRangeFunc(n, o, m)
                # d= globalDirectionTrend(localPercentChange, constant)

                varianceArray.append(variance1)
                percentArray.append(localPercentChange)
                # directionArray.append(d)
                i = i + 1
               
            # print(varianceArray)
            print(percentArray)
            # print(directionArray)
                
        
            
            
            # print(varianceArray)
            print(percentArray)
            # print(directionArray)

            # print(varianceArray)
            # print(percentArray)
            
            
            # ##### Slope apporach
            # yValueArrCorrectOrder = yValueArr[len(yValueArr)::-1]  ## Ordered correctly this time
            # # print(yValueArrCorrectOrder)

            # xValueArrCorrectOrder = xValueArr[len(xValueArr)::-1]  ## Ordered correctly this time
            # # print(xValueArrCorrectOrder)
            
            # slopeArray = []
            # i = 0
            # while i < (len(yValueArrCorrectOrder)-1):
            #     neumerator= yValueArrCorrectOrder[i+1]- yValueArrCorrectOrder[i]
            #     denominator= (i+1)- (i)
            #     slope= neumerator/denominator
            #     slopeArray.append(slope)
            #     i = i + 1
            
            # print(slopeArray)

            
            
            ##Start here # Plan: Normalize percentages and then use them to determine direction and trends everywhere and check if it works.
            
            ## Normalization of percentArray for smoothing
            # normalized_percentArray= []
            # mean_percentArray= mean(percentArray)
            # sd_percentArray = stdev(percentArray)
            # for i in range(0, len(percentArray)):
            #     normalized_val= (percentArray[i]- mean_percentArray)/sd_percentArray
            #     normalized_percentArray.append(normalized_val)
            # print("normalized_percentArray")
            
            
            ##### percentArraynormalization apporach
            # normalized_percentArray= []
            # min_val= min(percentArray)
            # max_val= max(percentArray)
            # # min_val= 0
            # # max_val= 100
            
            # for i in range(0, len(percentArray)):
            #     normalized_val= (100*(percentArray[i]- min_val))/(max_val-min_val)
            #     normalized_percentArray.append(normalized_val)
            # print("normalized_percentArray")
            
            
            # normalized_percentArrayCorrectOrder= normalized_percentArray[len(normalized_percentArray)::-1]
            # print(normalized_percentArrayCorrectOrder)
            
            # abs_slopeArray= [abs(number) for number in slopeArray] #neww
            # print(abs_slopeArray)
            
            # normalized_slopeArray= []
            
            # # minValSlope= min(abs_slopeArray)
            # # maxValSlope= max(abs_slopeArray)
            
            # meanSlope= mean(abs_slopeArray)
            # sdSlope= stdev(abs_slopeArray)
            
            # # for i in range(0, len(abs_slopeArray)):
            # #     normalized_slope= (100*(abs_slopeArray[i]- minValSlope))/(maxValSlope-minValSlope)
            # #     normalized_slopeArray.append(normalized_slope)
            # # print("normalized_slopeArray")
            
            # for i in range(0, len(abs_slopeArray)):
            #     normalized_slope= (abs_slopeArray[i]- meanSlope)/sdSlope
            #     normalized_slopeArray.append(normalized_slope)
            # print("normalized_slopeArray")

            # print(normalized_slopeArray)
            

            varianceArrayCorrectOrder =varianceArray  #varianceArray[len(varianceArray)::-1]  ## Ordered correctly this time
            percentArrayCorrectOrder = percentArray #percentArray[len(percentArray)::-1]  ## Ordered correctly this time
            

            # ##### Slope apporach
            # yValueArrCorrectOrder = yValueArr[len(yValueArr)::-1]  ## Ordered correctly this time
            # # print(yValueArrCorrectOrder)

            # xValueArrCorrectOrder = xValueArr[len(xValueArr)::-1]  ## Ordered correctly this time
            # # print(xValueArrCorrectOrder)

            # slopeArray = []
            # i = 0
            # while i < (len(yValueArrCorrectOrder)-1):
            #     neumerator= yValueArrCorrectOrder[i+1]- yValueArrCorrectOrder[i]
            #     denominator= (i+1)- (i)
            #     slope= neumerator/denominator
            #     slopeArray.append(slope)
            #     i = i + 1

            # print(slopeArray)

            ##Start here # Plan: Normalize percentages and then use them to determine direction and trends everywhere and check if it works.

            ## Normalization of percentArray for smoothing
            # normalized_percentArray= []
            # mean_percentArray= mean(percentArray)
            # sd_percentArray = stdev(percentArray)
            # for i in range(0, len(percentArray)):
            #     normalized_val= (percentArray[i]- mean_percentArray)/sd_percentArray
            #     normalized_percentArray.append(normalized_val)
            # print("normalized_percentArray")

            ##### percentArraynormalization apporach
            # normalized_percentArray= []
            # min_val= min(percentArray)
            # max_val= max(percentArray)
            # # min_val= 0
            # # max_val= 100

            # for i in range(0, len(percentArray)):
            #     normalized_val= (100*(percentArray[i]- min_val))/(max_val-min_val)
            #     normalized_percentArray.append(normalized_val)
            # print("normalized_percentArray")

            # normalized_percentArrayCorrectOrder= normalized_percentArray[len(normalized_percentArray)::-1]
            # print(normalized_percentArrayCorrectOrder)

            # abs_slopeArray= [abs(number) for number in slopeArray] #neww
            # print(abs_slopeArray)

            # normalized_slopeArray= []

            # # minValSlope= min(abs_slopeArray)
            # # maxValSlope= max(abs_slopeArray)

            # meanSlope= mean(abs_slopeArray)
            # sdSlope= stdev(abs_slopeArray)

            # # for i in range(0, len(abs_slopeArray)):
            # #     normalized_slope= (100*(abs_slopeArray[i]- minValSlope))/(maxValSlope-minValSlope)
            # #     normalized_slopeArray.append(normalized_slope)
            # # print("normalized_slopeArray")

            # for i in range(0, len(abs_slopeArray)):
            #     normalized_slope= (abs_slopeArray[i]- meanSlope)/sdSlope
            #     normalized_slopeArray.append(normalized_slope)
            # print("normalized_slopeArray")

            # print(normalized_slopeArray)

            varianceArrayCorrectOrder = varianceArray  # varianceArray[len(varianceArray)::-1]  ## Ordered correctly this time
            percentArrayCorrectOrder = percentArray  # percentArray[len(percentArray)::-1]  ## Ordered correctly this time

            # print(varianceArrayCorrectOrder)
            # print(percentArrayCorrectOrder) #neww

            ## percentArray Appraoch
            ## Mean of abs_percentArrayCorrectOrder
            abs_percentArrayCorrectOrder = [abs(number) for number in percentArrayCorrectOrder]  # neww
            # print(abs_percentArrayCorrectOrder)
            mean_percentArray = mean(abs_percentArrayCorrectOrder)  # mean of abosulte values of percentArray
            print(mean_percentArray)
            # sd_percentArray= stdev(abs_percentArrayCorrectOrder)
            # print(sd_percentArray)

            constant_rate = c_rate * mean_percentArray  # avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
            significant_rate = s_rate * mean_percentArray
            gradually_rate = g_rate * mean_percentArray
            rapidly_rate = r_rate * mean_percentArray

            # significant_rate=sig

            # constant_rate = mean_percentArray- 1*(sd_percentArray) # avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
            # significant_rate = mean_percentArray# avg(% chnage)*0.1 # Meaning any chnage >constant rate and less than this rate is considered not significant and so it's trend direction is chnaged to the trend of the succesive interval # Determines the start and end of the trend
            # gradually_rate= mean_percentArray+ 1*(sd_percentArray)
            # rapidly_rate= mean_percentArray+ 2*(sd_percentArray)

            # ## Standard Deviation of abs_percentArrayCorrectOrder
            # sd_percentArray= stdev(abs_percentArrayCorrectOrder)  # standard deviation of abosulte values of percentArray
            # print(sd_percentArray)

            directionArray = []
            i = 1
            while i < (len(yValueArrCorrectOrder)):
                d = directionTrend(yValueArrCorrectOrder[i],
                                   yValueArrCorrectOrder[i - 1],
                                   constant_rate)  # direction e.g. increase, decrease or constant
                directionArray.append(d)
                i = i + 1
            print("Orginal Direction Trend:")
            print(directionArray)

            ### Previously indexs reported for only increasing and decresing trends
            # trendChangeIdx = []
            # for idx in range(0, len(varianceArrayCorrectOrder) - 1):

            #     # checking for successive opposite index
            #     if varianceArrayCorrectOrder[idx] > 0 and varianceArrayCorrectOrder[idx + 1] < 0 or varianceArrayCorrectOrder[idx] < 0 and varianceArrayCorrectOrder[idx + 1] > 0:
            #         trendChangeIdx.append(idx)

            # print("Sign shift indices : " + str(trendChangeIdx))

            # percentArray approach to smoothing
            ## Smoothing directionArray. If percentChange >10% then direction of trend is that of the next interval (regardless if it was increasing or decreasing)
            directionArraySmoothed = []
            for idx in range(0, len(percentArrayCorrectOrder) - 1):  # neww
                # checking for percent chnage >5% (not constant) and <10% (not significant) and chnaging their direction to be the direction of the succesive interval
                if (abs(percentArrayCorrectOrder[idx]) > constant_rate and abs(
                        percentArrayCorrectOrder[idx]) < significant_rate):  # neww
                    d = directionArray[idx + 1]
                    directionArraySmoothed.append(d)
                else:
                    directionArraySmoothed.append(directionArray[idx])
            directionArraySmoothed.append(directionArray[len(
                percentArrayCorrectOrder) - 1])  # neww # The last value doesn't have a succesive interval so it will be appended as is
            print("Smoothed Direction Trend:")
            print(directionArraySmoothed)
            
            
            
            # constant_rate = meanSlope- 1*(sdSlope)
            # significant_rate = meanSlope 
            # gradually_rate= meanSlope+ 1*(sdSlope)
            # rapidly_rate= meanSlope + 2*(sdSlope)
            
            
            #slopeArray approach to smoothing
            ## Smoothing directionArray. If percentChange >10% then direction of trend is that of the next interval (regardless if it was increasing or decreasing)
            # directionArraySmoothed = []
            # for idx in range(0, len(normalized_slopeArray) - 1): #neww
            #     # checking for percent chnage >5% (not constant) and <10% (not significant) and chnaging their direction to be the direction of the succesive interval
            #     if (abs(normalized_slopeArray[idx]) > constant_rate and abs(normalized_slopeArray[idx]) < significant_rate): #neww
            #         d = directionArray[idx + 1]
            #         directionArraySmoothed.append(d)
            #     else:
            #         directionArraySmoothed.append(directionArray[idx])
            # directionArraySmoothed.append(directionArray[len(
            #     normalized_slopeArray) - 1])  #neww # The last value doesn't have a succesive interval so it will be appended as is
            # print("Smoothed Direction Trend:")
            # print(directionArraySmoothed)

            # constant_rate = meanSlope- 1*(sdSlope)
            # significant_rate = meanSlope
            # gradually_rate= meanSlope+ 1*(sdSlope)
            # rapidly_rate= meanSlope + 2*(sdSlope)

            # slopeArray approach to smoothing
            ## Smoothing directionArray. If percentChange >10% then direction of trend is that of the next interval (regardless if it was increasing or decreasing)
            # directionArraySmoothed = []
            # for idx in range(0, len(normalized_slopeArray) - 1): #neww
            #     # checking for percent chnage >5% (not constant) and <10% (not significant) and chnaging their direction to be the direction of the succesive interval
            #     if (abs(normalized_slopeArray[idx]) > constant_rate and abs(normalized_slopeArray[idx]) < significant_rate): #neww
            #         d = directionArray[idx + 1]
            #         directionArraySmoothed.append(d)
            #     else:
            #         directionArraySmoothed.append(directionArray[idx])
            # directionArraySmoothed.append(directionArray[len(
            #     normalized_slopeArray) - 1])  #neww # The last value doesn't have a succesive interval so it will be appended as is
            # print("Smoothed Direction Trend:")
            # print(directionArraySmoothed)

            trendChangeIdx = []
            for idx in range(0, len(directionArraySmoothed) - 1):

                # checking for successive opposite index
                if directionArraySmoothed[idx] != directionArraySmoothed[idx + 1]:
                    trendChangeIdx.append(idx)

            print("Sign shift indices : " + str(trendChangeIdx))

            # yValueArrCorrectOrder = yValueArr[len(yValueArr)::-1]  ## Ordered correctly this time
            # print(yValueArrCorrectOrder)

            # xValueArrCorrectOrder = xValueArr[len(xValueArr)::-1]  ## Ordered correctly this time
            # print(xValueArrCorrectOrder)

            # trendArrayCorrectOrder = trendArray[len(trendArray)::-1] # no need since have my own directionArray now ordered correctly
            # print(trendArrayCorrectOrder)

            # print(trendChangeIdx)

            # Slope Approach
            ## Find the new slopes for the trendChangeIdx points
            # refinedSlope_array= []
            refinedPercentChnage_array = []
            x = 0
            # max_y= max(yValueArrCorrectOrder)

            if trendChangeIdx:  # if trendChangeIdx is not empty
                for i in trendChangeIdx:
                    if (x == 0):
                        # neumerator= yValueArrCorrectOrder[i+1]- yValueArrCorrectOrder[0]
                        # denominator= (i+1)- (0)
                        # slope= neumerator/denominator
                        # refinedSlope_array.append(slope)
                        new = yValueArrCorrectOrder[i + 1]
                        old = yValueArrCorrectOrder[0]

                        # percentChange= ((new-old)/old)*100
                        percentChange = percentChnageFunc(new, old)  # to account for error: float division by zero
                        refinedPercentChnage_array.append(percentChange)

                        # localPercentChange= percentChnageRangeFunc(new, old, max_y)
                        # refinedPercentChnage_array.append(localPercentChange)


                    elif (x > 0 or x < len(trendChangeIdx) - 1):
                        # neumerator= yValueArrCorrectOrder[i+1]-  yValueArrCorrectOrder[trendChangeIdx[x - 1] + 1]
                        # denominator= (i+1)- (trendChangeIdx[x - 1] + 1)
                        # slope= neumerator/denominator
                        # refinedSlope_array.append(slope)

                        new = yValueArrCorrectOrder[i + 1]
                        old = yValueArrCorrectOrder[trendChangeIdx[x - 1] + 1]
                        # percentChange= ((new-old)/old)*100
                        percentChange = percentChnageFunc(new, old)  # to account for error: float division by zero
                        refinedPercentChnage_array.append(percentChange)

                        # localPercentChange= percentChnageRangeFunc(new, old, max_y)
                        # refinedPercentChnage_array.append(localPercentChange)

                    x = x + 1

                # neumerator= yValueArrCorrectOrder[-1]-  yValueArrCorrectOrder[trendChangeIdx[-1] + 1]
                # denominator= (x)- (trendChangeIdx[-1] + 1)
                # slope= neumerator/denominator
                # refinedSlope_array.append(slope)

                new = yValueArrCorrectOrder[-1]
                old = yValueArrCorrectOrder[trendChangeIdx[-1] + 1]
                # percentChange= ((new-old)/old)*100
                percentChange = percentChnageFunc(new, old)  # to account for error: float division by zero
                refinedPercentChnage_array.append(percentChange)

                # localPercentChange= percentChnageRangeFunc(new, old, max_y)
                # refinedPercentChnage_array.append(localPercentChange)


            else:
                # neumerator= yValueArrCorrectOrder[-1]- yValueArrCorrectOrder[0]
                # denominator= (len(yValueArrCorrectOrder)-1)- 0
                # slope= neumerator/denominator
                # refinedSlope_array.append(slope)

                new = yValueArrCorrectOrder[-1]
                old = yValueArrCorrectOrder[0]
                # percentChange= ((new-old)/old)*100
                percentChange = percentChnageFunc(new, old)  # to account for error: float division by zero
                refinedPercentChnage_array.append(percentChange)

                # localPercentChange= percentChnageRangeFunc(new, old, max_y)
                # refinedPercentChnage_array.append(localPercentChange)

            # print("Refined Slope")
            # print(refinedSlope_array)

            print("Refined Percent Change")
            print(refinedPercentChnage_array)

            # Mean of abs_refinedPercentChnage_array
            abs_refinedPercentChnage_array = [abs(number) for number in refinedPercentChnage_array]  # neww
            # print(abs_percentArrayCorrectOrder)
            mean_abs_refinedPercentChnage = mean(
                abs_refinedPercentChnage_array)  # mean of abosulte values of percentArray
            print(mean_abs_refinedPercentChnage)
            # sd_abs_refinedPercentChnage= stdev(abs_refinedPercentChnage_array)
            # print(sd_abs_refinedPercentChnage)

            constant_rate = c_rate * mean_abs_refinedPercentChnage  # avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
            significant_rate = s_rate * mean_abs_refinedPercentChnage
            gradually_rate = g_rate * mean_abs_refinedPercentChnage
            rapidly_rate = r_rate * mean_abs_refinedPercentChnage

            # constant_rate = mean_abs_refinedPercentChnage- 1*(sd_abs_refinedPercentChnage) # avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
            # significant_rate = mean_abs_refinedPercentChnage# avg(% chnage)*0.1 # Meaning any chnage >constant rate and less than this rate is considered not significant and so it's trend direction is chnaged to the trend of the succesive interval # Determines the start and end of the trend
            # gradually_rate= mean_abs_refinedPercentChnage+ 1*(sd_abs_refinedPercentChnage)
            # rapidly_rate= mean_abs_refinedPercentChnage+ 2*(sd_abs_refinedPercentChnage)

            # Trying out the percentage using max-0 range of charts instead of dividing by old
            # constant_rate = constant
            # gradually_rate= gradual
            # rapidly_rate= rapid

            print(constant_rate)
            print(significant_rate)
            print(gradually_rate)
            print(rapidly_rate)

            ## Normalize refined Slope
            # abs_refinedSlope_array= [abs(number) for number in refinedSlope_array] #neww
            # print(abs_refinedSlope_array)

            # normalized_refinedSlope_array= []

            # minValRefinedSlope= min(abs_refinedSlope_array)
            # maxValRefinedSlope= max(abs_refinedSlope_array)

            # for i in range(0, len(abs_refinedSlope_array)):
            #     normalized_slope= (100*(abs_refinedSlope_array[i]- minValRefinedSlope))/(maxValRefinedSlope-minValRefinedSlope)
            #     normalized_refinedSlope_array.append(normalized_slope)
            # print("normalized_refinedSlopeArray")

            # meanRefinedSlope= mean(abs_refinedSlope_array)
            # sdRefinedSlope= stdev(abs_refinedSlope_array)

            # for i in range(0, len(abs_refinedSlope_array)):
            #     normalized_slope= (abs_refinedSlope_array[i]- meanRefinedSlope)/sdRefinedSlope
            #     normalized_refinedSlope_array.append(normalized_slope)
            # print("normalized_refinedSlopeArray")

            # print(normalized_refinedSlope_array)

            # constant_rate = meanRefinedSlope- 1*(sdRefinedSlope)
            # significant_rate = meanRefinedSlope
            # gradually_rate= meanRefinedSlope+ 1*(sdRefinedSlope)
            # rapidly_rate= meanRefinedSlope + 2*(sdRefinedSlope)
            # print(constant_rate)
            # print(significant_rate)
            # print(gradually_rate)
            # print(rapidly_rate)

            ##### Print the summary

            summary3 = yLabel
            rateOfchange_array = []
            # rateOfchange_num_array= []
            x = 0
            if trendChangeIdx:  # if trendChangeIdx is not empty
                for i in trendChangeIdx:
                    if (x == 0):
                        # rateOfChange_num= rateOfChnageVal(yValueArrCorrectOrder[i + 1],yValueArrCorrectOrder[0], directionArraySmoothed[i], (i + 1), 0, max_val, min_val)
                        # rateOfchange_num_array.append(rateOfChange_num)

                        rateOfChange = rateOfChnage(refinedPercentChnage_array[x], directionArraySmoothed[i],
                                                    constant_rate, gradually_rate, rapidly_rate)
                        rateOfchange_array.append(rateOfChange)

                        summary3 += " is " + rateOfChange + " " + directionArraySmoothed[
                            i] + " from " + str(xValueArrCorrectOrder[0]) + " to " + str(xValueArrCorrectOrder[
                                                                                             i + 1]) + ", "
                    elif (x > 0 or x < len(trendChangeIdx) - 1):
                        # rateOfChange_num= rateOfChange(yValueArrCorrectOrder[i + 1], yValueArrCorrectOrder[trendChangeIdx[x - 1] + 1], directionArraySmoothed[i], (i + 1), (trendChangeIdx[x - 1] + 1), max_val, min_val)
                        # rateOfchange_num_array.append(rateOfChange_num)

                        rateOfChange = rateOfChnage(refinedPercentChnage_array[x], directionArraySmoothed[i],
                                                    constant_rate, gradually_rate, rapidly_rate)
                        rateOfchange_array.append(rateOfChange)

                        summary3 += rateOfChange + " " + \
                                    directionArraySmoothed[i] + " from " + str(xValueArrCorrectOrder[
                                                                                   trendChangeIdx[
                                                                                       x - 1] + 1]) + " to " + str(
                            xValueArrCorrectOrder[i + 1]) + ", "

                    x = x + 1

                # rateOfChange_num= rateOfChnageVal(yValueArrCorrectOrder[-1], yValueArrCorrectOrder[trendChangeIdx[-1] + 1], directionArraySmoothed[-1], (-1), (trendChangeIdx[-1] + 1), max_val, min_val)
                # rateOfchange_num_array.append(rateOfChange_num)

                rateOfChange = rateOfChnage(refinedPercentChnage_array[x], directionArraySmoothed[-1], constant_rate,
                                            gradually_rate, rapidly_rate)
                rateOfchange_array.append(rateOfChange)

                summary3 += "and lastly " + rateOfChange + " " + \
                            directionArraySmoothed[-1] + " from " + str(xValueArrCorrectOrder[
                                                                            trendChangeIdx[-1] + 1]) + " to " + str(
                    xValueArrCorrectOrder[-1]) + "."
            else:
                # rateOfChange_num= rateOfChnageVal(yValueArrCorrectOrder[-1], yValueArrCorrectOrder[0], directionArraySmoothed[-1], (-1), (0), max_val, min_val)
                # rateOfchange_num_array.append(rateOfChange_num)

                rateOfChange = rateOfChnage(refinedPercentChnage_array[x], directionArraySmoothed[-1], constant_rate,
                                            gradually_rate, rapidly_rate)
                rateOfchange_array.append(rateOfChange)

                summary3 += " is " + rateOfChange + " " + \
                            directionArraySmoothed[-1] + " from " + str(xValueArrCorrectOrder[0]) + " to " + \
                            str(xValueArrCorrectOrder[-1]) + "."

            if (len(trendChangeIdx) < 5):
                summaryArray.append(summary3)

            print(rateOfchange_array)
            # print(rateOfchange_num_array)

            ############# Steepest Slope ##############

            # Absolute value of varianceArrayCorrectOrder elements
            absoluteVariance = [abs(ele) for ele in varianceArrayCorrectOrder]

            max_value = max(absoluteVariance)
            max_index = absoluteVariance.index(max_value)

            # print(absoluteVariance)
            # print(max_value)
            # print(max_index)
            # print(directionArraySmoothed)

            if increaseDecrease(directionArraySmoothed[max_index]) != "stays the same":
                summary4 = "The steepest " + increaseDecrease(
                    directionArraySmoothed[max_index]) + " occurs in between the " + xLabel + " " + str(
                    xValueArrCorrectOrder[
                        max_index]) + " and " + str(xValueArrCorrectOrder[max_index + 1]) + "."
                summaryArray.append(summary4)

            ############# Extrema Max ##############
            # print(yValueArrCorrectOrder)

            max_index = get_indexes_max_value(yValueArrCorrectOrder)
            # print(max_index)
            # print(len(max_index))

            summary5 = "Max " + yLabel + " about " + str(
                yValueArrCorrectOrder[max_index[0]]) + " was recorded at " + xLabel

            if len(max_index) > 1:
                i = 0
                while i < (len(max_index) - 1):
                    summary5 += " " + str(xValueArrCorrectOrder[max_index[i]]) + ", "
                    i = i + 1
                summary5 += "and " + str(xValueArrCorrectOrder[max_index[-1]])
            else:
                summary5 += " " + str(xValueArrCorrectOrder[max_index[0]])

            summaryArray.append(summary5)

            ############# Extrema Min ##############
            # print(yValueArrCorrectOrder)

            min_index = get_indexes_min_value(yValueArrCorrectOrder)
            # print(min_index)
            # print(len(min_index))

            summary5 = "Min " + yLabel + " about " + str(
                yValueArrCorrectOrder[min_index[0]]) + " was recorded at " + xLabel

            if len(min_index) > 1:
                i = 0
                while i < (len(min_index) - 1):
                    summary5 += " " + str(xValueArrCorrectOrder[min_index[i]]) + ", "
                    i = i + 1
                summary5 += "and " + str(xValueArrCorrectOrder[min_index[-1]])
            else:
                summary5 += " " + str(xValueArrCorrectOrder[min_index[0]])

            summaryArray.append(summary5)

            newLine = "#########################################################################"
            summaryArray.append(newLine)

            ##########################################################
            # iterate through the variances and check for trends
            startIndex = 0
            trendLen = len(trendArray)
            # creates dictionary containing the trend length, direction, start and end indices, and the linear regression of the trend
            significanceRange = round(len(yValueArr) / 8)  ## Why divide by 8??
            significantTrendCount = 0
            significantTrendArray = []
            for n in range(trendLen):  # 0 to 10 (exclusive)
                currentVal = trendArray[
                    n - 1]  ## Traversing through the array backwards (from end to start == start to end for the chart)
                nextVal = trendArray[n]
                if (currentVal != nextVal or (currentVal == nextVal and n == (trendLen - 1))):
                    if (n == (trendLen - 1)):
                        endIndex = n + 1
                    else:
                        endIndex = n
                    trendLength = endIndex - startIndex + 1
                    if trendLength > significanceRange:
                        xRange = pd.Series(numericXValueArr).loc[startIndex:endIndex]
                        yRange = pd.Series(yValueArr).loc[startIndex:endIndex]
                        result = linregress(xRange, yRange)
                        intercept = round(result[1], 2)
                        slope = round(result[0], 2)
                        trendRange = {"Length": (endIndex - startIndex + 1), "direction": currentVal,
                                      "start": startIndex, "end": endIndex, "slope": slope, "intercept": intercept}
                        significantTrendArray.append(trendRange)
                        significantTrendCount += 1
                        startIndex = n
            # sort the trend dictionaries by length
            if (significantTrendCount > 1):
                # normalize trend slopes to get magnitudes for multi-trend charts
                slopes = np.array([trend['slope'] for trend in significantTrendArray]).reshape(-1, 1)
                scaler = preprocessing.MinMaxScaler()
                scaler.fit(slopes)
                scaledSlopes = scaler.transform(slopes)
                # print(significantTrendArray)
                for trend, normalizedSlope in zip(significantTrendArray, scaledSlopes):
                    trend['magnitude'] = getMagnitude(normalizedSlope[0])
                # print(significantTrendArray)

            sortedTrends = sorted(significantTrendArray, key=lambda i: i['Length'], reverse=True)
            # generate the textual summary from the significant trend dictionary array at m
            if (significantTrendCount > 0):
                startVal = str(xValueArr[(sortedTrends[0]['start'])])
                endVal = str(xValueArr[(sortedTrends[0]['end'])])
                direction = str(sortedTrends[0]['direction'])
                if (significantTrendCount > 1):
                    magnitude = str(sortedTrends[0]['magnitude'])
                    m = 1
                    # execute here if more than 1 significant trend
                    similarSynonyms = ["Similarly", "Correspondingly", "Likewise", "Additionally", "Also",
                                       "In a similar manner"]
                    contrarySynonyms = ["Contrarily", "Differently", "On the other hand", "Conversely",
                                        "On the contrary",
                                        "But"]
                    extraTrends = ""
                    localTrendSentence1 = "This line chart has an x axis representing " + xLabel + " and a y axis representing " + yLabel + ", with a total of " + str(
                        len(yValueArr)) \
                                          + " data points. The chart has " + str(
                        significantTrendCount) + " significant trends."
                    summaryArray.append(localTrendSentence1)
                    graphTrendArray.append({})
                    localTrendSummary = " The longest trend is " + magnitude + " " + direction + " which exists from " + endVal + " to " + startVal + "."
                    summaryArray.append(localTrendSummary)
                    graphTrendArray.append({"1": str(xValueArr.index(startVal)), "12": str(xValueArr.index(endVal))})

                    while (m < significantTrendCount):
                        # append conjunction between significant trends
                        if (direction == "increasing"):
                            length = len(similarSynonyms)
                            random_lbl = randint(0, length - 1)
                            synonym = similarSynonyms[random_lbl]
                            conjunction = synonym + ","
                        elif (
                                direction == "decreasing" or direction == "constant"):  # new #chnaged due to error of 'conjunction' referenced before assignmnet
                            length = len(contrarySynonyms)
                            random_lbl = randint(0, length - 1)
                            synonym = contrarySynonyms[random_lbl]
                            conjunction = synonym + ","
                        startVal = str(xValueArr[(sortedTrends[m]['start'])])
                        endVal = str(xValueArr[(sortedTrends[m]['end'])])
                        direction = str(sortedTrends[m]['direction'])
                        magnitude = str(sortedTrends[m]['magnitude'])
                        extraTrends = " " + conjunction + " the next significant trend is " + magnitude + " " + direction + " which exists from " + endVal + " to " + startVal + "."
                        summaryArray.append(extraTrends)
                        graphTrendArray.append(
                            {"3": str(xValueArr.index(startVal)), "14": str(xValueArr.index(endVal))})
                        m = m + 1
                # execute here if only 1 significant trend
                else:
                    localTrendSentence1 = "This line chart has an x axis representing " + xLabel + " and a y axis representing " + yLabel + " with a total of " + str(
                        len(yValueArr)) + " data points. The chart has one significant trend."
                    summaryArray.append(localTrendSentence1)
                    graphTrendArray.append({})
                    localTrendSummary = " This trend is " + direction + " which exists from " + startVal + " to " + endVal + "."
                    summaryArray.append(localTrendSummary)
                    graphTrendArray.append({"3": str(xValueArr.index(startVal)), "14": str(xValueArr.index(endVal))})
            dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
            websiteInput = {"title": title, "xAxis": xLabel, "yAxis": yLabel,
                            "columnType": "two",
                            "graphType": chartType, "summaryType": "baseline", "summary": summaryArray,
                            "trends": graphTrendArray,
                            "data": dataJson}
            with open(f'{websitePath}/{name}.json', 'w', encoding='utf-8') as websiteFile:
                json.dump(websiteInput, websiteFile, indent=3)
            # oneFile.writelines(''.join(summaryArray) + '\n')

    print(summaryArray)
    
   
    
    
    
    



# bar_data = "Hashtag|#forthepeople|x|bar_chart Total_number_of_mentions|6961|y|bar_chart Hashtag|#trumpshutdown|x|bar_chart Total_number_of_mentions|3992|y|bar_chart Hashtag|#protectourcare|x|bar_chart Total_number_of_mentions|2810|y|bar_chart Hashtag|#endgunviolence|x|bar_chart Total_number_of_mentions|2342|y|bar_chart Hashtag|#hr8|x|bar_chart Total_number_of_mentions|2206|y|bar_chart Hashtag|#actonclimate|x|bar_chart Total_number_of_mentions|2206|y|bar_chart Hashtag|#endtheshutdown|x|bar_chart Total_number_of_mentions|1769|y|bar_chart Hashtag|#sotu|x|bar_chart Total_number_of_mentions|1767|y|bar_chart Hashtag|#equalityact|x|bar_chart Total_number_of_mentions|1720|y|bar_chart Hashtag|#hr1|x|bar_chart Total_number_of_mentions|1683|y|bar_chart "
bar_data = "Month|Dec_19|x|bar_chart Market_share|5.9|y|bar_chart Month|Nov_19|x|bar_chart Market_share|6.2|y|bar_chart Month|Oct_19|x|bar_chart Market_share|6.4|y|bar_chart Month|Sep_19|x|bar_chart Market_share|6.9|y|bar_chart Month|Aug_19|x|bar_chart Market_share|5.7|y|bar_chart Month|Jul_19|x|bar_chart Market_share|5.6|y|bar_chart Month|Jun_19|x|bar_chart Market_share|4.9|y|bar_chart Month|May_19|x|bar_chart Market_share|5.3|y|bar_chart Month|Apr_19|x|bar_chart Market_share|5.5|y|bar_chart Month|Mar_19|x|bar_chart Market_share|4.9|y|bar_chart Month|Feb_19|x|bar_chart Market_share|5.3|y|bar_chart Month|Jan_19|x|bar_chart Market_share|5.3|y|bar_chart Month|Dec_18|x|bar_chart Market_share|6.6|y|bar_chart Month|Nov_18|x|bar_chart Market_share|6.4|y|bar_chart Month|Oct_18|x|bar_chart Market_share|6.8|y|bar_chart Month|Sep_18|x|bar_chart Market_share|7.2|y|bar_chart Month|Aug_18|x|bar_chart Market_share|4.6|y|bar_chart "
# group_bar_data = "Frequency|At_least_once_a_day|0|bar_chart White|64|1|bar_chart Hispanic|56|2|bar_chart Black|53|3|bar_chart Other|49|4|bar_chart Frequency|A_few_times_a_week|0|bar_chart White|12|1|bar_chart Hispanic|15|2|bar_chart Black|15|3|bar_chart Other|14|4|bar_chart Frequency|At_least_once_a_week|0|bar_chart White|4|1|bar_chart Hispanic|4|2|bar_chart Black|7|3|bar_chart Other|3|4|bar_chart Frequency|A_few_times_a_month|0|bar_chart White|3|1|bar_chart Hispanic|6|2|bar_chart Black|3|3|bar_chart Other|7|4|bar_chart Frequency|At_least_once_a_month|0|bar_chart White|4|1|bar_chart Hispanic|5|2|bar_chart Black|4|3|bar_chart Other|6|4|bar_chart Frequency|Never|0|bar_chart White|11|1|bar_chart Hispanic|13|2|bar_chart Black|13|3|bar_chart Other|19|4|bar_chart Frequency|Don't_know/No_opinion|0|bar_chart White|2|1|bar_chart Hispanic|1|2|bar_chart Black|5|3|bar_chart Other|2|4|bar_chart "

# group_bar_data = "Year|2006|0|bar_chart Practitioners|750|1|bar_chart Retail_Pharmacies|1350|2|bar_chart Chain_Pharmacies|1340|3|bar_chart Year|2007|0|bar_chart Practitioners|850|1|bar_chart Retail_Pharmacies|1360|2|bar_chart Chain_Pharmacies|1350|3|bar_chart Year|2008|0|bar_chart Practitioners|650|1|bar_chart Retail_Pharmacies|1380|2|bar_chart Chain_Pharmacies|1370|3|bar_chart Year|2009|0|bar_chart Practitioners|700|1|bar_chart Retail_Pharmacies|1400|2|bar_chart Chain_Pharmacies|1390|3|bar_chart Year|2010|0|bar_chart Practitioners|820|1|bar_chart Retail_Pharmacies|1420|2|bar_chart Chain_Pharmacies|1410|3|bar_chart Year|2011|0|bar_chart Practitioners|1200|1|bar_chart Retail_Pharmacies|1310|2|bar_chart Chain_Pharmacies|1300|3|bar_chart "
group_bar_data = "Month|Dec|0|bar_chart 2015|2.57|1|bar_chart 2016|2.64|2|bar_chart 2017|2.69|3|bar_chart 2018|2.56|4|bar_chart 2019|2.41|5|bar_chart Month|Nov|0|bar_chart 2015|2.76|1|bar_chart 2016|2.68|2|bar_chart 2017|2.66|3|bar_chart 2018|2.55|4|bar_chart 2019|2.45|5|bar_chart Month|Oct|0|bar_chart 2015|2.77|1|bar_chart 2016|2.72|2|bar_chart 2017|2.68|3|bar_chart 2018|2.67|4|bar_chart 2019|2.44|5|bar_chart Month|Sep|0|bar_chart 2015|2.83|1|bar_chart 2016|2.8|2|bar_chart 2017|2.74|3|bar_chart 2018|2.61|4|bar_chart 2019|2.52|5|bar_chart Month|Aug|0|bar_chart 2015|2.81|1|bar_chart 2016|2.88|2|bar_chart 2017|2.76|3|bar_chart 2018|2.7|4|bar_chart 2019|2.49|5|bar_chart Month|Jul|0|bar_chart 2015|2.81|1|bar_chart 2016|2.84|2|bar_chart 2017|2.79|3|bar_chart 2018|2.67|4|bar_chart 2019|2.5|5|bar_chart Month|Jun|0|bar_chart 2015|2.79|1|bar_chart 2016|2.75|2|bar_chart 2017|2.77|3|bar_chart 2018|2.58|4|bar_chart 2019|2.5|5|bar_chart Month|May|0|bar_chart 2015|2.81|1|bar_chart 2016|2.76|2|bar_chart 2017|2.94|3|bar_chart 2018|2.71|4|bar_chart 2019|2.56|5|bar_chart Month|Apr|0|bar_chart 2015|2.78|1|bar_chart 2016|2.78|2|bar_chart 2017|2.78|3|bar_chart 2018|2.71|4|bar_chart 2019|2.53|5|bar_chart Month|Mar|0|bar_chart 2015|2.81|1|bar_chart 2016|2.69|2|bar_chart 2017|2.83|3|bar_chart 2018|2.58|4|bar_chart 2019|2.59|5|bar_chart Month|Feb|0|bar_chart 2015|2.81|1|bar_chart 2016|2.73|2|bar_chart 2017|2.75|3|bar_chart 2018|2.69|4|bar_chart 2019|2.48|5|bar_chart Month|Jan|0|bar_chart 2015|2.88|1|bar_chart 2016|2.77|2|bar_chart 2017|2.82|3|bar_chart 2018|2.7|4|bar_chart 2019|2.61|5|bar_chart "
grp_bar_y_label = "Average retail price in CAD"
grp_bar_title = "Average retail price for white sugar in Canada 2015 to 2019"




# single_line_data = "Year|2024|x|line_chart GDP_per_capita_in_U.S._dollars|265.58|y|line_chart Year|2023|x|line_chart GDP_per_capita_in_U.S._dollars|270.37|y|line_chart Year|2022|x|line_chart GDP_per_capita_in_U.S._dollars|278.36|y|line_chart Year|2021|x|line_chart GDP_per_capita_in_U.S._dollars|244.0|y|line_chart Year|2020|x|line_chart GDP_per_capita_in_U.S._dollars|243.27|y|line_chart Year|2019|x|line_chart GDP_per_capita_in_U.S._dollars|275.18|y|line_chart Year|2018|x|line_chart GDP_per_capita_in_U.S._dollars|353.17|y|line_chart Year|2017|x|line_chart GDP_per_capita_in_U.S._dollars|273.14|y|line_chart Year|2016|x|line_chart GDP_per_capita_in_U.S._dollars|281.51|y|line_chart Year|2015|x|line_chart GDP_per_capita_in_U.S._dollars|1225.19|y|line_chart Year|2014|x|line_chart GDP_per_capita_in_U.S._dollars|1309.95|y|line_chart "
multi_line_data = "Year|2018|0|line_chart Export|55968.7|1|line_chart Import|108775.3|2|line_chart Year|2017|0|line_chart Export|45622.2|1|line_chart Import|94101.9|2|line_chart Year|2016|0|line_chart Export|48752.7|1|line_chart Import|78531.7|2|line_chart Year|2015|0|line_chart Export|71404.6|1|line_chart Import|88653.6|2|line_chart Year|2014|0|line_chart Export|43256.4|1|line_chart Import|69174.6|2|line_chart Year|2013|0|line_chart Export|91886.1|1|line_chart Import|103475.2|2|line_chart Year|2012|0|line_chart Export|55652.9|1|line_chart Import|72623.0|2|line_chart Year|2011|0|line_chart Export|65749.5|1|line_chart Import|75092.9|2|line_chart Year|2010|0|line_chart Export|48278.3|1|line_chart Import|57152.3|2|line_chart Year|2009|0|line_chart Export|29452.8|1|line_chart Import|44580.8|2|line_chart Year|2008|0|line_chart Export|42525.4|1|line_chart Import|62685.1|2|line_chart Year|2007|0|line_chart Export|38762.0|1|line_chart Import|59961.2|2|line_chart Year|2006|0|line_chart Export|28678.0|1|line_chart Import|42992.7|2|line_chart Year|2005|0|line_chart Export|24045.7|1|line_chart Import|37427.3|2|line_chart "

# summarize(data=bar_data, name="bar_data", title="Test")
# summarize(data = group_bar_data, name = "group_bar_data", title="Test")
# summarize(data = single_line_data, name = "single_line_data", title="Test")
summarize(data = multi_line_data, name = "multi_line_data", title="Test")

# summarize(data=group_bar_data, all_y_label=grp_bar_y_label.rstrip('\n'), name=grp_bar_title, title=grp_bar_title.rstrip('\n'))



### USE THIS PORTION TO RUN ALL CHARTS AT ONCE WITH Y LABELS

# with open(dataPath, 'r', encoding='utf-8') as dataFile, \
#         open(titlePath, 'r', encoding='utf-8') as titleFile, open('all_Y_labels.txt', 'r', encoding='utf-8') as all_y_label:
#     count = 1
#     fileIterators = zip(dataFile.readlines(), titleFile.readlines(), all_y_label.readlines())
#     for data, title, all_y_label in fileIterators:
#         summarize(data=data, all_y_label=all_y_label.rstrip('\n'), name=count, title=title.rstrip('\n'))
#         count += 1


# Single Line Charts

# print("\nChart 2")
# single_line_chart_7 = "Year|2018|x|line_chart Number_of_employees|12239|y|line_chart Year|2017|x|line_chart Number_of_employees|11886|y|line_chart Year|2016|x|line_chart Number_of_employees|11865|y|line_chart Year|2015|x|line_chart Number_of_employees|10997|y|line_chart Year|2014|x|line_chart Number_of_employees|10410|y|line_chart Year|2013|x|line_chart Number_of_employees|9694|y|line_chart Year|2012|x|line_chart Number_of_employees|8966|y|line_chart Year|2011|x|line_chart Number_of_employees|8294|y|line_chart"
# summarize(data = single_line_chart_7, name = "single_line_chart_7", title="Test")

# print("\nChart 3")
# single_line_chart_9 = "Year|2019|x|line_chart Average_attendance|67431|y|line_chart Year|2018|x|line_chart Average_attendance|65765|y|line_chart Year|2017|x|line_chart Average_attendance|63882|y|line_chart Year|2016|x|line_chart Average_attendance|64311|y|line_chart Year|2015|x|line_chart Average_attendance|66186|y|line_chart Year|2014|x|line_chart Average_attendance|67425|y|line_chart Year|2013|x|line_chart Average_attendance|71242|y|line_chart Year|2012|x|line_chart Average_attendance|66632|y|line_chart Year|2011|x|line_chart Average_attendance|65859|y|line_chart Year|2010|x|line_chart Average_attendance|66116|y|line_chart Year|2009|x|line_chart Average_attendance|68888|y|line_chart Year|2008|x|line_chart Average_attendance|72778|y|line_chart "
# summarize(data = single_line_chart_9, name = "single_line_chart_9", title="Test")

# # print("\nChart 4")
# # single_line_chart_15 = "Year|2018|x|line_chart Production_in_billion_cubic_meters|831.8|y|line_chart Year|2017|x|line_chart Production_in_billion_cubic_meters|745.8|y|line_chart Year|2016|x|line_chart Production_in_billion_cubic_meters|727.4|y|line_chart Year|2015|x|line_chart Production_in_billion_cubic_meters|740.3|y|line_chart Year|2014|x|line_chart Production_in_billion_cubic_meters|704.7|y|line_chart Year|2013|x|line_chart Production_in_billion_cubic_meters|655.7|y|line_chart Year|2012|x|line_chart Production_in_billion_cubic_meters|649.1|y|line_chart Year|2011|x|line_chart Production_in_billion_cubic_meters|617.4|y|line_chart Year|2010|x|line_chart Production_in_billion_cubic_meters|575.2|y|line_chart Year|2009|x|line_chart Production_in_billion_cubic_meters|557.6|y|line_chart Year|2008|x|line_chart Production_in_billion_cubic_meters|546.1|y|line_chart Year|2007|x|line_chart Production_in_billion_cubic_meters|521.9|y|line_chart Year|2006|x|line_chart Production_in_billion_cubic_meters|524.0|y|line_chart Year|2005|x|line_chart Production_in_billion_cubic_meters|511.1|y|line_chart Year|2004|x|line_chart Production_in_billion_cubic_meters|526.4|y|line_chart Year|2003|x|line_chart Production_in_billion_cubic_meters|540.8|y|line_chart Year|2002|x|line_chart Production_in_billion_cubic_meters|536.0|y|line_chart Year|2001|x|line_chart Production_in_billion_cubic_meters|555.5|y|line_chart Year|2000|x|line_chart Production_in_billion_cubic_meters|543.2|y|line_chart Year|1998|x|line_chart Production_in_billion_cubic_meters|538.7|y|line_chart "
# # summarize(data = single_line_chart_15, name = "single_line_chart_15", title="Test")

# print("\nChart 5")
# single_line_chart_19 = "Year|2024|x|line_chart Inflation_rate_compared_to_previous_year|3.97|y|line_chart Year|2023|x|line_chart Inflation_rate_compared_to_previous_year|3.98|y|line_chart Year|2022|x|line_chart Inflation_rate_compared_to_previous_year|4.05|y|line_chart Year|2021|x|line_chart Inflation_rate_compared_to_previous_year|4.07|y|line_chart Year|2020|x|line_chart Inflation_rate_compared_to_previous_year|4.09|y|line_chart Year|2019|x|line_chart Inflation_rate_compared_to_previous_year|3.44|y|line_chart Year|2018|x|line_chart Inflation_rate_compared_to_previous_year|3.43|y|line_chart Year|2017|x|line_chart Inflation_rate_compared_to_previous_year|3.6|y|line_chart Year|2016|x|line_chart Inflation_rate_compared_to_previous_year|4.5|y|line_chart Year|2015|x|line_chart Inflation_rate_compared_to_previous_year|4.9|y|line_chart Year|2014|x|line_chart Inflation_rate_compared_to_previous_year|5.8|y|line_chart Year|2013|x|line_chart Inflation_rate_compared_to_previous_year|9.4|y|line_chart Year|2012|x|line_chart Inflation_rate_compared_to_previous_year|10|y|line_chart Year|2011|x|line_chart Inflation_rate_compared_to_previous_year|9.5|y|line_chart Year|2010|x|line_chart Inflation_rate_compared_to_previous_year|10.53|y|line_chart Year|2009|x|line_chart Inflation_rate_compared_to_previous_year|12.31|y|line_chart Year|2008|x|line_chart Inflation_rate_compared_to_previous_year|9.09|y|line_chart Year|2007|x|line_chart Inflation_rate_compared_to_previous_year|6.2|y|line_chart Year|2006|x|line_chart Inflation_rate_compared_to_previous_year|6.7|y|line_chart Year|2005|x|line_chart Inflation_rate_compared_to_previous_year|4.4|y|line_chart Year|2004|x|line_chart Inflation_rate_compared_to_previous_year|3.82|y|line_chart Year|2003|x|line_chart Inflation_rate_compared_to_previous_year|3.86|y|line_chart Year|2002|x|line_chart Inflation_rate_compared_to_previous_year|3.98|y|line_chart Year|2001|x|line_chart Inflation_rate_compared_to_previous_year|4.31|y|line_chart Year|2000|x|line_chart Inflation_rate_compared_to_previous_year|3.83|y|line_chart Year|1999|x|line_chart Inflation_rate_compared_to_previous_year|5.7|y|line_chart Year|1998|x|line_chart Inflation_rate_compared_to_previous_year|13.13|y|line_chart Year|1997|x|line_chart Inflation_rate_compared_to_previous_year|6.84|y|line_chart Year|1996|x|line_chart Inflation_rate_compared_to_previous_year|9.43|y|line_chart Year|1995|x|line_chart Inflation_rate_compared_to_previous_year|9.96|y|line_chart Year|1994|x|line_chart Inflation_rate_compared_to_previous_year|10.28|y|line_chart Year|1993|x|line_chart Inflation_rate_compared_to_previous_year|7.28|y|line_chart Year|1992|x|line_chart Inflation_rate_compared_to_previous_year|9.86|y|line_chart Year|1991|x|line_chart Inflation_rate_compared_to_previous_year|13.48|y|line_chart Year|1990|x|line_chart Inflation_rate_compared_to_previous_year|11.2|y|line_chart Year|1989|x|line_chart Inflation_rate_compared_to_previous_year|4.57|y|line_chart Year|1988|x|line_chart Inflation_rate_compared_to_previous_year|7.21|y|line_chart Year|1987|x|line_chart Inflation_rate_compared_to_previous_year|9.06|y|line_chart Year|1986|x|line_chart Inflation_rate_compared_to_previous_year|8.89|y|line_chart Year|1985|x|line_chart Inflation_rate_compared_to_previous_year|6.25|y|line_chart Year|1984|x|line_chart Inflation_rate_compared_to_previous_year|6.52|y|line_chart "
# summarize(data = single_line_chart_19, name = "single_line_chart_19", title="Test")

# print("\nChart 6")
# single_line_chart_24= "Year|2024|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|90.56|y|line_chart Year|2023|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|87.02|y|line_chart Year|2022|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|84.05|y|line_chart Year|2021|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|81.9|y|line_chart Year|2020|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|78.66|y|line_chart Year|2019|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|76.61|y|line_chart Year|2018|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|79.28|y|line_chart Year|2017|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|70.6|y|line_chart Year|2016|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|65.48|y|line_chart Year|2015|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|68.92|y|line_chart Year|2014|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|81.08|y|line_chart Year|2013|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|78.78|y|line_chart Year|2012|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|76.62|y|line_chart Year|2011|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|68.02|y|line_chart Year|2010|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|57.05|y|line_chart Year|2009|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|48.39|y|line_chart Year|2008|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|60.91|y|line_chart Year|2007|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|42.09|y|line_chart Year|2006|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|37.22|y|line_chart Year|2005|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|31.08|y|line_chart Year|2004|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|24.76|y|line_chart Year|2003|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|21.63|y|line_chart Year|2002|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|20.14|y|line_chart Year|2001|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|19.45|y|line_chart Year|2000|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|19.51|y|line_chart Year|1999|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|15.59|y|line_chart Year|1998|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|14.0|y|line_chart Year|1997|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|15.84|y|line_chart Year|1996|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|15.28|y|line_chart Year|1995|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|13.8|y|line_chart Year|1994|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|12.92|y|line_chart Year|1993|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|12.49|y|line_chart Year|1992|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|12.45|y|line_chart Year|1991|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|11.34|y|line_chart Year|1990|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|11.69|y|line_chart Year|1989|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|9.37|y|line_chart Year|1988|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|8.39|y|line_chart Year|1987|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|8.63|y|line_chart Year|1986|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|8.23|y|line_chart Year|1985|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|10.4|y|line_chart Year|1984|x|line_chart Gross_domestic_product_in_billion_U.S._dollars|9.36|y|line_chart "
# summarize(data = single_line_chart_24, name = "single_line_chart_24", title="Test")

# print("\nChart 7")
# single_line_chart_27= "Year|2018|x|line_chart Average_annual_wages_in_euros|29601|y|line_chart Year|2017|x|line_chart Average_annual_wages_in_euros|29558|y|line_chart Year|2016|x|line_chart Average_annual_wages_in_euros|29843|y|line_chart Year|2015|x|line_chart Average_annual_wages_in_euros|29634|y|line_chart Year|2014|x|line_chart Average_annual_wages_in_euros|29390|y|line_chart Year|2013|x|line_chart Average_annual_wages_in_euros|29277|y|line_chart Year|2012|x|line_chart Average_annual_wages_in_euros|29231|y|line_chart Year|2011|x|line_chart Average_annual_wages_in_euros|30140|y|line_chart Year|2010|x|line_chart Average_annual_wages_in_euros|30620|y|line_chart Year|2009|x|line_chart Average_annual_wages_in_euros|30330|y|line_chart Year|2008|x|line_chart Average_annual_wages_in_euros|30179|y|line_chart Year|2007|x|line_chart Average_annual_wages_in_euros|30148|y|line_chart Year|2006|x|line_chart Average_annual_wages_in_euros|30176|y|line_chart Year|2005|x|line_chart Average_annual_wages_in_euros|29970|y|line_chart Year|2004|x|line_chart Average_annual_wages_in_euros|29636|y|line_chart Year|2003|x|line_chart Average_annual_wages_in_euros|29042|y|line_chart Year|2002|x|line_chart Average_annual_wages_in_euros|29073|y|line_chart Year|2001|x|line_chart Average_annual_wages_in_euros|29272|y|line_chart Year|2000|x|line_chart Average_annual_wages_in_euros|29125|y|line_chart "
# summarize(data = single_line_chart_27, name = "single_line_chart_27", title="Test")

# print("\nChart 8")
# single_line_chart_28= "Year|2022|x|line_chart Net_revenue_in_billion_U.S._dollars|356$|y|line_chart Year|2021|x|line_chart Net_revenue_in_billion_U.S._dollars|316$|y|line_chart Year|2020|x|line_chart Net_revenue_in_billion_U.S._dollars|276$|y|line_chart Year|2019|x|line_chart Net_revenue_in_billion_U.S._dollars|238$|y|line_chart Year|2018|x|line_chart Net_revenue_in_billion_U.S._dollars|201$|y|line_chart Year|2017|x|line_chart Net_revenue_in_billion_U.S._dollars|166$|y|line_chart Year|2016|x|line_chart Net_revenue_in_billion_U.S._dollars|136$|y|line_chart Year|2015|x|line_chart Net_revenue_in_billion_U.S._dollars|107$|y|line_chart Year|2014|x|line_chart Net_revenue_in_billion_U.S._dollars|89$|y|line_chart Year|2013|x|line_chart Net_revenue_in_billion_U.S._dollars|74$|y|line_chart Year|2012|x|line_chart Net_revenue_in_billion_U.S._dollars|61$|y|line_chart Year|2011|x|line_chart Net_revenue_in_billion_U.S._dollars|48$|y|line_chart Year|2010|x|line_chart Net_revenue_in_billion_U.S._dollars|34$|y|line_chart Year|2009|x|line_chart Net_revenue_in_billion_U.S._dollars|25$|y|line_chart Year|2008|x|line_chart Net_revenue_in_billion_U.S._dollars|19$|y|line_chart Year|2007|x|line_chart Net_revenue_in_billion_U.S._dollars|15$|y|line_chart Year|2006|x|line_chart Net_revenue_in_billion_U.S._dollars|11$|y|line_chart Year|2005|x|line_chart Net_revenue_in_billion_U.S._dollars|8$|y|line_chart Year|2004|x|line_chart Net_revenue_in_billion_U.S._dollars|7$|y|line_chart Year|2003|x|line_chart Net_revenue_in_billion_U.S._dollars|5$|y|line_chart Year|2002|x|line_chart Net_revenue_in_billion_U.S._dollars|4$|y|line_chart "
# summarize(data = single_line_chart_28, name = "single_line_chart_28", title="Test")

# print("\nChart 9")
# single_line_chart_31= "Year|2000|x|line_chart Unemployment_rate|6.8|y|line_chart Year|2001|x|line_chart Unemployment_rate|6.4|y|line_chart Year|2002|x|line_chart Unemployment_rate|6.3|y|line_chart Year|2003|x|line_chart Unemployment_rate|5.7|y|line_chart Year|2004|x|line_chart Unemployment_rate|5.7|y|line_chart Year|2005|x|line_chart Unemployment_rate|5.4|y|line_chart Year|2006|x|line_chart Unemployment_rate|5.3|y|line_chart Year|2007|x|line_chart Unemployment_rate|4.8|y|line_chart Year|2008|x|line_chart Unemployment_rate|4.7|y|line_chart Year|2009|x|line_chart Unemployment_rate|7|y|line_chart Year|2010|x|line_chart Unemployment_rate|8.2|y|line_chart Year|2011|x|line_chart Unemployment_rate|8.1|y|line_chart Year|2012|x|line_chart Unemployment_rate|8|y|line_chart Year|2013|x|line_chart Unemployment_rate|7.3|y|line_chart Year|2014|x|line_chart Unemployment_rate|6|y|line_chart Year|2015|x|line_chart Unemployment_rate|5.8|y|line_chart Year|2016|x|line_chart Unemployment_rate|5.2|y|line_chart Year|2017|x|line_chart Unemployment_rate|4.2|y|line_chart Year|2018|x|line_chart Unemployment_rate|3.9|y|line_chart Year|2019|x|line_chart Unemployment_rate|3.6|y|line_chart "
# summarize(data = single_line_chart_31, name = "single_line_chart_31", title="Test")

# print("\nChart 10")
# single_line_chart_36= "Year|2024|x|line_chart Inflation_rate_compared_to_previous_year|2|y|line_chart Year|2023|x|line_chart Inflation_rate_compared_to_previous_year|1.9|y|line_chart Year|2022|x|line_chart Inflation_rate_compared_to_previous_year|1.8|y|line_chart Year|2021|x|line_chart Inflation_rate_compared_to_previous_year|1.4|y|line_chart Year|2020|x|line_chart Inflation_rate_compared_to_previous_year|0.89|y|line_chart Year|2019|x|line_chart Inflation_rate_compared_to_previous_year|0.46|y|line_chart Year|2018|x|line_chart Inflation_rate_compared_to_previous_year|1.48|y|line_chart Year|2017|x|line_chart Inflation_rate_compared_to_previous_year|1.94|y|line_chart Year|2016|x|line_chart Inflation_rate_compared_to_previous_year|0.97|y|line_chart Year|2015|x|line_chart Inflation_rate_compared_to_previous_year|0.71|y|line_chart Year|2014|x|line_chart Inflation_rate_compared_to_previous_year|1.28|y|line_chart Year|2013|x|line_chart Inflation_rate_compared_to_previous_year|1.3|y|line_chart Year|2012|x|line_chart Inflation_rate_compared_to_previous_year|2.19|y|line_chart Year|2011|x|line_chart Inflation_rate_compared_to_previous_year|4.03|y|line_chart Year|2010|x|line_chart Inflation_rate_compared_to_previous_year|2.94|y|line_chart Year|2009|x|line_chart Inflation_rate_compared_to_previous_year|2.76|y|line_chart Year|2008|x|line_chart Inflation_rate_compared_to_previous_year|4.67|y|line_chart Year|2007|x|line_chart Inflation_rate_compared_to_previous_year|2.54|y|line_chart Year|2006|x|line_chart Inflation_rate_compared_to_previous_year|2.24|y|line_chart Year|2005|x|line_chart Inflation_rate_compared_to_previous_year|2.75|y|line_chart Year|2004|x|line_chart Inflation_rate_compared_to_previous_year|3.59|y|line_chart Year|2003|x|line_chart Inflation_rate_compared_to_previous_year|3.52|y|line_chart Year|2002|x|line_chart Inflation_rate_compared_to_previous_year|2.76|y|line_chart Year|2001|x|line_chart Inflation_rate_compared_to_previous_year|4.07|y|line_chart Year|2000|x|line_chart Inflation_rate_compared_to_previous_year|2.26|y|line_chart Year|1999|x|line_chart Inflation_rate_compared_to_previous_year|0.81|y|line_chart Year|1998|x|line_chart Inflation_rate_compared_to_previous_year|7.51|y|line_chart Year|1997|x|line_chart Inflation_rate_compared_to_previous_year|4.44|y|line_chart Year|1996|x|line_chart Inflation_rate_compared_to_previous_year|4.93|y|line_chart Year|1995|x|line_chart Inflation_rate_compared_to_previous_year|4.48|y|line_chart Year|1994|x|line_chart Inflation_rate_compared_to_previous_year|6.27|y|line_chart Year|1993|x|line_chart Inflation_rate_compared_to_previous_year|4.8|y|line_chart Year|1992|x|line_chart Inflation_rate_compared_to_previous_year|6.21|y|line_chart Year|1991|x|line_chart Inflation_rate_compared_to_previous_year|9.33|y|line_chart Year|1990|x|line_chart Inflation_rate_compared_to_previous_year|8.57|y|line_chart Year|1989|x|line_chart Inflation_rate_compared_to_previous_year|5.7|y|line_chart Year|1988|x|line_chart Inflation_rate_compared_to_previous_year|7.15|y|line_chart Year|1987|x|line_chart Inflation_rate_compared_to_previous_year|3.05|y|line_chart Year|1986|x|line_chart Inflation_rate_compared_to_previous_year|2.75|y|line_chart Year|1985|x|line_chart Inflation_rate_compared_to_previous_year|2.46|y|line_chart Year|1984|x|line_chart Inflation_rate_compared_to_previous_year|2.27|y|line_chart "
# summarize(data = single_line_chart_36, name = "single_line_chart_36", title="Test")

# print("\nChart 11")
# single_line_chart_50 ="Year|2019|x|line_chart Dividend_per_share_in_euros|0.9|y|line_chart Year|2018|x|line_chart Dividend_per_share_in_euros|3.25|y|line_chart Year|2017|x|line_chart Dividend_per_share_in_euros|3.65|y|line_chart Year|2016|x|line_chart Dividend_per_share_in_euros|3.25|y|line_chart Year|2015|x|line_chart Dividend_per_share_in_euros|3.25|y|line_chart Year|2014|x|line_chart Dividend_per_share_in_euros|2.45|y|line_chart Year|2013|x|line_chart Dividend_per_share_in_euros|2.25|y|line_chart Year|2012|x|line_chart Dividend_per_share_in_euros|2.2|y|line_chart Year|2011|x|line_chart Dividend_per_share_in_euros|2.2|y|line_chart Year|2010|x|line_chart Dividend_per_share_in_euros|1.85|y|line_chart Year|2009|x|line_chart Dividend_per_share_in_euros|0.0|y|line_chart Year|2008|x|line_chart Dividend_per_share_in_euros|0.6|y|line_chart Year|2007|x|line_chart Dividend_per_share_in_euros|2.0|y|line_chart Year|2006|x|line_chart Dividend_per_share_in_euros|1.5|y|line_chart "
# summarize(data = single_line_chart_50, name = "single_line_chart_50", title="Test")

# print("\nChart 12")
# single_line_chart_89= "Year|2024|x|line_chart Inflation_rate_compared_to_previous_year|4|y|line_chart Year|2023|x|line_chart Inflation_rate_compared_to_previous_year|4|y|line_chart Year|2022|x|line_chart Inflation_rate_compared_to_previous_year|4|y|line_chart Year|2021|x|line_chart Inflation_rate_compared_to_previous_year|3.9|y|line_chart Year|2020|x|line_chart Inflation_rate_compared_to_previous_year|3.52|y|line_chart Year|2019|x|line_chart Inflation_rate_compared_to_previous_year|4.68|y|line_chart Year|2018|x|line_chart Inflation_rate_compared_to_previous_year|2.88|y|line_chart Year|2017|x|line_chart Inflation_rate_compared_to_previous_year|3.68|y|line_chart Year|2016|x|line_chart Inflation_rate_compared_to_previous_year|7.04|y|line_chart Year|2015|x|line_chart Inflation_rate_compared_to_previous_year|15.53|y|line_chart Year|2014|x|line_chart Inflation_rate_compared_to_previous_year|7.82|y|line_chart Year|2013|x|line_chart Inflation_rate_compared_to_previous_year|6.76|y|line_chart Year|2012|x|line_chart Inflation_rate_compared_to_previous_year|5.07|y|line_chart Year|2011|x|line_chart Inflation_rate_compared_to_previous_year|8.44|y|line_chart Year|2010|x|line_chart Inflation_rate_compared_to_previous_year|6.85|y|line_chart Year|2009|x|line_chart Inflation_rate_compared_to_previous_year|11.65|y|line_chart Year|2008|x|line_chart Inflation_rate_compared_to_previous_year|14.11|y|line_chart Year|2007|x|line_chart Inflation_rate_compared_to_previous_year|9.01|y|line_chart Year|2006|x|line_chart Inflation_rate_compared_to_previous_year|9.68|y|line_chart Year|2005|x|line_chart Inflation_rate_compared_to_previous_year|12.68|y|line_chart Year|2004|x|line_chart Inflation_rate_compared_to_previous_year|10.89|y|line_chart Year|2003|x|line_chart Inflation_rate_compared_to_previous_year|13.67|y|line_chart Year|2002|x|line_chart Inflation_rate_compared_to_previous_year|15.78|y|line_chart Year|2001|x|line_chart Inflation_rate_compared_to_previous_year|21.46|y|line_chart Year|2000|x|line_chart Inflation_rate_compared_to_previous_year|20.78|y|line_chart Year|1999|x|line_chart Inflation_rate_compared_to_previous_year|85.74|y|line_chart Year|1998|x|line_chart Inflation_rate_compared_to_previous_year|27.68|y|line_chart Year|1997|x|line_chart Inflation_rate_compared_to_previous_year|14.77|y|line_chart Year|1996|x|line_chart Inflation_rate_compared_to_previous_year|47.74|y|line_chart Year|1995|x|line_chart Inflation_rate_compared_to_previous_year|197.47|y|line_chart Year|1994|x|line_chart Inflation_rate_compared_to_previous_year|307.63|y|line_chart "
# summarize(data = single_line_chart_89, name = "single_line_chart_89", title="Test")






# Multi Line Chart

# print("\nChart 2")
# multi_line_chart_32= "Year|2002|0|line_chart Male|10.4|1|line_chart Female|4.7|2|line_chart Year|2012|0|line_chart Male|10.6|1|line_chart Female|5.1|2|line_chart Year|2016|0|line_chart Male|10.9|1|line_chart Female|5.2|2|line_chart"
# summarize(data = multi_line_chart_32, name = "multi_line_chart_32", title="Test")

# print("\nChart 3")
# multi_line_chart_85= "Year|2006|0|line_chart Establishments|7.6|1|line_chart Employees|156|2|line_chart Year|2007|0|line_chart Establishments|9.3|1|line_chart Employees|184|2|line_chart Year|2008|0|line_chart Establishments|4.5|1|line_chart Employees|137|2|line_chart Year|2009|0|line_chart Establishments|4.9|1|line_chart Employees|115|2|line_chart "
# summarize(data = multi_line_chart_85, name = "multi_line_chart_85", title="Test")


# print("\nChart 4")
# multi_line_chart_96= "Year|2019|0|line_chart Agriculture|1.79|1|line_chart Industry|17.81|2|line_chart Services|80.39|3|line_chart Year|2018|0|line_chart Agriculture|1.81|1|line_chart Industry|17.98|2|line_chart Services|80.21|3|line_chart Year|2017|0|line_chart Agriculture|1.83|1|line_chart Industry|18.17|2|line_chart Services|80.01|3|line_chart Year|2016|0|line_chart Agriculture|1.89|1|line_chart Industry|18.2|2|line_chart Services|79.91|3|line_chart Year|2015|0|line_chart Agriculture|2.04|1|line_chart Industry|18.29|2|line_chart Services|79.68|3|line_chart Year|2014|0|line_chart Agriculture|1.97|1|line_chart Industry|18.6|2|line_chart Services|79.42|3|line_chart Year|2013|0|line_chart Agriculture|2.03|1|line_chart Industry|19.16|2|line_chart Services|78.81|3|line_chart Year|2012|0|line_chart Agriculture|2.06|1|line_chart Industry|19.61|2|line_chart Services|78.33|3|line_chart Year|2011|0|line_chart Agriculture|1.99|1|line_chart Industry|19.93|2|line_chart Services|78.08|3|line_chart Year|2010|0|line_chart Agriculture|2.1|1|line_chart Industry|19.88|2|line_chart Services|78.02|3|line_chart Year|2009|0|line_chart Agriculture|2.18|1|line_chart Industry|20.16|2|line_chart Services|77.67|3|line_chart "
# summarize(data = multi_line_chart_96, name = "multi_line_chart_96", title="Test")

print("\nChart 5")
multi_line_chart_112= "Year|2016|0|line_chart Research_and_development|27169|1|line_chart Sales_and_marketing|20902|2|line_chart General_and_administrative|9695|3|line_chart Operations|14287|4|line_chart Year|2015|0|line_chart Research_and_development|23336|1|line_chart Sales_and_marketing|19082|2|line_chart General_and_administrative|8452|3|line_chart Operations|10944|4|line_chart Year|2014|0|line_chart Research_and_development|20832|1|line_chart Sales_and_marketing|17621|2|line_chart General_and_administrative|7510|3|line_chart Operations|7637|4|line_chart Year|2013|0|line_chart Research_and_development|18593|1|line_chart Sales_and_marketing|15348|2|line_chart General_and_administrative|6563|3|line_chart Operations|7252|4|line_chart Year|2012|0|line_chart Research_and_development|19746|1|line_chart Sales_and_marketing|15306|2|line_chart General_and_administrative|6214|3|line_chart Operations|12595|4|line_chart Year|2011|0|line_chart Research_and_development|11665|1|line_chart Sales_and_marketing|11933|2|line_chart General_and_administrative|4651|3|line_chart Operations|4218|4|line_chart Year|2010|0|line_chart Research_and_development|9508|1|line_chart Sales_and_marketing|8778|2|line_chart General_and_administrative|3346|3|line_chart Operations|2768|4|line_chart Year|2009|0|line_chart Research_and_development|7443|1|line_chart Sales_and_marketing|7338|2|line_chart General_and_administrative|2941|3|line_chart Operations|2113|4|line_chart Year|2008|0|line_chart Research_and_development|7254|1|line_chart Sales_and_marketing|8002|2|line_chart General_and_administrative|3109|3|line_chart Operations|1857|4|line_chart"
summarize(data = multi_line_chart_112, all_y_label="TEST", name = "multi_line_chart_112", title="Test")

# print("\nChart 6")
# multi_line_chart_117="Year|2010|0|line_chart Anthracite_and_bituminous|404762|1|line_chart Sub-bituminous_and_lignite|456176|2|line_chart Year|2012|0|line_chart Anthracite_and_bituminous|404762|1|line_chart Sub-bituminous_and_lignite|456176|2|line_chart Year|2014|0|line_chart Anthracite_and_bituminous|403199|1|line_chart Sub-bituminous_and_lignite|488332|2|line_chart Year|2016|0|line_chart Anthracite_and_bituminous|816214|1|line_chart Sub-bituminous_and_lignite|323117|2|line_chart Year|2018|0|line_chart Anthracite_and_bituminous|734903|1|line_chart Sub-bituminous_and_lignite|319879|2|line_chart "
# summarize(data = multi_line_chart_117, name = "multi_line_chart_117", title="Test")

# print("\nChart 7")
# multi_line_chart_128="Year|2012|0|line_chart Taste|87|1|line_chart Price|73|2|line_chart Healthfulness|61|3|line_chart Convenience|53|4|line_chart Sustainability|35|5|line_chart Year|2013|0|line_chart Taste|89|1|line_chart Price|71|2|line_chart Healthfulness|64|3|line_chart Convenience|56|4|line_chart Sustainability|36|5|line_chart Year|2014|0|line_chart Taste|90|1|line_chart Price|73|2|line_chart Healthfulness|71|3|line_chart Convenience|51|4|line_chart Sustainability|38|5|line_chart Year|2015|0|line_chart Taste|83|1|line_chart Price|68|2|line_chart Healthfulness|60|3|line_chart Convenience|52|4|line_chart Sustainability|35|5|line_chart Year|2016|0|line_chart Taste|84|1|line_chart Price|71|2|line_chart Healthfulness|64|3|line_chart Convenience|52|4|line_chart Sustainability|41|5|line_chart "
# summarize(data = multi_line_chart_128, name = "multi_line_chart_128", title="Test")

# print("\nChart 8")
# multi_line_chart_140= "Year|2019|0|line_chart EMA|12.89|1|line_chart Americas|11.72|2|line_chart Asia_Pacific|5.14|3|line_chart Year|2018|0|line_chart EMA|12.98|1|line_chart Americas|11.1|2|line_chart Asia_Pacific|4.88|3|line_chart Year|2017|0|line_chart EMA|11.5|1|line_chart Americas|10.48|2|line_chart Asia_Pacific|4.42|3|line_chart Year|2016|0|line_chart EMA|11.34|1|line_chart Americas|10.02|2|line_chart Asia_Pacific|4.06|3|line_chart Year|2015|0|line_chart EMA|11.31|1|line_chart Americas|9.34|2|line_chart Asia_Pacific|3.79|3|line_chart Year|2014|0|line_chart EMA|12.45|1|line_chart Americas|8.51|2|line_chart Asia_Pacific|3.86|3|line_chart Year|2013|0|line_chart EMA|11.64|1|line_chart Americas|7.88|2|line_chart Asia_Pacific|3.9|3|line_chart Year|2012|0|line_chart EMA|11.51|1|line_chart Americas|7.45|2|line_chart Asia_Pacific|4.07|3|line_chart Year|2011|0|line_chart EMA|11.66|1|line_chart Americas|7.05|2|line_chart Asia_Pacific|4.0|3|line_chart Year|2010|0|line_chart EMA|10.83|1|line_chart Americas|6.37|2|line_chart Asia_Pacific|3.43|3|line_chart "
# summarize(data = multi_line_chart_140, name = "multi_line_chart_140", title="Test")

# print("\nChart 9")
# multi_line_chart_274= "Year|2019|0|line_chart Jan|11.3|1|line_chart Feb|8.6|2|line_chart Mar|7.6|3|line_chart Apr|6.4|4|line_chart May|0|5|line_chart Jun|0|6|line_chart Jul|0|7|line_chart Aug|0|8|line_chart Sep|0|9|line_chart Oct|0|10|line_chart Nov|0|11|line_chart Dec|0|12|line_chart Year|2018|0|line_chart Jan|10.2|1|line_chart Feb|12.4|2|line_chart Mar|10.6|3|line_chart Apr|6.0|4|line_chart May|2.8|5|line_chart Jun|0.5|6|line_chart Jul|0|7|line_chart Aug|0.5|8|line_chart Sep|2.1|9|line_chart Oct|4.9|10|line_chart Nov|7.3|11|line_chart Dec|8.7|12|line_chart Year|2017|0|line_chart Jan|11.2|1|line_chart Feb|9.3|2|line_chart Mar|7.0|3|line_chart Apr|6.5|4|line_chart May|2.9|5|line_chart Jun|0.7|6|line_chart Jul|0.1|7|line_chart Aug|0.5|8|line_chart Sep|2.1|9|line_chart Oct|3.2|10|line_chart Nov|8.5|11|line_chart Dec|10.4|12|line_chart Year|2016|0|line_chart Jan|9.8|1|line_chart Feb|10.4|2|line_chart Mar|9.4|3|line_chart Apr|8.0|4|line_chart May|3.4|5|line_chart Jun|0.9|6|line_chart Jul|0.4|7|line_chart Aug|0.1|8|line_chart Sep|0.7|9|line_chart Oct|4.6|10|line_chart Nov|9.7|11|line_chart Dec|9|12|line_chart Year|2015|0|line_chart Jan|10.7|1|line_chart Feb|11.2|2|line_chart Mar|9.2|3|line_chart Apr|6.4|4|line_chart May|4.6|5|line_chart Jun|1.9|6|line_chart Jul|0.7|7|line_chart Aug|0.4|8|line_chart Sep|2.8|9|line_chart Oct|4.6|10|line_chart Nov|6|11|line_chart Dec|6|12|line_chart Year|2014|0|line_chart Jan|9.9|1|line_chart Feb|9.2|2|line_chart Mar|7.9|3|line_chart Apr|5.4|4|line_chart May|3.3|5|line_chart Jun|0.6|6|line_chart Jul|0.1|7|line_chart Aug|0.8|8|line_chart Sep|0.9|9|line_chart Oct|3.3|10|line_chart Nov|7.1|11|line_chart Dec|10|12|line_chart "
# summarize(data = multi_line_chart_274, name = "multi_line_chart_274", title="Test")
