import json  # Serialization: process of encoding data into JSON format (like converting a Python list to JSON). Deserialization: process of decoding JSON data back into native objects you can work with (like reading JSON data into a Python list)

import math  # To use mathematical functions
import re  # Regular Expression, The functions in this module let you check if a particular string matches a given regular expression
import random  # random number generation. random() function, generates random numbers between 0 and 1.
from random import randint  # randint() is an inbuilt function of the random module in Python3
from statistics import mean, median  # mean() function can be used to calculate mean/average of a given list of numbers.
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

constant_rate = 20  # avg(% chnage)*0.1 # Meaning any chnage less than 5% is considered roughly constant slope  # Determines if a trend is increasing, decreasing or constant
significant_rate = 30  # avg(% chnage)*0.1 # Meaning any chnage >constant rate and less than this rate is considered not significant and so it's trend direction is chnaged to the trend of the succesive interval # Determines the start and end of the trend


def directionTrend(new, old):
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


def rateOfChnage(new, old):
    if (old != 0):
        percentageChange = ((new - old) / old) * 100
    else:
        old = 0.00000000001
        percentageChange = ((new - old) / old) * 100

    absChnage = abs(percentageChange)
    if (absChnage > 60):
        return "rapidly"
    elif (absChnage > 30):
        return "gradually"
    elif (absChnage > constant_rate):
        return "slightly"
    else:
        return "roughly"


def increaseDecrease(x):
    if (x == "increasing"):
        return "increase"
    elif (x == "decreasing"):
        return "decrease"
    else:
        return "stays the same"


def get_indexes_max_value(l):
    max_value = max(l)
    return [i for i, x in enumerate(l) if x == max(l)]


def get_indexes_min_value(l):
    min_value = min(l)
    return [i for i, x in enumerate(l) if x == min(l)]


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

        print("columnCount -> " + str(columnCount))

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

        ## for multi line charts
        elif (chartType == "line"):
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

            summaryArr = []

            summary1 = "This is a multi-line chart with " + str(lineCount) + " lines representing " + line_names + ". "
            summary2 = "The line for " + str(maxLine[0]) + " has the highest values across " + str(
                stringLabels[0]) + " with a mean value of " + str(maxLine[1]) + ", "
            summary3 = "and it peaked at " + str(maxXValue) + " with a value of " + str(maxLineData) + "."

            summaryArr.append(summary1)
            summaryArr.append(summary2)
            summaryArr.append(summary3)

            if lineCount > 2:
                summary4 = "Followed by " + str(secondLine[0]) + ", with a mean value of " + str(secondLine[1]) + ". "
                summary5 = "This line peaked at " + str(secondXValue) + " with a value of " + str(secondLineData) + ". "
                summaryArr.append(summary4)
                summaryArr.append(summary5)

            summary6 = "The least dominant line is " + str(minLine[0]) + " with a mean value of " + str(
                minLine[1]) + ", "
            summary7 = "which peaked at " + str(minXValue) + " having " + str(minLineData) + ". "
            summaryArr.append(summary6)
            summaryArr.append(summary7)

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
            yValueArrCorrectOrder = yValueArr[len(yValueArr)::-1]  ## Ordered correctly this time
            # print(yValueArrCorrectOrder)
            directionArray = []
            i = 1
            while i < (len(yValueArrCorrectOrder)):
                d = directionTrend(yValueArrCorrectOrder[i],
                                   yValueArrCorrectOrder[i - 1])  # direction e.g. increase, decrease or constant
                directionArray.append(d)
                i = i + 1
            print("Orginal Direction Trend")
            print(directionArray)

            ############# GlobalTrend ##############
            globalDifference = float(yValueArr[0]) - float(yValueArr[len(yValueArr) - 1])
            globalPercentChange = (globalDifference / float(yValueArr[len(yValueArr) - 1])) * 100
            # print(yValueArr)
            # print(globalDifference)
            # print(globalPercentChange)

            localTrendSentence1 = "This line chart has an x axis representing " + xLabel + " and a y axis representing " + yLabel + ", with a total of " + str(
                len(yValueArr)) \
                                  + " data points."
            summaryArray.append(localTrendSentence1)

            summary2 = " Overall " + yLabel + " has "
            if globalPercentChange > 0:
                summary2 += "increased"
            elif globalPercentChange < 0:
                summary2 += "decreased"
            else:
                summary2 += "constant"

            summary2 += " over the " + xLabel + "."
            summaryArray.append(summary2)

            ############# LocalTrend ##############
            varianceArray = []
            percentArray = []
            i = 1
            while i < (len(yValueArr)):
                old = yValueArr[i]
                if (old == 0 or old == 0.0):
                    old = 0.00000000001
                variance1 = float(yValueArr[i - 1]) - float(
                    old)  # 2nd yVal- Prev yVal # Note that xValueArr and yValueArr are ordered such that the start values are written at the end of the array
                localPercentChange = (variance1 / float(old)) * 100
                varianceArray.append(variance1)
                percentArray.append(localPercentChange)
                i = i + 1

            # print(varianceArray)
            # print(percentArray)

            varianceArrayCorrectOrder = varianceArray[len(varianceArray)::-1]  ## Ordered correctly this time
            percentArrayCorrectOrder = percentArray[len(percentArray)::-1]  ## Ordered correctly this time

            # print(varianceArrayCorrectOrder)
            print(percentArrayCorrectOrder)

            ### Previously indexs reported for only increasing and decresing trends
            # trendChangeIdx = []
            # for idx in range(0, len(varianceArrayCorrectOrder) - 1):

            #     # checking for successive opposite index
            #     if varianceArrayCorrectOrder[idx] > 0 and varianceArrayCorrectOrder[idx + 1] < 0 or varianceArrayCorrectOrder[idx] < 0 and varianceArrayCorrectOrder[idx + 1] > 0:
            #         trendChangeIdx.append(idx)

            # print("Sign shift indices : " + str(trendChangeIdx))

            ## Smoothing directionArray. If percentChange >10% then direction of trend is that of the next interval (regardless if it was increasing or decreasing)
            directionArraySmoothed = []
            for idx in range(0, len(percentArrayCorrectOrder) - 1):
                # checking for percent chnage >5% (not constant) and <10% (not significant) and chnaging their direction to be the direction of the succesive interval
                if (percentArrayCorrectOrder[idx] > constant_rate and percentArrayCorrectOrder[idx] < significant_rate):
                    d = directionArray[idx + 1]
                    directionArraySmoothed.append(d)
                else:
                    directionArraySmoothed.append(directionArray[idx])
            directionArraySmoothed.append(directionArray[len(
                percentArrayCorrectOrder) - 1])  # The last value doesn't have a succesive interval so it will be appended as is
            print("Smoothed Direction Trend")
            print(directionArraySmoothed)

            trendChangeIdx = []
            for idx in range(0, len(directionArraySmoothed) - 1):

                # checking for successive opposite index
                if directionArraySmoothed[idx] != directionArraySmoothed[idx + 1]:
                    trendChangeIdx.append(idx)

            print("Sign shift indices : " + str(trendChangeIdx))

            yValueArrCorrectOrder = yValueArr[len(yValueArr)::-1]  ## Ordered correctly this time
            # print(yValueArrCorrectOrder)

            xValueArrCorrectOrder = xValueArr[len(xValueArr)::-1]  ## Ordered correctly this time
            # print(xValueArrCorrectOrder)

            # trendArrayCorrectOrder = trendArray[len(trendArray)::-1] # no need since have my own directionArray now ordered correctly
            # print(trendArrayCorrectOrder)

            summary3 = yLabel
            print(trendChangeIdx)
            x = 0
            if trendChangeIdx:  # if trendChangeIdx is not empty

                for i in trendChangeIdx:
                    if (x == 0):
                        summary3 += " is " + rateOfChnage(yValueArrCorrectOrder[i + 1],
                                                          yValueArrCorrectOrder[0]) + " " + directionArraySmoothed[
                                        i] + " from " + xValueArrCorrectOrder[0] + " to " + xValueArrCorrectOrder[
                                        i + 1] + ", "
                    else:
                        summary3 += rateOfChnage(yValueArrCorrectOrder[i + 1],
                                                 yValueArrCorrectOrder[trendChangeIdx[x - 1] + 1]) + " " + \
                                    directionArraySmoothed[i] + " from " + xValueArrCorrectOrder[
                                        trendChangeIdx[x - 1] + 1] + " to " + xValueArrCorrectOrder[i + 1] + ", "
                    x = x + 1

                summary3 += "and lastly " + rateOfChnage(yValueArrCorrectOrder[-1],
                                                         yValueArrCorrectOrder[trendChangeIdx[-1] + 1]) + " " + \
                            directionArraySmoothed[-1] + " from " + xValueArrCorrectOrder[
                                trendChangeIdx[-1] + 1] + " to " + xValueArrCorrectOrder[-1] + "."
            else:
                summary3 += " is " + rateOfChnage(yValueArrCorrectOrder[-1], yValueArrCorrectOrder[0]) + " " + \
                            directionArraySmoothed[-1] + " from " + xValueArrCorrectOrder[0] + " to " + \
                            xValueArrCorrectOrder[-1] + "."

            summaryArray.append(summary3)

            ############# Steepest Slope ##############

            # Absolute value of varianceArrayCorrectOrder elements
            absoluteVariance = [abs(ele) for ele in varianceArrayCorrectOrder]

            max_value = max(absoluteVariance)
            max_index = absoluteVariance.index(max_value)

            # print(absoluteVariance)
            # print(max_value)
            # print(max_index)
            # print(directionArraySmoothed)

            summary4 = "The steepest " + increaseDecrease(
                directionArraySmoothed[max_index]) + " occurs in between the " + xLabel + " " + xValueArrCorrectOrder[
                           max_index] + " and " + xValueArrCorrectOrder[max_index + 1] + "."
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
                    summary5 += " " + xValueArrCorrectOrder[max_index[i]] + ", "
                    i = i + 1
                summary5 += "and " + xValueArrCorrectOrder[max_index[-1]]
            else:
                summary5 += " " + xValueArrCorrectOrder[max_index[0]]

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
                    summary5 += " " + xValueArrCorrectOrder[min_index[i]] + ", "
                    i = i + 1
                summary5 += "and " + xValueArrCorrectOrder[min_index[-1]]
            else:
                summary5 += " " + xValueArrCorrectOrder[min_index[0]]

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
single_line_data = "Year|2013|x|line_chart Amount_spent_in_U.S._dollars|46.58|y|line_chart Year|2014|x|line_chart Amount_spent_in_U.S._dollars|47.39|y|line_chart Year|2015|x|line_chart Amount_spent_in_U.S._dollars|51.52|y|line_chart Year|2016|x|line_chart Amount_spent_in_U.S._dollars|56.15|y|line_chart "

# summarize(data=bar_data, name="bar_data", title="Test")
# summarize(data = group_bar_data, name = "group_bar_data", title="Test")
# summarize(data = single_line_data, name = "single_line_data", title="Test")
# summarize(data = multi_line_data, name = "multi_line_data", title="Test")

# summarize(data=group_bar_data, all_y_label=grp_bar_y_label.rstrip('\n'), name=grp_bar_title, title=grp_bar_title.rstrip('\n'))


#
# file = open(dataPath, 'r')
# lines = file.readlines()
#
# count = 0
#
# for line in lines:
#     count += 1
#     print(line)
#     summarize(data = line, name = count, title="Test")


with open(dataPath, 'r', encoding='utf-8') as dataFile, \
        open(titlePath, 'r', encoding='utf-8') as titleFile, open('all_Y_labels.txt', 'r', encoding='utf-8') as all_y_label:
    count = 1
    fileIterators = zip(dataFile.readlines(), titleFile.readlines(), all_y_label.readlines())
    for data, title, all_y_label in fileIterators:
        summarize(data=data, all_y_label=all_y_label.rstrip('\n'), name=count, title=title.rstrip('\n'))
        count += 1
