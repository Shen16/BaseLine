import json
import math
import re
import random
from random import randint
from statistics import mean
from operator import itemgetter
from scipy.stats import linregress
from sklearn import preprocessing
import pandas as pd
import numpy as np

dataPath = 'Data/test/testData.txt'
titlePath = 'Data/test/testTitle.txt'

# websitePath = 'results/generated_baseline'
websitePath = 'static/generated'

summaryList = []


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


# scaler = preprocessing.MinMaxScaler()
count = 0


# with open(dataPath, 'r', encoding='utf-8') as dataFile, \
#         open(titlePath, 'r', encoding='utf-8') as titleFile:
#
#     fileIterators = zip(dataFile.readlines(), titleFile.readlines())
#     for data, title in fileIterators:

def summarize(data, name, title):
    scaler = preprocessing.MinMaxScaler()
    # count += 1
    datum = data.split()
    # check if data is multi column
    columnType = datum[0].split('|')[2].isnumeric()

    # print("Column Type -> " + str(columnType) + " this is -> " + str(datum[0].split('|')[2]))

    if columnType:  # MULTI
        labelArr = []
        chartType = datum[0].split('|')[3].split('_')[0]

        values = [value.split('|')[1] for value in datum]

        # print("VALUES")
        # for a in values:
        #     print(a)

        # find number of columns:
        columnCount = max([int(data.split('|')[2]) for data in datum]) + 1
        # Get labels
        for i in range(columnCount):
            label = datum[i].split('|')[0].split('_')
            labelArr.append(label)

        # print(labelArr)

        stringLabels = [' '.join(label) for label in labelArr]

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
            groupedLabels.append(str(stringLabels[i]).replace('_',' '))


        print("groupedLabels")
        for a in groupedLabels:
            print(a)


        a = 0
        b = 0

        groupedCol = int(len(values) / len(stringLabels))

        row = groupedCol
        col = columnCount
        arr = np.empty((row, col), dtype=object)
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
                arr[m][b % columnCount] = str(values[b]).replace('_',' ')
            else:
                arr[m][b % columnCount] = float(values[b])

            n += 1
            a += 1

        max_row = []
        max_row_val = []
        min_row = []
        min_row_val = []

        number_of_group = len(groupedLabels)-1

        for i in range(len(groupedLabels)-1):
            arr1 = arr[arr[:, (i+1)].argsort()]
            min_row.append(arr1[0][0])
            min_row_val.append(arr1[0][i+1])
            arr2 = arr[arr[:, (i+1)].argsort()[::-1]]
            max_row.append(arr2[0][0])
            max_row_val.append(arr2[0][i+1])

        print("MAX")
        for a in max_row:
            print(a)
        for a in max_row_val:
            print(a)
        print("MIN")
        for a in min_row:
            print(a)
        for a in min_row_val:
            print(a)

        group_max_min_without_value = (" In case of " + str(max_row[0]) + ", " + str(groupedLabels[1]) + " shows more dominance than any other " + str(groupedLabels[0]) + " and it has the lowest value in " + str(min_row[0]) + ". ")



        if len(groupedLabels) > 3:
            if float(random.uniform(0, 1)) < 0.70:
                group_max_min_without_value += (
                            "On the other hand, group " + str(groupedLabels[-1]) + " is highest in case of " + str(
                        max_row[-1]) + " and has minimum impact for " + str(groupedLabels[0]) + " " + str(min_row[-1]) + ". ")

            else:
                group_max_min_without_value += (
                            "On the other hand, group " + str(groupedLabels[2]) + " is the second impactful group in this chart, and it's highest in case of " + str(
                        max_row[2]) + " and has the minimum dominance for " + str(groupedLabels[0]) + " " + str(min_row[2]) + ". ")

        # print(group_max_min_without_value)

        # group_max_min_with_value = (" Group " + str(groupedLabels[1]) + " shows dominance in case of " + str(max_row[0]) + " with a value " + str(max_row_val[0]) + " and has less significance for " + str(groupedLabels[0]) + " " + str(min_row[0]) + " with only " + str(min_row_val[0]) + ". ")
        group_max_min_with_value = (" In case of " + str(max_row[0]) + ", " + str(groupedLabels[1]) + " shows more dominance than any other " + str(groupedLabels[0]) + " with a value " + str(max_row_val[0]) + " and it has the lowest value " + str(min_row_val[0]) + " in " + str(min_row[0]) + ". ")


        if len(groupedLabels) > 3:
            if float(random.uniform(0, 1)) < 0.50:
                group_max_min_with_value += (" However, " + str(groupedLabels[-1]) + " being the 2nd most important group, has the highest value " + str(
                    max_row_val[-1]) + " for " + str(groupedLabels[0]) + " " + str(
                    max_row[-1]) + " and the lowest " + str(min_row_val[-1]) + " in case of " + str(min_row[-1]) + ". ")

            else:
                group_max_min_with_value += (" However, " + str(groupedLabels[2]) + " is the second impactful group, and it has the highest value " + str(
                    max_row_val[2]) + " for " + str(groupedLabels[0]) + " " + str(
                    max_row[2]) + " and the lowest " + str(min_row_val[2]) + " in case of " + str(min_row[2]) + ". ")

        # print(group_max_min_with_value)

        chosen_summary = ""

        if float(random.uniform(0, 1)) < 0.50:
            chosen_summary = group_max_min_without_value
        else:
            chosen_summary = group_max_min_with_value

        rowCount = round(len(datum) / columnCount)
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
            if rowCount > 2:
                for category in categoricalValueArr:
                    meanCategoricalDict[category[0]] = mean(category[1:])
                sortedCategories = sorted(meanCategoricalDict.items(), key=lambda x: x[1])
                numerator = abs(sortedCategories[-1][1] - sortedCategories[-2][1])
                denominator = (sortedCategories[-1][1] + sortedCategories[-2][1]) / 2
                topTwoDelta = round((numerator / denominator) * 100, 1)

                numerator1 = abs(sortedCategories[-1][1] - sortedCategories[0][1])
                denominator1 = (sortedCategories[-1][1] + sortedCategories[0][1]) / 2
                minMaxDelta = round((numerator1 / denominator1) * 100, 1)

                summary1 = f"This grouped bar chart has {rowCount} categories of {stringLabels[0]} on the x axis representing {str(number_of_group)} groups: {categories}."
                summary2 = f" Averaging these {str(number_of_group)} groups, the highest category is found for {str(groupedLabels[0])} {sortedCategories[-1][0]} with a mean value of {round(sortedCategories[-1][1],2)}."
                summaryArray.append(summary1)
                summaryArray.append(summary2)
                maxValueIndex = cleanValArr[0].index(sortedCategories[-1][0])
                secondValueIndex = cleanValArr[0].index(sortedCategories[-2][0])

                if float(random.uniform(0, 1)) < 0.60:
                    summary3 = f" Followed by {sortedCategories[-2][0]} with a mean value of {round(sortedCategories[-2][1],2)}."
                    summaryArray.append(summary3)

                # if topTwoDelta >= 3:
                #     summary4 = f" This represents a difference of {topTwoDelta}%."
                #     summaryArray.append(summary4)

                summary5 = f" The lowest category is observed at {sortedCategories[0][0]} that has a mean value of {round(sortedCategories[0][1], 2)}"
                summaryArray.append(summary5)

                if minMaxDelta >= 2:
                    summary6 = f" showing a difference of {minMaxDelta}% with the maximum value."
                    summaryArray.append(summary6)

                summaryArray.append(chosen_summary)

                trendsArray = [
                    {},{"2": ["0", str(maxValueIndex)], "13": [str(columnCount-1), str(maxValueIndex)]},
                    {"2": ["0", str(secondValueIndex)], "14": [str(columnCount-1), str(secondValueIndex)]},{}
                ]
            elif rowCount == 2:
                for category in categoricalValueArr:
                    meanCategoricalDict[category[0]] = mean(category[1:])
                sortedCategories = sorted(meanCategoricalDict.items(), key=lambda x: x[1])
                numerator = abs(sortedCategories[-1][1] - sortedCategories[-2][1])
                denominator = (sortedCategories[-1][1] + sortedCategories[-2][1]) / 2
                topTwoDelta = round((numerator / denominator) * 100, 1)

                summary1 = f"This grouped bar chart has {rowCount} categories of {stringLabels[0]} on the x axis representing {str(number_of_group)} groups: {categories}."
                summary2 = f" Averaging the {str(number_of_group)} groups, the highest category is found for {str(groupedLabels[0])} {sortedCategories[-1][0]} with a mean value of {round(sortedCategories[-1][1],2)}."
                summaryArray.append(summary1)
                summaryArray.append(summary2)
                maxValueIndex = cleanValArr[0].index(sortedCategories[-1][0])
                secondValueIndex = cleanValArr[0].index(sortedCategories[-2][0])
                summary3 = f" The minimum category is found at {sortedCategories[-2][0]} with a mean value of {round(sortedCategories[-2][1],2)}."
                summaryArray.append(summary3)

                if topTwoDelta >= 5:
                    summary4 = f" This represents a difference of {topTwoDelta}%."
                    summaryArray.append(summary4)

                summaryArray.append(chosen_summary)
                trendsArray = [
                    {},{"2": ["0", str(maxValueIndex)], "13": [str(columnCount-1), str(maxValueIndex)]},
                    {"2": ["0", str(secondValueIndex)], "14": [str(columnCount-1), str(secondValueIndex)]},{}
                ]
            else:
                summary1 = f"This grouped bar chart has 1 category for the x axis of {stringLabels[0]}."
                summary2 = f" This category is {stringLabels[1]}, with a mean value of {round(mean(categoricalValueArr[1]),2)}."
                summaryArray.append(summary1)
                summaryArray.append(summary2)
                summaryArray.append(chosen_summary)
                trendsArray = [{},{"3": ["0", "0"], "9": ["0", "0"]}]
            websiteInput = {"title": title.strip(), "labels": [' '.join(label) for label in labelArr],
                            "columnType": "multi",
                            "graphType": chartType, "summaryType": "baseline", "summary": summaryArray,
                            "trends": trendsArray,
                            "data": dataJson}
            with open(f'{websitePath}/{name}.json', 'w', encoding='utf-8') as websiteFile:
                json.dump(websiteInput, websiteFile, indent=3)
            # oneFile.writelines(''.join(summaryArray)+'\n')
        elif (chartType == "line"):
            # clean data
            intData = []
            # print(valueArr[1:])
            for line in valueArr[1:]:
                cleanLine = []
                for data in line:
                    if data.isnumeric():
                        cleanLine.append(float(data))
                    else:
                        cleanData = re.sub("[^\d\.]", "", data)
                        if len(cleanData) > 0:
                            cleanLine.append(float(cleanData[:4]))
                        else:
                            cleanLine.append(float(cleanData))
                intData.append(cleanLine)
                # print(len(intData))
            # calculate mean for each line
            meanLineVals = []
            assert len(stringLabels[1:]) == len(intData)
            for label, data in zip(stringLabels[1:], intData):
                x = (label, round(mean(data), 1))
                # print(x)
                meanLineVals.append(x)
            sortedLines = sorted(meanLineVals,key=itemgetter(1))
            # if more than 2 lines
            lineCount = len(labelArr) - 1
            maxLine = sortedLines[-1]
            index1 = stringLabels.index(maxLine[0]) - 1
            maxLineData = round(max(intData[index1]), 2)
            maxXValue = valueArr[0][intData[index1].index(maxLineData)]

            secondLine = sortedLines[-2]
            rowIndex1 = intData[index1].index(maxLineData)
            index2 = stringLabels.index(secondLine[0]) - 1
            secondLineData = round(max(intData[index2]), 2)
            secondXValue = valueArr[0][intData[index2].index(secondLineData)]
            rowIndex2 = intData[index2].index(secondLineData)

            minLine = sortedLines[0]
            index_min = stringLabels.index(minLine[0]) - 1
            minLineData = round(max(intData[index_min]), 2)
            minXValue = valueArr[0][intData[index_min].index(minLineData)]

            line_names = ""
            for i in range(len(stringLabels) - 1):
                if i < len(stringLabels) - 2:
                    line_names += stringLabels[i+1] + ", "
                else:
                    line_names += "and " + stringLabels[i+1]

            summaryArr = []

            summary1 = "This is a multi-line chart with " + str(lineCount) + " lines representing " + line_names + ". "
            summary2 = "The line for " + str(maxLine[0]) + " has the highest values across " + str(stringLabels[0]) + " with a mean value of " + str(maxLine[1]) + ", "
            summary3 = "and it peaked at " + str(maxXValue) + " with a value of " + str(maxLineData) + "."

            summaryArr.append(summary1)
            summaryArr.append(summary2)
            summaryArr.append(summary3)

            if lineCount > 2:
                summary4 = "Followed by " + str(secondLine[0]) + ", with a mean value of " + str(secondLine[1]) + ". "
                summary5 = "This line peaked at " + str(secondXValue) + " with a value of " + str(secondLineData) + ". "
                summaryArr.append(summary4)
                summaryArr.append(summary5)

            summary6 = "The least dominant line is " + str(minLine[0]) + " with a mean value of " + str(minLine[1]) + ", "
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
        avgValueOfAllBars = totalValue/len(yValueArr)

        derived_val_avg = "The average " + yLabel + " for all " + str(len(yValueArr)) + " " + xLabel + "s is " + str(avgValueOfAllBars)

        print(derived_val_avg)

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

        if type(yValueArr[maxValueIndex]) == int or type(yValueArr[maxValueIndex]) == float:
            # proportion = int(math.ceil(yValueArr[maxValueIndex] / yValueArr[minValueIndex]))
            proportion = round((yValueArr[maxValueIndex] / yValueArr[minValueIndex]), 2)
            max_avg_diff_rel = round((yValueArr[maxValueIndex] / avgValueOfAllBars), 2)
            max_min_diff = (yValueArr[maxValueIndex] - yValueArr[minValueIndex])
            max_avg_diff_abs = (yValueArr[maxValueIndex] - avgValueOfAllBars)

            # print("proportion -> " + str(proportion))
            # print("max_min_diff -> " + str(max_min_diff))
            # print("max_avg_diff_rel -> " + str(max_avg_diff_rel))
            # print("max_avg_diff -> " + str(max_avg_diff_abs))
        else:
            print('The variable is not a number')


        # run pie
        if (chartType == "pie"):

            summary1 = "This is a pie chart showing the distribution of " + str(len(xValueArr)) + " different " + xLabel + "."
            summary2 = xValueArr[maxValueIndex] + " " + xLabel + " has the highest proportion with " + str(maxPercentage) + "% of the pie chart area"
            summary3 = "followed by " + xLabel + " " + xValueArr[secondMaxIndex] + ", with a proportion of " + str(secondMaxPercentage) + "%. "
            summary4 = "Finally, " + xLabel + " " + xValueArr[minValueIndex] + " has the minimum contribution of " + str(minPercentage) + "%."

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

            summary1 = "This bar chart has " + str(len(xValueArr)) + " categories on the x axis representing " + xLabel + ", and " + yLabel + " in each " + xLabel + " on the y axis."
            summaryArray.append(summary1)

            summary2_extrema_max = " The highest category is found at " + position_in_X_axis_for_max_value + " where " + yLabel + " is " + str(yValueArr[maxValueIndex]) + "."

            summary3_order_2nd_max = ""
            if len(xValueArr) > 2:
                summary3_order_2nd_max = " Followed by " + str(yValueArr[secondMaxIndex]) + " in " + position_in_X_axis_for_second_max_value + "."

            summary4_extrema_min = " The lowest category is found at " + position_in_X_axis_for_min_value + " where " + yLabel + " is " + str(yValueArr[minValueIndex]) + "."

            if len(xValueArr) > 3:
                summary6 = position_in_X_axis_for_max_value + " is higher than any other categories with value " + str(yValueArr[maxValueIndex]) + ", " \
                          "followed by " + position_in_X_axis_for_second_max_value + ", and " + position_in_X_axis_for_third_max_value + ". " \
                          "Down to category " + position_in_X_axis_for_min_value + " with the lowest value of " + str(yValueArr[minValueIndex])  + ". "

            if float(random.uniform(0, 1)) > 0.60:
                summaryArray.append(summary2_extrema_max)
                summaryArray.append(summary3_order_2nd_max)
                summaryArray.append(summary4_extrema_min)
            else:
                summaryArray.append(summary6)

            if proportion >= 1.5:
                comparison_rel = " The highest value at " + position_in_X_axis_for_max_value + " is almost " + str(proportion) + " times larger than the minimum value of " + position_in_X_axis_for_min_value + ". "
                summaryArray.append(comparison_rel)
                comparison_rel_with_avg = " The highest value " + str(yValueArr[maxValueIndex]) + " at " + position_in_X_axis_for_max_value + " is almost " + str(max_avg_diff_rel) + " times larger than the average value " + str(avgValueOfAllBars) + ". "
                summaryArray.append(comparison_rel)
                comparison_abs = " The difference between the max " + xLabel + " " + position_in_X_axis_for_max_value + " and min " + xLabel + " " + position_in_X_axis_for_min_value + " is " + str(max_min_diff) + ". "
                summaryArray.append(comparison_abs)

            # print("comparison_rel_with_avg -> " + comparison_rel_with_avg)


            trendsArray = [{},{"7": maxValueIndex, "12": maxValueIndex},
                              {"7": minValueIndex, "12": minValueIndex},{}]
            dataJson = [{xLabel: xVal, yLabel: yVal} for xVal, yVal in zip(cleanXArr, cleanYArr)]
            websiteInput = {"title": title, "xAxis": xLabel, "yAxis": yLabel,
                            "columnType": "two",
                            "graphType": chartType, "summaryType": "baseline", "summary": summaryArray,
                            "trends": trendsArray,
                            "data": dataJson}
            with open(f'{websitePath}/{name}.json', 'w', encoding='utf-8') as websiteFile:
                json.dump(websiteInput, websiteFile, indent=3)
            # oneFile.writelines(''.join(summaryArray)+'\n')



        # run line
        elif (chartType == "line"):
            trendArray = []
            numericXValueArr = []
            for xVal, index in zip(xValueArr, range(len(xValueArr))):
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
            while i < (len(yValueArr)):
                variance1 = float(yValueArr[i]) - float(yValueArr[i - 1])
                if (variance1 > 0):
                    type1 = "negative"
                elif (variance1 < 0):
                    type1 = "positive"
                else:
                    type1 = "neutral"
                trendArray.append(type1)
                i = i + 1
            # iterate through the variances and check for trends
            startIndex = 0
            trendLen = len(trendArray)
            # creates dictionary containing the trend length, direction, start and end indices, and the linear regression of the trend
            significanceRange = round(len(yValueArr) / 8)
            significantTrendCount = 0
            significantTrendArray = []
            for n in range(trendLen):
                currentVal = trendArray[n - 1]
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
                                      "start": startIndex, "end": endIndex, "slope": slope, "intercept":intercept}
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
                    graphTrendArray.append({"1":str(xValueArr.index(startVal)), "12":str(xValueArr.index(endVal))})
                    while (m < significantTrendCount):
                        # append conjunction between significant trends
                        if (direction == "positive"):
                            length = len(similarSynonyms)
                            random_lbl = randint(0, length - 1)
                            synonym = similarSynonyms[random_lbl]
                            conjunction = synonym + ","
                        elif (direction == "negative"):
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
                        graphTrendArray.append({"3":str(xValueArr.index(startVal)), "14":str(xValueArr.index(endVal))})
                        m = m + 1
                # execute here if only 1 significant trend
                else:
                    localTrendSentence1 = "This line chart has an x axis representing " + xLabel + " and a y axis representing " + yLabel + " with a total of " + str(
                        len(yValueArr)) + " data points. The chart has one significant trend."
                    summaryArray.append(localTrendSentence1)
                    graphTrendArray.append({})
                    localTrendSummary = " This trend is " + direction + " which exists from " + startVal + " to " + endVal + "."
                    summaryArray.append(localTrendSummary)
                    graphTrendArray.append({"3":str(xValueArr.index(startVal)), "14":str(xValueArr.index(endVal))})
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


bar_data = "Hashtag|#forthepeople|x|bar_chart Total_number_of_mentions|6961|y|bar_chart Hashtag|#trumpshutdown|x|bar_chart Total_number_of_mentions|3992|y|bar_chart Hashtag|#protectourcare|x|bar_chart Total_number_of_mentions|2810|y|bar_chart Hashtag|#endgunviolence|x|bar_chart Total_number_of_mentions|2342|y|bar_chart Hashtag|#hr8|x|bar_chart Total_number_of_mentions|2206|y|bar_chart Hashtag|#actonclimate|x|bar_chart Total_number_of_mentions|2206|y|bar_chart Hashtag|#endtheshutdown|x|bar_chart Total_number_of_mentions|1769|y|bar_chart Hashtag|#sotu|x|bar_chart Total_number_of_mentions|1767|y|bar_chart Hashtag|#equalityact|x|bar_chart Total_number_of_mentions|1720|y|bar_chart Hashtag|#hr1|x|bar_chart Total_number_of_mentions|1683|y|bar_chart "
group_bar_data = "Movie|Furious_7|0|bar_chart North_America|353.01|1|bar_chart Worldwide|1516.0|2|bar_chart Movie|The_Fate_of_the_Furious|0|bar_chart North_America|226.01|1|bar_chart Worldwide|1236.0|2|bar_chart Movie|Fast_&_Furious_6|0|bar_chart North_America|238.68|1|bar_chart Worldwide|788.7|2|bar_chart Movie|Fast_&_Furious_Presents:_Hobbs_&_Shaw|0|bar_chart North_America|164.34|1|bar_chart Worldwide|721.04|2|bar_chart Movie|Fast_Five|0|bar_chart North_America|209.84|1|bar_chart Worldwide|626.1|2|bar_chart Movie|Fast_and_Furious|0|bar_chart North_America|155.06|1|bar_chart Worldwide|363.2|2|bar_chart Movie|2_Fast_2_Furious|0|bar_chart North_America|127.15|1|bar_chart Worldwide|236.4|2|bar_chart Movie|The_Fast_and_the_Furious|0|bar_chart North_America|144.53|1|bar_chart Worldwide|207.3|2|bar_chart Movie|The_Fast_and_the_Furious:_Tokyo_Drift|0|bar_chart North_America|62.51|1|bar_chart Worldwide|158.5|2|bar_chart "
single_line_data = "Year|2024|x|line_chart GDP_per_capita_in_U.S._dollars|265.58|y|line_chart Year|2023|x|line_chart GDP_per_capita_in_U.S._dollars|270.37|y|line_chart Year|2022|x|line_chart GDP_per_capita_in_U.S._dollars|278.36|y|line_chart Year|2021|x|line_chart GDP_per_capita_in_U.S._dollars|244.0|y|line_chart Year|2020|x|line_chart GDP_per_capita_in_U.S._dollars|243.27|y|line_chart Year|2019|x|line_chart GDP_per_capita_in_U.S._dollars|275.18|y|line_chart Year|2018|x|line_chart GDP_per_capita_in_U.S._dollars|353.17|y|line_chart Year|2017|x|line_chart GDP_per_capita_in_U.S._dollars|273.14|y|line_chart Year|2016|x|line_chart GDP_per_capita_in_U.S._dollars|281.51|y|line_chart Year|2015|x|line_chart GDP_per_capita_in_U.S._dollars|1225.19|y|line_chart Year|2014|x|line_chart GDP_per_capita_in_U.S._dollars|1309.95|y|line_chart "
multi_line_data = "Year|2018|0|line_chart Export|55968.7|1|line_chart Import|108775.3|2|line_chart Year|2017|0|line_chart Export|45622.2|1|line_chart Import|94101.9|2|line_chart Year|2016|0|line_chart Export|48752.7|1|line_chart Import|78531.7|2|line_chart Year|2015|0|line_chart Export|71404.6|1|line_chart Import|88653.6|2|line_chart Year|2014|0|line_chart Export|43256.4|1|line_chart Import|69174.6|2|line_chart Year|2013|0|line_chart Export|91886.1|1|line_chart Import|103475.2|2|line_chart Year|2012|0|line_chart Export|55652.9|1|line_chart Import|72623.0|2|line_chart Year|2011|0|line_chart Export|65749.5|1|line_chart Import|75092.9|2|line_chart Year|2010|0|line_chart Export|48278.3|1|line_chart Import|57152.3|2|line_chart Year|2009|0|line_chart Export|29452.8|1|line_chart Import|44580.8|2|line_chart Year|2008|0|line_chart Export|42525.4|1|line_chart Import|62685.1|2|line_chart Year|2007|0|line_chart Export|38762.0|1|line_chart Import|59961.2|2|line_chart Year|2006|0|line_chart Export|28678.0|1|line_chart Import|42992.7|2|line_chart Year|2005|0|line_chart Export|24045.7|1|line_chart Import|37427.3|2|line_chart "

# summarize(data = bar_data, name = "bar_data", title="Test")
# summarize(data = group_bar_data, name = "group_bar_data", title="Test")
# summarize(data = single_line_data, name = "single_line_data", title="Test")
summarize(data = multi_line_data, name = "multi_line_data", title="Test")