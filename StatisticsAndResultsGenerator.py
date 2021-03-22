import copy

from constants.ALL_THEMES_LIST import ALL_THEMES_LIST


def convertThemeValueToString(themeValue):
    return ALL_THEMES_LIST[themeValue]


def getPercentageCorrect(predictions, actualResults):
    correctPredictions = 0
    for i in range(len(predictions)):
        if predictions[i] == actualResults[i]:
            correctPredictions = correctPredictions + 1

    return (correctPredictions / len(predictions)) * 100


def getSortedAggregatedThemes(numericThemesList, descending=True):
    aggregatedThemes = dict()

    for numericTheme in numericThemesList:
        theme = convertThemeValueToString(numericTheme)
        if theme not in aggregatedThemes.keys():
            aggregatedThemes[theme] = 1
        else:
            aggregatedThemes[theme] = aggregatedThemes[theme] + 1

    sortedThemes = [[value, key] for key, value in aggregatedThemes.items()]
    sortedThemes.sort(key=lambda sortResult: sortResult[0], reverse=descending)
    return sortedThemes


def getFractionOfAllThemes(numericThemesList):
    sortedThemes = getSortedAggregatedThemes(numericThemesList)

    totalSum = 0
    for sortedTheme in sortedThemes:
        totalSum = totalSum + sortedTheme[0]

    fractionalThemes = copy.deepcopy(sortedThemes)
    for i in range (len(sortedThemes)):
        fractionalThemes[i][0] = sortedThemes[i][0] / totalSum

    return fractionalThemes
