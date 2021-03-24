import copy
import numpy as np


from Parameters.AllThemes import ALL_THEMES_LIST


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
    for i in range(len(sortedThemes)):
        fractionalThemes[i][0] = sortedThemes[i][0] / totalSum

    return fractionalThemes


def getF1Score(predictions, actualResults):
    avgPR = getAveragePrecisionRecall(predictions, actualResults)
    return 2 * ((avgPR[0] * avgPR[1]) / (avgPR[0] + avgPR[1]))


# TODO: Final returned values for average precision and recall appear to always be the same. Why?
def getAveragePrecisionRecall(predictions, actualResults):
    confusionMatrix = computeConfusionMatrix(predictions, actualResults)
    sumTruePositives = 0
    sumFalsePositives = 0
    sumFalseNegatives = 0

    for themeIndex in range(len(ALL_THEMES_LIST)):
        sumTruePositives += confusionMatrix[themeIndex, themeIndex]
        sumFalsePositives += (sum(confusionMatrix[themeIndex, :]) - confusionMatrix[themeIndex, themeIndex])
        sumFalseNegatives += (sum(confusionMatrix[:, themeIndex]) - confusionMatrix[themeIndex, themeIndex])

    averagePrecision = sumTruePositives / (sumTruePositives + sumFalsePositives)
    averageRecall = sumTruePositives / (sumTruePositives + sumFalseNegatives)
    return [averagePrecision, averageRecall]


# TODO: In this method it is possible to end up dividing by zero resulting in NaNs, needs resolving
def getPrecisionRecall(predictions, actualResults):
    confusionMatrix = computeConfusionMatrix(predictions, actualResults)
    allPrecisionRecall = []

    for themeIndex in range(len(ALL_THEMES_LIST)):
        # True positives divided by true positives and false positives
        precision = confusionMatrix[themeIndex, themeIndex] / sum(confusionMatrix[themeIndex, :])
        # True positives divided by true positives and false negatives
        recall = confusionMatrix[themeIndex, themeIndex] / sum(confusionMatrix[:, themeIndex])
        allPrecisionRecall.append([precision, recall])
    return allPrecisionRecall


def computeConfusionMatrix(predictions, actualResults):
    # Generate a confusion matrix of size N_CLASSES * N_CLASSES
    confusionMatrix = np.zeros(shape=(len(ALL_THEMES_LIST), len(ALL_THEMES_LIST)))

    # First dimension is predictions, second dimension is actual classes
    for i in range(len(predictions)):
        confusionMatrix[predictions[i], actualResults[i]] += 1

    return confusionMatrix
