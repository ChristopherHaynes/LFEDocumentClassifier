import copy
import math
import numpy as np

from Parameters.AllThemes import ALL_THEMES_LIST


def convertThemeValueToString(themeValue):
    return ALL_THEMES_LIST[themeValue]


def getAccuracyPercent(predictions, actualResults):
    correctPredictions = 0
    for i in range(len(predictions)):
        if predictions[i] == actualResults[i]:
            correctPredictions = correctPredictions + 1

    return (correctPredictions / len(predictions)) * 100


def getAccuracyVariance(accuracyPerTest):
    averageAccuracy = sum(accuracyPerTest) / len(accuracyPerTest)

    squaredSumDifferences = 0
    for accuracy in accuracyPerTest:
        squaredSumDifferences += math.pow(accuracy - averageAccuracy, 2)

    return squaredSumDifferences / len(accuracyPerTest)


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


def getAverageClassDistribution(themeIndicesPerTest):
    classCount = list(np.zeros(shape=(len(ALL_THEMES_LIST))))
    for themeIndices in themeIndicesPerTest:
        for themeIndex in themeIndices:
            classCount[themeIndex] += 1

    averageClassCount = []
    for count in classCount:
        averageClassCount.append(round(count / len(themeIndicesPerTest), 3))

    return averageClassCount


def getAverageClassProportion(themeIndicesPerTest):
    classCount = list(np.zeros(shape=(len(ALL_THEMES_LIST))))
    totalItemCount = len(themeIndicesPerTest) * len(themeIndicesPerTest[0])
    for themeIndices in themeIndicesPerTest:
        for themeIndex in themeIndices:
            classCount[themeIndex] += 1

    averageClassDistribution = []
    for count in classCount:
        averageClassDistribution.append(round((count / totalItemCount) * 100, 3))

    return averageClassDistribution


def getAverageF1Score(predictions, actualResults):
    precisionRecalls = getPrecisionRecall(predictions, actualResults)
    sumF1 = 0
    for pr in precisionRecalls:
        sumF1 += getF1Score(pr[0], pr[1])
    return sumF1 / len(precisionRecalls)


def getF1Score(precision, recall):
    if precision + recall == 0:
        return 0.0
    return 2 * ((precision * recall) / (precision + recall))


# TODO: Final returned values for average precision and recall appear to always be the same. DON'T TRUST THIS METHOD.
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


def getAveragePrecisionRecallPerClass(precisionRecallsPerTest):
    precisionAverages = list(np.zeros(shape=(len(ALL_THEMES_LIST))))
    recallAverages = list(np.zeros(shape=(len(ALL_THEMES_LIST))))

    for precisionRecallPerClass in precisionRecallsPerTest:
        for i in range(len(precisionRecallPerClass)):
            precisionAverages[i] += precisionRecallPerClass[i][0]
            recallAverages[i] += precisionRecallPerClass[i][1]

    for i in range(len(precisionAverages)):
        precisionAverages[i] = round(precisionAverages[i] / len(precisionRecallsPerTest), 3)
        recallAverages[i] = round(recallAverages[i] / len(precisionRecallsPerTest), 3)

    return precisionAverages, recallAverages


def getPrecisionRecall(predictions, actualResults):
    confusionMatrix = computeConfusionMatrix(predictions, actualResults)
    allPrecisionRecall = []

    for iTheme in range(len(ALL_THEMES_LIST)):
        # True positives divided by true positives and false positives
        precision = confusionMatrix[iTheme, iTheme] / sum(confusionMatrix[iTheme, :]) if sum(confusionMatrix[iTheme, :]) != 0 else 0
        # True positives divided by true positives and false negatives
        recall = confusionMatrix[iTheme, iTheme] / sum(confusionMatrix[:, iTheme]) if sum(confusionMatrix[:, iTheme]) != 0 else 0
        allPrecisionRecall.append([precision, recall])
    return allPrecisionRecall


def computeConfusionMatrix(predictions, actualResults):
    # Generate a confusion matrix of size N_CLASSES * N_CLASSES
    confusionMatrix = np.zeros(shape=(len(ALL_THEMES_LIST), len(ALL_THEMES_LIST)))

    # First dimension is predictions, second dimension is actual classes
    for i in range(len(predictions)):
        confusionMatrix[predictions[i], actualResults[i]] += 1

    return confusionMatrix


# Do ANY of the predictions match the CLASS of the actual themes (non-dependant on order)
def getMultiThemeLooseAccuracy(predictions, actualResults):
    correctPredictions = 0
    for i in range(len(predictions)):
        for prediction in predictions[i]:
            if prediction in actualResults[i]:
                correctPredictions += 1
                break

    return (correctPredictions / len(predictions)) * 100


# What percentage of the predictions match any CLASS of the actual themes (non-dependant on order)
def getMultiThemeMidAccuracy(predictions, actualResults):
    correctPredictions = 0
    for i in range(len(predictions)):
        shortestListLength = len(predictions[i]) if len(predictions[i]) <= len(actualResults[i]) else len(actualResults[i])
        for j in range(shortestListLength):
            if predictions[i][j] in actualResults[i]:
                correctPredictions += 1/shortestListLength

    return (correctPredictions / len(predictions)) * 100


# Does EACH prediction match the CLASS and POSITION of the actual themes (dependant on order)
def getMultiThemePerfectAccuracy(predictions, actualResults):
    correctPredictions = 0
    for i in range(len(predictions)):
        shortestListLength = len(predictions[i]) if len(predictions[i]) <= len(actualResults[i]) else len(actualResults[i])
        for j in range(shortestListLength):
            if predictions[i][j] == actualResults[i][j]:
                correctPredictions += 1/shortestListLength

    return (correctPredictions / len(predictions)) * 100
