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


def getAverageClassDistribution(actualResultsPerTest):
    classCount = list(np.zeros(shape=(len(ALL_THEMES_LIST))))
    for actualResults in actualResultsPerTest:
        for result in actualResults:
            classCount[result] += 1

    averageClassCount = []
    for count in classCount:
        averageClassCount.append(round(count / len(actualResultsPerTest), 3))

    return averageClassCount


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