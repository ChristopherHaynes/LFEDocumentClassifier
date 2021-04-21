from StatisticsGenerator import *


def runTests(classifier, epochs, useMultiLabelClassification, crossValidate, folds=5, printProgress=False):
    results = []

    if crossValidate:
        print("Starting cross validation testing using " + classifier.name + " classifier.")
        results = classifier.crossValidate(folds)
    else:
        print("Starting " + str(epochs) + " epochs using " + classifier.name + " classifier.")
        for epoch in range(0, epochs):
            classifier.splitTestTrainData()
            classifier.train()
            if useMultiLabelClassification:
                results.append(classifier.classifyMultiClass())
            else:
                results.append(classifier.classifySingleClass())
            if printProgress:
                print("Completed epoch " + str(epoch + 1))
    return results


def getMultiLabelTestStats(results, epochs):
    testStats = dict()

    accuracyPercentsLoose = []
    accuracyPercentsMid = []
    accuracyPercentsPerfect = []
    for result in results:
        accuracyPercentsLoose.append(getMultiThemeLooseAccuracy(result[0], result[1]))
        accuracyPercentsMid.append(getMultiThemeMidAccuracy(result[0], result[1]))
        accuracyPercentsPerfect.append(getMultiThemePerfectAccuracy(result[0], result[1]))

    # Per TEST stats
    testStats["Epochs"] = epochs
    testStats["LooseAverageAccuracy"] = round(sum(accuracyPercentsLoose) / len(accuracyPercentsLoose), 3)
    testStats["LooseAccuracyVariance"] = round(getAccuracyVariance(accuracyPercentsLoose), 3)
    testStats["LooseMaxAccuracy"] = round(max(accuracyPercentsLoose), 3)
    testStats["LooseMinAccuracy"] = round(min(accuracyPercentsLoose), 3)

    testStats["MidAverageAccuracy"] = round(sum(accuracyPercentsMid) / len(accuracyPercentsMid), 3)
    testStats["MidAccuracyVariance"] = round(getAccuracyVariance(accuracyPercentsMid), 3)
    testStats["MidMaxAccuracy"] = round(max(accuracyPercentsMid), 3)
    testStats["MidMinAccuracy"] = round(min(accuracyPercentsMid), 3)

    testStats["PerfectAverageAccuracy"] = round(sum(accuracyPercentsPerfect) / len(accuracyPercentsPerfect), 3)
    testStats["PerfectAccuracyVariance"] = round(getAccuracyVariance(accuracyPercentsPerfect), 3)
    testStats["PerfectMaxAccuracy"] = round(max(accuracyPercentsPerfect), 3)
    testStats["PerfectMinAccuracy"] = round(min(accuracyPercentsPerfect), 3)

    return testStats


def getTestStats(results, epochs):
    testStats = dict()

    accuracyPercents = []
    precisionRecalls = []
    for result in results:
        accuracyPercents.append(getAccuracyPercent(result[0], result[1]))
        precisionRecalls.append(getPrecisionRecall(result[0], result[1]))

    precisionAverages, recallAverages = getAveragePrecisionRecallPerClass(precisionRecalls)
    averageF1 = []
    for i in range(len(precisionAverages)):
        averageF1.append(round(getF1Score(precisionAverages[i], recallAverages[i]), 3))

    # Per TEST stats
    testStats["Epochs"] = epochs
    testStats["AverageAccuracy"] = round(sum(accuracyPercents) / len(accuracyPercents), 3)
    testStats["AccuracyVariance"] = round(getAccuracyVariance(accuracyPercents), 3)
    testStats["MaxAccuracy"] = round(max(accuracyPercents), 3)
    testStats["MinAccuracy"] = round(min(accuracyPercents), 3)
    # Per CLASS stats
    testStats["PrecisionAverages"] = precisionAverages
    testStats["RecallAverages"] = recallAverages
    testStats["AverageF1"] = averageF1
    testStats["PredictionAverageClassDistribution"] = getAverageClassProportion([x[0] for x in results])
    testStats["ActualAverageClassDistribution"] = getAverageClassProportion([x[1] for x in results])

    return testStats
