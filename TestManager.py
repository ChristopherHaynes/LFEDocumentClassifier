from StatisticsGenerator import *
from Parameters import *


# TODO: Add multi class classifier branching options
def runTests(classifier, printProgress=False):
    results = []
    print("Starting " + str(EPOCHS) + " epochs using " + classifier.name + " classifier.")
    for epoch in range(0, EPOCHS):
        classifier.splitTestTrainData()
        classifier.train()
        if USE_MULTI_LABEL_CLASSIFICATION:
            results.append(classifier.classifyMultiClass())
        else:
            results.append(classifier.classifySingleClass())
        if printProgress:
            print("Completed epoch " + str(epoch + 1))
    return results


def getMultiLabelTestStats(results):
    testStats = dict()

    accuracyPercents = []
    for result in results:
        accuracyPercents.append(getMultiThemeAccuracy(result[0], result[1]))

    # Per TEST stats
    testStats["Epochs"] = EPOCHS
    testStats["AverageAccuracy"] = round(sum(accuracyPercents) / len(accuracyPercents), 3)
    testStats["AccuracyVariance"] = round(getAccuracyVariance(accuracyPercents), 3)
    testStats["MaxAccuracy"] = round(max(accuracyPercents), 3)
    testStats["MinAccuracy"] = round(min(accuracyPercents), 3)

    return testStats


def getTestStats(results):
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
    testStats["Epochs"] = EPOCHS
    testStats["AverageAccuracy"] = round(sum(accuracyPercents) / len(accuracyPercents), 3)
    testStats["AccuracyVariance"] = round(getAccuracyVariance(accuracyPercents), 3)
    testStats["MaxAccuracy"] = round(max(accuracyPercents), 3)
    testStats["MinAccuracy"] = round(min(accuracyPercents), 3)
    # Per CLASS stats
    testStats["PrecisionAverages"] = precisionAverages
    testStats["RecallAverages"] = recallAverages
    testStats["AverageF1"] = averageF1
    testStats["AverageClassSize"] = getAverageClassDistribution([x[1] for x in results])

    return testStats
