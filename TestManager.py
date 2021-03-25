from StatisticsGenerator import *
from Parameters import *


def runTests(classifier, printProgress=False):
    results = []
    print("Starting " + str(EPOCHS) + " epochs using " + classifier.name + " classifier.")
    for epoch in range(0, EPOCHS):
        classifier.splitTestTrainData()
        classifier.train()
        results.append(classifier.classifySingleClass())
        if printProgress:
            print("Completed epoch " + str(epoch + 1))
    return results


# TODO: ALL THIS IS UNTESTED!!!!!!!!!
def getTestStats(results,
                 usePrecisionRecall=True,
                 useF1Score=True,
                 useMinMax=True,
                 useAverageF1=True):
    testStats = dict()

    accuracyPercents = []
    precisionRecalls = []
    F1Scores = []
    for result in results:
        accuracyPercents.append(getAccuracyPercent(result[0], result[1]))
        if usePrecisionRecall:
            precisionRecalls.append(getAveragePrecisionRecall(result[0], result[1]))
        if useF1Score:
            F1Scores.append(getAverageF1Score(result[0], result[1]))

    testStats["Epochs"] = EPOCHS
    testStats["AverageAccuracy"] = sum(accuracyPercents) / len(accuracyPercents)
    testStats["AccuracyList"] = accuracyPercents
    if usePrecisionRecall:
        testStats["Precision"] = precisionRecalls[0, :]  # TODO: ARE THESE SLICES THE CORRECT WAY ROUND? CHECK IT!!!
        testStats["Recall"] = precisionRecalls[1, :]  # TODO: ARE THESE SLICES THE CORRECT WAY ROUND? CHECK IT!!!
    if useF1Score:
        testStats["F1List"] = F1Scores
    if useMinMax:
        testStats["MaxAccuracy"] = max(accuracyPercents)
        testStats["MinAccuracy"] = min(accuracyPercents)
    if useAverageF1:
        testStats["AverageF1"] = sum(precisionRecalls) / len(precisionRecalls)

    return testStats
