from pathlib import Path
import os
import csv

from Parameters.AllThemes import ALL_THEMES_LIST

headers = ["ClassifierName", "WordEmbeddingMethod", "StopWordsRemoved", "WordsStemmed", "Epochs", "AverageAccuracy",
           "AccuracyVariance", "MaxAccuracy", "MinAccuracy", "", "CLASS STATS:"] + ALL_THEMES_LIST


def writeStatsToFile(testStats, fileName, classifierAbbreviation, wordEmbeddingMethod, removeStopwords, stemText):
    # If an "Output" directory doesn't exist in the working directory, then create one
    rootPath = Path(__file__).parent
    outputDirPath = (rootPath / "./Output").resolve()
    try:
        os.mkdir(outputDirPath)
    except OSError:
        pass

    # If a results file already exists then open it, otherwise create a new file and write the headers
    filePath = (rootPath / "./Output/" / fileName).resolve()
    isExistingFile = filePath.is_file()
    with open(filePath, 'a', newline='') as file:
        csvWriter = csv.writer(file)
        if not isExistingFile:
            csvWriter.writerow(headers)
        for row in range(1, 6):
            csvWriter.writerow(generateRowData(testStats, row, classifierAbbreviation, wordEmbeddingMethod, removeStopwords, stemText))
        csvWriter.writerow("")


def generateRowData(testStats, rowID, classifierAbbreviation, wordEmbeddingMethod, removeStopwords, stemText):
    if rowID == 1:
        rowData = [convertClassifierAbbreviation(classifierAbbreviation),
                   convertWordEmbeddingAbbreviation(wordEmbeddingMethod),
                   str(removeStopwords),
                   str(stemText)]

        for i in range(4, 9):
            rowData.append(testStats[headers[i]])
    else:
        rowData = ["", "", "", "", "", "", "", "", ""]

    rowData += ["", convertRowIDToHeader(rowID)]
    listData = testStats[convertRowIDToHeader(rowID)]
    for value in listData:
        rowData.append(value)

    return rowData


def writeDictionaryToCSV(testStats, fileName, classifierAbbreviation, wordEmbeddingMethod, removeStopwords, stemText):
    # If an "Output" directory doesn't exist in the working directory, then create one
    rootPath = Path(__file__).parent
    outputDirPath = (rootPath / "./Output").resolve()
    try:
        os.mkdir(outputDirPath)
    except OSError:
        pass

    # Generate headers from fixed information and dict keys
    generatedHeader = ["ClassifierName", "WordEmbeddingMethod", "StopWordsRemoved", "WordsStemmed"] + list(testStats.keys())

    # Add the general information to the dict
    testStats["ClassifierName"] = convertClassifierAbbreviation(classifierAbbreviation)
    testStats["WordEmbeddingMethod"] = convertWordEmbeddingAbbreviation(wordEmbeddingMethod)
    testStats["StopWordsRemoved"] = str(removeStopwords)
    testStats["WordsStemmed"] = str(stemText)

    # If a results file already exists then open it, otherwise create a new file and write the headers
    filePath = (rootPath / "./Output/" / fileName).resolve()
    isExistingFile = filePath.is_file()
    with open(filePath, 'a', newline='') as file:
        csvWriter = csv.DictWriter(file, fieldnames=generatedHeader)
        if not isExistingFile:
            csvWriter.writeheader()
        csvWriter.writerow(testStats)


def convertWordEmbeddingAbbreviation(wordEmbeddingMethod):
    if wordEmbeddingMethod == "rake":
        return "Rapid Automatic Keyword Extraction"
    elif wordEmbeddingMethod == "text_rank":
        return "Text Rank"
    elif wordEmbeddingMethod == "word_count":
        return "Word Count"
    elif wordEmbeddingMethod == "tf_idf":
        return "Term Frequency Inverse Document Frequency"
    else:
        return "UNDEFINED - " + wordEmbeddingMethod


def convertClassifierAbbreviation(classifierAbbreviation):
    if classifierAbbreviation == "knn":
        return "K Nearest Neighbors"
    elif classifierAbbreviation == "cnb":
        return "Compliment Naive Bayes"
    elif classifierAbbreviation == "nn":
        return "Sequential Neural Network"
    elif classifierAbbreviation == "svm":
        return "Support Vector Machine"
    else:
        return "UNDEFINED - " + classifierAbbreviation


def convertRowIDToHeader(rowID):
    if rowID == 1:
        return "PrecisionAverages"
    elif rowID == 2:
        return "RecallAverages"
    elif rowID == 3:
        return "AverageF1"
    elif rowID == 4:
        return "PredictionAverageClassDistribution"
    elif rowID == 5:
        return "ActualAverageClassDistribution"
    else:
        return "ERROR"
