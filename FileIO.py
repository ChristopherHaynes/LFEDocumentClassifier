from pathlib import Path
import os
import csv

from Parameters import *

headers = ["ClassifierName", "WordEmbeddingMethod", "StopWordsRemoved", "WordsStemmed", "Epochs", "AverageAccuracy",
           "AccuracyVariance", "MaxAccuracy", "MinAccuracy", "", "CLASS STATS:"] + ALL_THEMES_LIST


def writeStatsToFile(testStats, fileName='testStats.csv'):
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
        for row in range(1, 5):
            csvWriter.writerow(generateRowData(testStats, row))
        csvWriter.writerow("")


def generateRowData(testStats, rowID):
    if rowID == 1:
        rowData = [convertClassifierAbbreviation(), convertWordEmbeddingAbbreviation(), str(REMOVE_STOPWORDS),
                   str(STEM_TEXT)]

        for i in range(4, 9):
            rowData.append(testStats[headers[i]])
    else:
        rowData = ["", "", "", "", "", "", "", "", ""]

    rowData += ["", convertRowIDToHeader(rowID)]
    listData = testStats[convertRowIDToHeader(rowID)]
    for value in listData:
        rowData.append(value)

    return rowData


def convertWordEmbeddingAbbreviation():
    if KEYWORD_ID_METHOD == "rake":
        return "Rapid Automatic Keyword Extraction"
    elif KEYWORD_ID_METHOD == "text_rank":
        return "Text Rank"
    elif KEYWORD_ID_METHOD == "word_count":
        return "Word Count"
    elif KEYWORD_ID_METHOD == "tf_idf":
        return "Term Frequency Inverse Document Frequency"
    else:
        return "UNDEFINED"


def convertClassifierAbbreviation():
    if CLASSIFIER_NAME == "knn":
        return "K Nearest Neighbors"
    elif CLASSIFIER_NAME == "cnb":
        return "Compliment Naive Bayes"
    else:
        return "UNDEFINED"


def convertRowIDToHeader(rowID):
    if rowID == 1:
        return "PrecisionAverages"
    elif rowID == 2:
        return "RecallAverages"
    elif rowID == 3:
        return "AverageF1"
    elif rowID == 4:
        return "AverageClassSize"
    else:
        return "ERROR"
