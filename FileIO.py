from pathlib import Path
import os
import csv

from StatisticsGenerator import *


def writeResultsToFile(results):
    # If an "Output" directory doesn't exist in the working directory, then create one
    outputDirPath = '/Output'
    try:
        os.mkdir(outputDirPath)
    except OSError:
        print('Creation of output directory failed!')

    # TODO: Continue writing csv output, use dictionary writer to work with variable columns (see TestManager)
    # If a results file already exists then open it, otherwise create a new file and write the headers
    filePath = Path('/Output/results.csv')
    with open('results.csv', 'a', newline='') as file:
        if not filePath.is_file():
            pass
        else:
            pass


