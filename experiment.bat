@ECHO OFF
REM python LFEDocumentClassifier.py -c knn -kn 15 -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
REM python LFEDocumentClassifier.py -c knn -kn 15 -st -sw -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
REM python LFEDocumentClassifier.py -c cnb -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
REM python LFEDocumentClassifier.py -c cnb -st -sw -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
REM python LFEDocumentClassifier.py -c nn -nb 32 -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
REM (COME BACK TO THIS ONE!) python LFEDocumentClassifier.py -c nn -nb 32 -st -sw -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
REM python LFEDocumentClassifier.py -c svm -sk rbf -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
REM python LFEDocumentClassifier.py -c svm -sk rbf -sw -st -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
REM python LFEDocumentClassifier.py -c svm -sk rbf -sc -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
python LFEDocumentClassifier.py -c svm -sk rbf -sc -sw -st -tr 1 -cv -s -fn CVExperimentsREUTERS.csv
PAUSE
