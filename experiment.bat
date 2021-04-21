@ECHO OFF
python LFEDocumentClassifier.py -c knn -kn 15 -tr 1 -cv -s -fn knnCVExperiments.csv
python LFEDocumentClassifier.py -c knn -kn 30 -tr 1 -cv -s -fn knnCVExperiments.csv
python LFEDocumentClassifier.py -c knn -kn 60 -tr 1 -cv -s -fn knnCVExperiments.csv
python LFEDocumentClassifier.py -c cnb -tr 1 -cv -s -fn cnbCVExperiments.csv
python LFEDocumentClassifier.py -c cnb -sw -tr 1 -cv -s -fn cnbCVExperiments.csv
python LFEDocumentClassifier.py -c cnb -st -tr 1 -cv -s -fn cnbCVExperiments.csv
python LFEDocumentClassifier.py -c cnb -st -sw -tr 1 -cv -s -fn cnbCVExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 32 -ne 25 -tr 1 -cv -s -fn nnCVExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 128 -ne 25 -tr 1 -cv -s -fn nnCVExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 32 -ne 25 -st -sw -tr 1 -cv -s -fn nnCVExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 128 -ne 25 -st -sw -tr 1 -cv -s -fn nnCVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -tr 1 -cv -s -fn svmCVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -sc -tr 1 -cv -s -fn svmCVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -sw -st -tr 1 -cv -s -fn svmCVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -sc -sw -st -tr 1 -cv -s -fn svmCVExperiments.csv
PAUSE
