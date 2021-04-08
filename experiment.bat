@ECHO OFF
python LFEDocumentClassifier.py -c knn -kn 15 -tr 3 -e 200 -s -fn knnExperiments.csv
python LFEDocumentClassifier.py -c knn -kn 30 -tr 3 -e 200 -s -fn knnExperiments.csv
python LFEDocumentClassifier.py -c knn -kn 60 -tr 3 -e 200 -s -fn knnExperiments.csv
python LFEDocumentClassifier.py -c cnb -tr 3 -e 200 -s -fn cnbExperiments.csv
python LFEDocumentClassifier.py -c cnb -sw -tr 3 -e 200 -s -fn cnbExperiments.csv
python LFEDocumentClassifier.py -c cnb -st -tr 3 -e 200 -s -fn cnbExperiments.csv
python LFEDocumentClassifier.py -c cnb -st -sw -tr 3 -e 200 -s -fn cnbExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 32 -ne 25 -tr 3 -e 50 -s -fn nnExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 128 -ne 25 -tr 3 -e 50 -s -fn nnExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 32 -ne 25 -st -sw -tr 3 -e 50 -s -fn nnExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 128 -ne 25 -st -sw -tr 3 -e 50 -s -fn nnExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -tr 3 -e 25 -s -fn svmExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -sc -tr 3 -e 25 -s -fn svmExperiments.csv
python LFEDocumentClassifier.py -c svm -sk poly -tr 3 -e 25 -s -fn svmExperiments.csv
python LFEDocumentClassifier.py -c svm -sk poly -sc -tr 3 -e 25 -s -fn svmExperiments.csv
PAUSE
