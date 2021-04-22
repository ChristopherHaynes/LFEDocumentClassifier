@ECHO OFF
python LFEDocumentClassifier.py -c knn -kn 15 -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c knn -kn 15 -sw -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c knn -kn 15 -st -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c knn -kn 15 -st -sw -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c cnb -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c cnb -sw -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c cnb -st -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c cnb -st -sw -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 32 -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 32 -sw -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 32 -st -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c nn -nb 32 -st -sw -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -sw -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -st -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -sw -st -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -cb -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -cb -sw -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -cb -st -tr 1 -cv -s -fn CVExperiments.csv
python LFEDocumentClassifier.py -c svm -sk rbf -cb -sw -st -tr 1 -cv -s -fn CVExperiments.csv
PAUSE
