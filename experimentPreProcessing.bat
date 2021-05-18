@ECHO OFF
REM python LFEDocumentClassifier.py -c cnb -we tf_idf -tr 1 -cv -s -fn finalPreProcessingExperiment.csv
REM python LFEDocumentClassifier.py -c cnb -we tf_idf -st -tr 1 -cv -s -fn finalPreProcessingExperiment.csv
REM python LFEDocumentClassifier.py -c cnb -we tf_idf -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv
REM python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv

REM python LFEDocumentClassifier.py -c cnb -we tf_idf -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ure
REM python LFEDocumentClassifier.py -c cnb -we tf_idf -st -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ure
REM python LFEDocumentClassifier.py -c cnb -we tf_idf -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ure
REM python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ure

python LFEDocumentClassifier.py -c cnb -we tf_idf -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Amazon_Hierarchical_Reviews.csv -inam Text -tnam Cat1
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Amazon_Hierarchical_Reviews.csv -inam Text -tnam Cat1
python LFEDocumentClassifier.py -c cnb -we tf_idf -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Amazon_Hierarchical_Reviews.csv -inam Text -tnam Cat1
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Amazon_Hierarchical_Reviews.csv -inam Text -tnam Cat1

python LFEDocumentClassifier.py -c cnb -we tf_idf -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Corona_NLP_tweets.csv -inam OriginalTweet -tnam Sentiment
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Corona_NLP_tweets.csv -inam OriginalTweet -tnam Sentiment
python LFEDocumentClassifier.py -c cnb -we tf_idf -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Corona_NLP_tweets.csv -inam OriginalTweet -tnam Sentiment
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Corona_NLP_tweets.csv -inam OriginalTweet -tnam Sentiment

python LFEDocumentClassifier.py -c cnb -we tf_idf -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\twitter_mediatype_data.csv -inam text -tnam type
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\twitter_mediatype_data.csv -inam text -tnam type
python LFEDocumentClassifier.py -c cnb -we tf_idf -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\twitter_mediatype_data.csv -inam text -tnam type
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -tr 1 -cv -s -fn finalPreProcessingExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\twitter_mediatype_data.csv -inam text -tnam type

PAUSE
