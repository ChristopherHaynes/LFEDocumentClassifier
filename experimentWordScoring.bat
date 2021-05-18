@ECHO OFF
python LFEDocumentClassifier.py -c cnb -we rake -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv
python LFEDocumentClassifier.py -c cnb -we text_rank -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv
python LFEDocumentClassifier.py -c cnb -we word_count -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv

python LFEDocumentClassifier.py -c cnb -we rake -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ure
python LFEDocumentClassifier.py -c cnb -we text_rank -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ure
python LFEDocumentClassifier.py -c cnb -we word_count -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ure
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ure

python LFEDocumentClassifier.py -c cnb -we rake -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Amazon_Hierarchical_Reviews.csv -inam Text -tnam Cat1
python LFEDocumentClassifier.py -c cnb -we text_rank -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Amazon_Hierarchical_Reviews.csv -inam Text -tnam Cat1
python LFEDocumentClassifier.py -c cnb -we word_count -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Amazon_Hierarchical_Reviews.csv -inam Text -tnam Cat1
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Amazon_Hierarchical_Reviews.csv -inam Text -tnam Cat1

python LFEDocumentClassifier.py -c cnb -we rake -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Corona_NLP_tweets.csv -inam OriginalTweet -tnam Sentiment
python LFEDocumentClassifier.py -c cnb -we text_rank -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Corona_NLP_tweets.csv -inam OriginalTweet -tnam Sentiment
python LFEDocumentClassifier.py -c cnb -we word_count -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Corona_NLP_tweets.csv -inam OriginalTweet -tnam Sentiment
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\Corona_NLP_tweets.csv -inam OriginalTweet -tnam Sentiment

python LFEDocumentClassifier.py -c cnb -we rake -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\twitter_mediatype_data.csv -inam text -tnam type
python LFEDocumentClassifier.py -c cnb -we text_rank -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\twitter_mediatype_data.csv -inam text -tnam type
python LFEDocumentClassifier.py -c cnb -we word_count -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\twitter_mediatype_data.csv -inam text -tnam type
python LFEDocumentClassifier.py -c cnb -we tf_idf -st -sw -tr 1 -cv -s -fn finalWordScoringExperiment.csv -ucsv -path C:\\Users\\Chris\\Desktop\\Data\\LFE\\twitter_mediatype_data.csv -inam text -tnam type

PAUSE
