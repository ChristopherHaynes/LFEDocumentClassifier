import pandas as pd
import nltk as nltk
import spacy
import string

nlp = spacy.load('en_core_web_sm')
#nltk.download('all')

class PreProcessor:
        def __init__(self, dataFile):
            self.rawDataFile = dataFile
            self.themes = []
            
                 
        def extractThemePairs(self):
            print("Extracting theme pairs.")
            fullTexts = pd.DataFrame(self.rawDataFile, columns= ['excellenceText'])
            themes = pd.DataFrame(self.rawDataFile, columns= ['themeExcellence'])
            self.themePairs = []

            for i in range(0, len(themes.index) - 1):
                fullText = fullTexts.loc[i].max()
                theme = themes.loc[i].max()   
                if isinstance(fullText, str) and len(fullText) > 0 and isinstance(theme, str) and len(theme) > 0:
                    self.themePairs.append([fullText.lower(), theme])
            
            self.themePairs = self.splitOnSentenceAndWords(self.themePairs)
            self.themePairs = self.removeStopWords(self.themePairs)           
            self.themePairs = self.stemText(self.themePairs)  
                              
            self.getFirstThemes()

        def splitOnSentenceAndWords(self, themePairs):
            print("Spliting sentences and words.")
            for i in range(len(themePairs)):
                doc = nlp(themePairs[i][0])

                sentences = []
                for sent in doc.sents:
                    selectedWords = []
                    for token in sent:
                        if str(token) not in string.punctuation:
                            selectedWords.append(token)
                    sentences.append(selectedWords)
                themePairs[i][0] = sentences
            return themePairs


        def stemText(self, themePairs):
            print("Stemming.")
            stemmer = nltk.stem.PorterStemmer()  

            for i in range(len(themePairs)):
                newText = []
                for sentence in themePairs[i][0]:
                    newSentence = []
                    for word in sentence:
                        newSentence.append(stemmer.stem(str(word)))
                    newText.append(newSentence)
                themePairs[i][0] = newText
            return themePairs

        def removeStopWords(self, themePairs):
            print("Removing stop words.")
            stopWords = set(nltk.corpus.stopwords.words('english'))

            for i in range(len(themePairs)):
                filteredText = []
                for sentence in themePairs[i][0]:
                    newSentence = []
                    for word in sentence:
                        if str(word) not in stopWords:
                            newSentence.append(str(word))
                    filteredText.append(newSentence)
                themePairs[i][0] = filteredText
            return themePairs                    

        def getFirstThemes(self):
            print("Cleaning themes and selecting first theme.")
            for pair in self.themePairs:
                if type(pair[1]) is str:
                    for i in range(0, len(pair[1]) - 1):
                        if pair[1][i] is '\n':
                            pair[1] = pair[1][:i].lower().strip()
                            if pair[1] not in self.themes:
                                self.themes.append(pair[1])
                            break
                else:
                    self.themePairs[i].remove

