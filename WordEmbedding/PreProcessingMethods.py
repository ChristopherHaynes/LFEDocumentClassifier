import nltk as nltk
import stanza
import string
import copy
import spacy

try:
    nlp = spacy.load('en_core_web_sm')
    stanza.download('en')
except:
    print("CONNECTION ERROR WARNING!")
    pass


def stanfordNLPPreProcessor(themePairs):
    themePairsClone = copy.deepcopy(themePairs)

    stanfordNLP = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma', use_gpu=True)

    for i in range(len(themePairsClone)):
        doc = stanfordNLP(themePairsClone[i][0])
        newSentences = []
        for sentence in doc.sentences:
            words = [word.lemma for word in sentence.words]
            newWords = []
            for word in words:
                if word not in string.punctuation:
                    newWords.append(word)
            newSentences.append(newWords)
        themePairsClone[i][0] = newSentences
        print(i)  # TESTING SPEED OF OPERATIONS

    return themePairsClone


def removeStopWords(themePairs):
    stopWords = set(nltk.corpus.stopwords.words('english'))
    themePairsClone = copy.deepcopy(themePairs)

    for i in range(len(themePairsClone)):
        filteredText = []
        for sentence in themePairsClone[i][0]:
            newSentence = []
            for word in sentence:
                if str(word) not in stopWords:
                    newSentence.append(str(word))
            filteredText.append(newSentence)
        themePairsClone[i][0] = filteredText
    return themePairsClone


def splitOnSentenceAndWords(themePairs):
    themePairsClone = copy.deepcopy(themePairs)
    for pair in themePairsClone:
        wordAndSentence = []
        sentences = nltk.sent_tokenize(pair[0])
        for sentence in sentences:
            words = nltk.word_tokenize(sentence)
            wordAndSentence.append([word for word in words if word.isalnum()])
        pair[0] = wordAndSentence
    return themePairsClone


def DEPRECIATED_stemText(themePairs):
    stemmer = nltk.stem.PorterStemmer()
    themePairsClone = copy.deepcopy(themePairs)

    for i in range(len(themePairsClone)):
        newText = []
        for sentence in themePairsClone[i][0]:
            newSentence = []
            for word in sentence:
                newSentence.append(stemmer.stem(str(word)))
            newText.append(newSentence)
        themePairsClone[i][0] = newText
    return themePairsClone


def DEPRECIATED_splitOnSentenceAndWords(themePairs):
    themePairsClone = copy.deepcopy(themePairs)
    for i in range(len(themePairsClone)):
        doc = nlp(themePairsClone[i][0])
        sentences = []
        for sent in doc.sents:
            selectedWords = []
            for token in sent:
                if str(token) not in string.punctuation:
                    selectedWords.append(token)
            sentences.append(selectedWords)
        themePairsClone[i][0] = sentences
    return themePairsClone
