from nltk.corpus import reuters
import string


def getReutersFeatureClassPairs():
    # List of tuples where [<text>, <class>]
    themePairs = []

    # List of all used categories
    categories = []

    #TEST DICT
    catCount = dict()

    # Iterate through all the files IDs
    for fileID in reuters.fileids():
        # Only consider the single-class items
        classification = reuters.categories(fileID)
        if len(classification) > 1:
            continue
        else:
            themePairs.append([reuters.raw(fileID), classification])
            if classification[0] not in catCount.keys():
                catCount[classification[0]] = 1
            else:
                catCount[classification[0]] += 1
            if classification[0] not in categories:
                categories.append(classification[0])

    # Clean the data
    themePairs = removeNumericCharactersFromText(themePairs)
    themePairs = removeSingleLettersFromText(themePairs)
    themePairs = removeExtraSpacesFromText(themePairs)

    test = [[key, value] for key, value in catCount.items()]
    test.sort(key=lambda pair: pair[1])

    return themePairs, categories


def removeNumericCharactersFromText(themePairs):
    for pair in themePairs:
        rawText = pair[0]
        newText = ''

        for i in range(0, len(rawText)):
            character = rawText[i]

            if character.isnumeric():
                continue

            # Check the next character for being a decimal point or other numeric punctuation
            if character in string.punctuation and i + 1 < len(rawText) and rawText[i + 1].isnumeric():
                continue

            newText = newText + character
        pair[0] = newText
    return themePairs


def removeSingleLettersFromText(themePairs):
    for pair in themePairs:
        rawText = pair[0]
        newText = ''

        for i in range(0, len(rawText)):
            # If the character is a valid letter and surrounded by whitespace then remove it
            if rawText[i] in string.ascii_letters and i + 1 < len(rawText):
                if rawText[i - 1].isspace() and rawText[i + 1].isspace():
                    continue
            newText = newText + rawText[i]
        pair[0] = newText
    return themePairs


def removeExtraSpacesFromText(themePairs):
    punctuationWithoutPrecedingSpaces = [',', '.', '!', '?']
    for pair in themePairs:
        rawText = pair[0]
        newText = ''

        for i in range(0, len(rawText)):
            if rawText[i].isspace() and i + 1 < len(rawText):
                if rawText[i + 1].isspace() or rawText[i + 1] in punctuationWithoutPrecedingSpaces:
                    continue
            newText = newText + rawText[i]
        pair[0] = newText
    return themePairs