import pandas as pd
import string
from constants.ALL_THEMES_LIST import ALL_THEMES_LIST


class PreProcessor:
    def __init__(self, dataFile, themePairs):
        self.rawDataFile = dataFile
        self.themePairs = themePairs  # List of tuples, first item is the text (features), second item is the theme (categories)
        self.themesCount = dict()  # Key is theme, value is number of occurrences
        self.primaryThemesCount = dict()  # Key is theme, value is the number of occurrences as the first theme
        self.unclassifiedThemes = []  # TESTING - Holding list for any strings that don't match approved themes
        self.totalEntries = 0  # TESTING - Keep track of the total original number of entries
        self.totalCulledEntries = 0  # TESTING - Keep track of how many entries are invalid and removed

        # Generate the theme pair list from the data file and discard any invalid entries
        self.extractThemePairs()

        # Convert the raw theme string into a list of strings in the theme pairs list
        self.convertThemesToList()

        # Remove any further entries which are now empty or invalid (lacking in either valid feature text or category)
        self.cullEmptyEntries()

        # Process the counts of all themes and primary themes for each pair in the theme pairs list
        self.getThemesCounts()

    def cleanText(self, removeNumeric=True, removeSingleLetters=True, removeKeywords=True, removeExtraSpaces=True):
        if removeNumeric:
            self.removeNumericCharactersFromText()
        if removeSingleLetters:
            self.removeSingleLettersFromText()
        if removeKeywords:
            self.removeKeywordsFromText()
        if removeExtraSpaces:
            self.removeExtraSpacesFromText()

    def extractThemePairs(self):
        fullTexts = pd.DataFrame(self.rawDataFile, columns=['excellenceText'])
        themesDataFrame = pd.DataFrame(self.rawDataFile, columns=['themeExcellence'])
        self.totalEntries = len(themesDataFrame.index)

        for i in range(0, len(themesDataFrame.index) - 1):
            fullText = fullTexts.loc[i].max()
            theme = themesDataFrame.loc[i].max()
            if isinstance(fullText, str) and len(fullText) > 0 and not fullText.lower() == 'x' \
                    and isinstance(theme, str) and len(theme) > 0 and not theme.lower() == 'x':
                self.themePairs.append([fullText.lower(), theme])
            else:
                self.totalCulledEntries = self.totalCulledEntries + 1

    def convertThemesToList(self):
        for pair in self.themePairs:
            themeList = []
            lastSplitIndex = -1
            themeString = pair[1].lower().strip(' ,.()').replace('\n', ' ')

            if type(themeString) is str and len(themeString) > 0:
                for i in range(0, len(themeString)):
                    splitTheme = themeString[lastSplitIndex + 1:i + 1].strip(' ,.()')

                    # If the split theme is a valid theme and not already recorded for this entry, add it to the list
                    if splitTheme in ALL_THEMES_LIST:
                        if splitTheme not in themeList:
                            themeList.append(splitTheme)

                        # If the "ith" character is a space increase the index by an extra 1
                        if themeString[i] == ' ':
                            lastSplitIndex = i + 1
                        else:
                            lastSplitIndex = i

                # TESTING - Catch all other themes
                if len(themeList) == 0 or len(themeString) - lastSplitIndex > 1:
                    unclassifiedTheme = themeString[lastSplitIndex + 1:i + 1]
                    if unclassifiedTheme not in self.unclassifiedThemes and not unclassifiedTheme.isspace():
                        self.unclassifiedThemes.append(themeString[lastSplitIndex + 1:i + 1])

            # Always add the theme list to ensure all entries at of list type (stops empty string theme errors)
            pair[1] = themeList

    def getThemesCounts(self):
        for pair in self.themePairs:
            themesList = pair[1]

            # First theme listed is the "primary" theme for that document
            if themesList[0] not in self.primaryThemesCount.keys():
                self.primaryThemesCount[themesList[0]] = 1
            else:
                self.primaryThemesCount[themesList[0]] = self.primaryThemesCount[themesList[0]] + 1

            for theme in themesList:
                # Add to theme dictionary for maintaining count of themes
                if theme not in self.themesCount.keys():
                    self.themesCount[theme] = 1
                else:
                    self.themesCount[theme] = self.themesCount[theme] + 1

    def cullEmptyEntries(self):
        for pair in self.themePairs:
            if not isinstance(pair[0], str) or len(pair[0]) == 0 or pair[0].lower() == 'x' or len(pair[1]) == 0:
                self.themePairs.remove(pair)
                self.totalCulledEntries = self.totalCulledEntries + 1

    def removeNumericCharactersFromText(self):
        for pair in self.themePairs:
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

    def removeSingleLettersFromText(self):
        for pair in self.themePairs:
            rawText = pair[0]
            newText = ''

            for i in range(0, len(rawText)):
                # If the character is a valid letter and surrounded by whitespace then remove it
                if rawText[i] in string.ascii_letters and i + 1 < len(rawText):
                    if rawText[i - 1].isspace() and rawText[i + 1].isspace():
                        continue
                newText = newText + rawText[i]
            pair[0] = newText

    def removeKeywordsFromText(self):
        #  TODO - Create method for removing certain keywords (also consider keyword list generation (names, titles))
        pass

    def removeExtraSpacesFromText(self):
        punctuationWithoutPrecedingSpaces = [',', '.', '!', '?']
        for pair in self.themePairs:
            rawText = pair[0]
            newText = ''

            for i in range(0, len(rawText)):
                if rawText[i].isspace() and i + 1 < len(rawText):
                    if rawText[i + 1].isspace() or rawText[i + 1] in punctuationWithoutPrecedingSpaces:
                        continue
                newText = newText + rawText[i]
            pair[0] = newText
