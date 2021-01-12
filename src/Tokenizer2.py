
import re
import Stemmer


class Tokenizer2:

    def __init__(self, stopwordsfile='../snowball_stopwords_EN.txt'):
        self.stemmer = Stemmer.Stemmer("english")
        self.stopwords = self.buildStopWords(stopwordsfile)
        self.regex = re.compile('[^a-zA-Z0-9\' -]')

    def buildStopWords(self, file):
        reader = open(file, 'r')
        return set(reader.read().splitlines())

    def removeStopWords(self, words):
        return [word for word in words if not word in self.stopwords]

    def process(self, *phrases):
        terms = []

        for phrase in phrases:
 
            phrase = self.regex.sub(' ', phrase)
            phrase = phrase.lower()
            words = phrase.split(' ')

            words = [word.strip('.\'-') for word in words if len(word.strip('.\'-')) > 2]  # removes small words, https part and remove apostrofes and hifens in the begining and end
            
            words = self.removeStopWords(words)
            words = self.stemmer.stemWords(words)

            terms += words

        return terms
