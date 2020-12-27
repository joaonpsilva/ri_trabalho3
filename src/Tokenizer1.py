import re


class Tokenizer1:

    def __init__(self):
        self.regex = re.compile('[^a-zA-Z]')

    def process(self, *phrases):

        terms = []
        for phrase in phrases:

            phrase = self.regex.sub(' ', phrase)
            phrase = phrase.lower()
            phraseVector = phrase.split(' ')

            for word in phraseVector:
                if len(word) < 3:
                    continue
                terms.append(word)

        return terms
