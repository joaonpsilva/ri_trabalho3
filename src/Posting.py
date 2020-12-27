from math import log10

class Posting:

    def __init__(self, docID, score, positions=[]):
        self.docID = docID
        self.score = score
        self.positions = positions #list of ints

    def __repr__(self):
        return str(self.docID) + ", " + str(self.score)