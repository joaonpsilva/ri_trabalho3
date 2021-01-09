class Posting_Iterator:

    def __init__(self, term, postingList):
        self.term = term
        self.idf = postingList[0]
        self.postingList = postingList[1]
        self.i = 0
        self.j = 0
        self.alreadyAccounted = False
    
    def getPosting(self):

        if self.i == len(self.postingList):
            return None

        posting = self.postingList[self.i]

        return (posting.docID , posting.positions[self.j])
    

    def getScore(self):
        return self.postingList[self.i].score
    

    def setAccounted(self):
        self.alreadyAccounted = True        #only account for score once per doc
    
    def isAccounted(self):
        return self.alreadyAccounted
    
    def increment(self):
        
        if  self.j == len(self.postingList[self.i].positions) - 1:  #next doc
            self.j = 0
            self.i += 1
            self.alreadyAccounted = False
        else:                                                               #next pos
            self.j += 1