class Posting_Iterator:
    ''' Iterate over posting list
    Create the ilusion that each position of a posting is a posting by itself so that it is easier
    to iterate over positions in  doc'''

    def __init__(self, term, postingList):
        self.term = term
        self.postingList = postingList
        self.i = 0
        self.j = 0
    
    def getPosting(self):

        if self.i == len(self.postingList):
            return None

        posting = self.postingList[self.i]

        return (posting.docID , posting.positions[self.j])  #(docId, position) 
    

    def increment(self):
        #increment to new position, or posting if positions are over
        
        if  self.j == len(self.postingList[self.i].positions) - 1:  #next doc
            self.j = 0
            self.i += 1
        else:                                                               #next pos
            self.j += 1