from Posting import Posting

class RangeIndex():

    def __init__(self, indexFile, begin, end):
        self.indexFile = indexFile
        self.begin = begin
        self.end = end
        self.index = {}
        self.used = 0
    
    def termInIndex(self, term):
        return term in self.index

    def getPostingList(self, term):
        return self.index[term]
    
    def belongs(self,term):
        if self.begin > term:       #term cant be indexed
            return 2
        if self.end < term:         #maybe next
            return 1
        
        self.used += 1              #term will be here
        return 0
    
    def clean(self):
        print("Discarding {}".format(self.indexFile))
        self.index = {}
    
    def isloaded(self):
        return not self.index == {}
    
    def read_Index(self):  # size => numero de linhas que se quer ler, -1 -> ler ficheiro tudo
    
        print("Reading {}".format(self.indexFile))
        
        f = open(self.indexFile)

        while True:
            line = f.readline()  # ler linha a linha

            if not line:
                break

            # formato de cada linha:
            # term:idf ; doc_id:term_weight:pos1,pos2,pos3... ; doc_id:term_weight:pos1,pos2,pos3 ; ...
            line = line.split(";")
            info = line[0].split(":")
            term = info[0]
            idf = float(info[1])

            postingList = [Posting(
                docID=int(values.split(":")[0]),  # doc_id
                score=float(values.split(":")[1]),  # term_weight
                positions=[int(position) for position in values.split(":")[2].split(",")])  # pos1,pos2,pos3...
                for values in line[1:]]  # values = [doc_id:term_weight:pos1,pos2 , doc_id:term_weight:pos1,pos2]

            self.index[term] = [idf, postingList]

        f.close()

    
    

