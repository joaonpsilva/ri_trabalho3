from Posting import Posting
import os
import pickle

class Block_Reader():

    def __init__(self, filename):

        self.filename = filename

        print("Reading {}".format(filename))
        try:
            self.f = open(filename, 'r')
        except FileNotFoundError:
            print('File {} was not found'.format(filename))
            return

        self.chunk_Size = 10000
        self.i = 0
        self.indexList = []

        self.read_chunk(self.chunk_Size)

    def getEntry(self):
        return self.indexList[self.i]   #(term, [idf, postingList])
    
    def increment(self):
        
        if self.i == self.chunk_Size - 1:
            self.read_chunk(self.chunk_Size)
        else:
            self.i+=1

    def delete(self):
        print("Deleting {}".format(self.filename))
        os.remove(self.filename)

    def read_chunk(self, size=-1):  # size => numero de linhas que se quer ler, -1 -> ler ficheiro tudo

        self.indexList = []
        self.i = 0
        readLines = 0
        while True:
            
            if (readLines >= size != -1):
                break
            
            line = self.f.readline()  # ler linha a linha
            readLines += 1
            
            if not line:
                self.indexList.append(None)
                self.f.close()
                break

            # formato de cada linha:
            # term:idf ; doc_id:term_weight:pos1,pos2,pos3... ; doc_id:term_weight:pos1,pos2,pos3 ; ...
            l = line.split(";")
            term = l[0].split(":")[0]
            idf = float(l[0].split(":")[1])

            postingList = line[len(l[0]):]  #keep str format

            self.indexList.append( (term, [idf, postingList]) )
    
