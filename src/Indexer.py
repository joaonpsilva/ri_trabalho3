from Corpus import CorpusReader
from Tokenizer1 import Tokenizer1
from Tokenizer2 import Tokenizer2
from Posting import Posting
import time
import heapq
import argparse
import os
import psutil
import pickle
from Block_Reader import Block_Reader
from math import log10

process = psutil.Process(os.getpid())


class Indexer():
    def __init__(self, tokenizer, outputFile):
        self.tokenizer = tokenizer
        self.idMap = {}
        self.invertedIndex = {}
        self.docID = 0
        self.blocks = 0
        self.blocksFolder = "../blocks/"
        self.outputFile = outputFile

    def hasEnoughMemory(self):
        print(process.memory_info().rss)
        return process.memory_info().rss < 3000000000   #4 GB

    def loadIDMap(self, idMapFile):
        with open(idMapFile, 'rb') as f:
            self.idMap = pickle.load(f)

    def extractDocData(self, tokens):
        # count occurs and positions
        # receives tokens
        # returns dict, key=term, value=(occur, [positions])
        tokensCount = {}
        for ind in range(len(tokens)):
            word = tokens[ind]

            if word not in tokensCount:
                tokensCount[word] = (1, [ind])
            else:
                info = tokensCount[word]
                info[1].append(ind)
                tokensCount[word] = (info[0] + 1, info[1])

        return tokensCount

    def addTokensToIndex(self, tokens):
        #DOESNT MATTER, its overide by bm25 and tfidf

        for word in tokens:  # Iterate over token of documents

            if word not in self.invertedIndex:
                self.invertedIndex[word] = [1, [self.docID]]
            else:
                if self.docID != self.invertedIndex[word][1][
                    -1]:  # check if word did not happen previously in same doc.
                    self.invertedIndex[word][1].append(self.docID)
                    self.invertedIndex[word][0] += 1

    def index(self, CorpusReader):

        count = 0
        flag = True

        while True:

            if self.hasEnoughMemory():
                data = CorpusReader.getNextChunk()
                if data is None:
                    print("Finished")
                    break

                for document in data:  # Iterate over Chunk of documents
                    doi, title, abstract = document[0], document[1], document[2]
                    self.idMap[self.docID] = doi  # map ordinal id used in index to real id

                    tokens = self.tokenizer.process(title, abstract)

                    self.addTokensToIndex(tokens)

                    self.docID += 1
            else:
                #No space available, save block
                self.dumpBlock()
                flag = False

        if flag:
            #only 1 block, write directly to outputfile
            self.write_to_file(self.outputFile)
        else:
            self.dumpBlock()
            #MERGE INDEXES
            self.mergeIndexes(self.outputFile)

    def dumpBlock(self):
        self.write_to_file(self.blocksFolder + str(self.blocks) + "Block.txt")
        self.blocks += 1
        self.invertedIndex = {}
        self.idMap = {}

    def mergeIndexes(self, filename):

        finalIndex = open(filename, 'w')
        print("Writing Index to {}".format(filename))

        numberOfBlocks = len([name for name in os.listdir(self.blocksFolder)]) / 2
        numberOfBlocks = int(numberOfBlocks)
        block_readers = [ Block_Reader(self.blocksFolder + str(i) + "Block.txt") for i in range(numberOfBlocks)]
        
        # WRITE IDMAP
        idMapFile = os.path.splitext(filename)[0] + "_idMapFile.pickle"
        print("Writing idMap to {}".format(idMapFile))

        content = {}
        for b in block_readers:
            content.update(b.read_IdMap())

        with open(idMapFile, 'wb') as f:
            pickle.dump(content, f)
        
        #MERGE INDEXES
        flag = False
        toRemove = []

        while True:

            idf = 0
            postingList = ""

            #GET NEXT WORD IN ALPHABETIC ORDER
            words = []
            for b in block_readers:
                
                entry = b.getEntry()

                if entry is None:    #Block is empty
                    flag = True
                    toRemove.append(b)
                else:
                    nextword = entry[0]
                    words.append(nextword)
                            
            #REMOVE READER WHICH BLOCK IS EMPTY 
            if flag:    
                for b in toRemove:
                    block_readers.remove(b)
                    b.delete()
                    numberOfBlocks-=1
                toRemove = []
                flag = False

                if numberOfBlocks == 0:
                    finalIndex.close()
                    return

            nextword = min(words)
            #GET INFO FOR THE GIVEN TERM FROM ALL BLOCKS
            for b in block_readers:
                entry = b.getEntry()
                term = entry[0]
                if term == nextword:
                    idf += entry[1][0]
                    postingList += ";" + entry[1][1]            #going through blocks in order so doc ids will be ordered
                    b.increment()   #if reader had the term, go to next line
            
            idf = log10(self.docID / idf)
            string = ('{}:{}{}'.format(term, idf, postingList)) + "\n"
            finalIndex.write(string)



    def write_to_file(self, file="../Index.txt"):
        print("Writing to {}".format(file))
        # Apagar o ficheiro caso ele exista
        try:
            os.remove(file)
        except OSError:
            pass

        # Abrir o ficheiro em "append" para adicionar linha a linha (em vez de uma string enorme)
        f = open(file, "a")

        for term, values in sorted(self.invertedIndex.items()):
            string = ('{}:{}'.format(term, values[0]))
            for posting in values[1]:
                string += (';{}:{}:'.format(posting.docID, posting.score))  # doc_id:term_weight
                for position in posting.positions:
                    string += ('{},'.format(position))  # pos1,pos2,pos3,…
                string = string[:-1]  # remover a virgula final (para ficar bonito)
            string += "\n"
            f.write(string)

        print("File {} created".format(file))
        f.close()

        # WRITE IDMAP
        idMapFile = os.path.splitext(file)[0] + "_idMapFile.pickle"
        print("Writing idMap to {}".format(idMapFile))
        with open(idMapFile, 'wb') as f:
            pickle.dump(self.idMap, f)


    def read_file(self, file="../Index.txt", size=-1):  # size => numero de linhas que se quer ler, -1 -> ler ficheiro tudo
        print("Reading {}".format(file))
        try:
            f = open(file)
        except FileNotFoundError:
            print('File {} was not found'.format(file))
            return

        self.invertedindex = {}
        readLines = 0
        while True:
            line = f.readline()  # ler linha a linha
            readLines += 1

            if not line or (readLines >= size != -1):
                break

            # formato de cada linha:
            # term:idf ; doc_id:term_weight:pos1,pos2,pos3... ; doc_id:term_weight:pos1,pos2,pos3 ; ...
            line = line.split(";")
            term = line[0].split(":")[0]
            idf = float(line[0].split(":")[1])

            postingList = [Posting(
                docID=int(values.split(":")[0]),  # doc_id
                score=float(values.split(":")[1]),  # term_weight
                positions=[int(position) for position in values.split(":")[2].split(",")])  # pos1,pos2,pos3...
                for values in line[1:]]  # values = [doc_id:term_weight:pos1,pos2 , doc_id:term_weight:pos1,pos2]

            self.invertedIndex[term] = [idf, postingList]

        f.close()
        print("Index created from file {}".format(file))

        # LOAD IDMAP
        idMapFile = os.path.splitext(file)[0] + "_idMapFile.pickle"
        print("Reading idMap from {}".format(idMapFile))
        self.loadIDMap(idMapFile)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-tokenizer", type=int, choices=[1, 2], required=True, help="tokenizer")
    parser.add_argument("-c", type=str, default="../metadata_2020-03-27.csv", help="Corpus file")
    args = parser.parse_args()

    corpusreader = CorpusReader(args.c)
    if args.tokenizer == 1:
        tokenizer = Tokenizer1()
    else:
        tokenizer = Tokenizer2()

    # CREATE INDEXER
    indexer = Indexer(tokenizer)

    # GET RESULTS
    t1 = time.time()
    indexer.index(corpusreader)
    t2 = time.time()

    print('seconds: ', t2 - t1)
    print("Total memory used by program: ", process.memory_info().rss)

    keyList = list(indexer.invertedIndex.keys())
    print('Vocabulary size: ', len(keyList))

    lessUsed = heapq.nsmallest(10, indexer.invertedIndex.items(), key=lambda item: (item[0], item[1][0]))
    print("First 10 terms with 1 doc freq: ", [i[0] for i in lessUsed])

    mostUsed = heapq.nlargest(10, indexer.invertedIndex.items(), key=lambda item: item[1][0])
    print("Higher doc freq: ", [(i[0], i[1][0]) for i in mostUsed])
