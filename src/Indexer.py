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
from Posting_Iterator import Posting_Iterator
from shutil import rmtree

process = psutil.Process(os.getpid())


class Indexer():
    def __init__(self, tokenizer, outputFolder):
        self.tokenizer = tokenizer
        self.idMap = {}
        self.invertedIndex = {}
        self.docID = 0
        self.blocks = 0
        self.outputFolder = outputFolder
        self.blocksFolder = outputFolder + "Blocks/"
        self.idMapFile = outputFolder + "idMap.pickle"

        if os.path.exists(self.outputFolder):
            rmtree(self.outputFolder)
        
        print("Creating {} folder".format(self.outputFolder))
        print("Creating {} folder".format(self.blocksFolder))
        os.makedirs(self.outputFolder)
        os.makedirs(self.blocksFolder)
        
        print("Creating {} file".format(self.idMapFile))
        with open(self.idMapFile, 'wb') as f:
            pickle.dump({}, f)

    def hasEnoughMemory(self):
        #print(process.memory_info().rss)
        return process.memory_info().rss < 3000000000   #4 GB

    def loadIDMap(self):
        with open(self.idMapFile, 'rb') as f:
            self.idMap = pickle.load(f)
    
    def writeIDMap(self):
        print("Updating IdMapFile")
       
        with open(self.idMapFile, 'rb') as f:
            content = pickle.load(f)
        
        self.idMap.update(content)

        with open(self.idMapFile, 'wb') as f:
            pickle.dump(self.idMap, f)

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
    
    def build_idf(self):
        for term, valList in self.invertedIndex.items():
            valList[0] = log10(self.docID/valList[0])

    def index(self, CorpusReader):
        print("Indexing...")
        count = 0
        flag = True

        while True:

            if self.hasEnoughMemory():
                data = CorpusReader.getNextChunk()
                if data is None:
                    print("Finished Indexing")
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
            self.build_idf()
            self.write_to_file(self.outputFolder + "000-zzz.txt")
            self.writeIDMap()

        else:
            self.dumpBlock()
            #MERGE INDEXES
            self.mergeBlocks()
        
        os.rmdir(self.blocksFolder)

    def dumpBlock(self):
        self.write_to_file(self.blocksFolder + str(self.blocks) + "Block.txt")
        self.invertedIndex = {}

        self.writeIDMap()
        self.idMap = {}

        self.blocks += 1

    def mergeBlocks(self):
        
        print("Merging Indexes")

        block_readers = [ Block_Reader(self.blocksFolder + str(i) + "Block.txt") for i in range(self.blocks)]
        optiFileSize = os.path.getsize(self.blocksFolder + "0Block.txt") - 1000 #-1000 for good mesure 
        
        #MERGE INDEXES
        flag = False
        toRemove = []
        currentFile = None
        firstWord = ""
        lastWord = ""

        while True:

            idf = 0
            postingList = ""

            #GET List OF NEXT WORDS FROM BLOCKS IN ALPHABETIC ORDER
            words = []
            for b in block_readers:
                
                entry = b.getEntry()

                if entry is None:    #Block is empty
                    flag = True
                    toRemove.append(b)
                else:
                    words.append(entry[0])
                            
            #REMOVE READER WHICH BLOCK IS EMPTY 
            if flag:    
                for b in toRemove:
                    block_readers.remove(b)
                    b.delete()
                    self.blocks-=1
                toRemove = []
                flag = False

                if self.blocks == 0:
                    currentFile.close()
                    os.rename(self.outputFolder + firstWord + ".txt", self.outputFolder + firstWord + "_" + lastWord + ".txt")
                    print("Finished Merging")
                    return
            
            nextword = min(words)   #get next word in alphabetic order

            if currentFile is None: #create file with 1st term
                firstWord = nextword
                currentFile = open(self.outputFolder + firstWord + ".txt", "w")


            #GET INFO FOR THE GIVEN TERM FROM ALL BLOCKS
            for b in block_readers:
                entry = b.getEntry()
                term = entry[0]
                if term == nextword:
                    idf += entry[1][0]
                    postingList += ";" + entry[1][1]            #going through blocks in order so doc ids will be ordered
                    b.increment()   #if reader had the term, go to next line
            
            idf = log10(self.docID / idf)
            string = ('{}:{}{}'.format(nextword, idf, postingList)) + "\n"
            currentFile.write(string)                                           #write result to file

            lastWord = nextword

            if currentFile.tell() > optiFileSize:           #good size, close file
                currentFile.close()
                os.rename(self.outputFolder + firstWord + ".txt", self.outputFolder + firstWord + "_" + lastWord + ".txt")
                currentFile = None



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

        print("Done writting to {}".format(file))
        f.close()



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
        print("Reading idMap from {}".format(self.idMapFile))
        self.loadIDMap()
    
    def proximityScore(self, query, ndocs=None):
        queryTokens = self.tokenizer.process(query)

        if len(queryTokens) == 1:
            return self.score(query, ndocs)

        postList = [Posting_Iterator(term, self.invertedIndex[term]) for term in queryTokens if term in self.invertedIndex]
        toRemove = []

        docScores = {}
        while True:
            
            #Get next Posting
            smallestPost = None
            for p in postList:
                postingInfo = p.getPosting()

                if postingInfo == None: #if postingList got to the end
                    toRemove.append(p)
                    continue
                
                if smallestPost == None or postingInfo < smallestPost.getPosting(): #get next posting: compare docID, then position
                    smallestPost = p
            

            #remove postings that got to the end
            if len(toRemove) != 0:
                for p in toRemove:
                    postList.remove(p)
                toRemove = []
            if len(postList) < 2:
                break


            #iterate over the postings
            for p in postList:

                if self.areNextToEachOther(smallestPost.getPosting(), p.getPosting()):
                    
                    #Account for postings
                    score = self.getScore(smallestPost) + self.getScore(p) 
                    if score != 0:
                        doc = smallestPost.getPosting()[0]

                        if doc in docScores:
                            docScores[doc] += score
                        else:
                            docScores[doc] = score
            
            smallestPost.increment()
        
        if ndocs == None:
            bestDocs = sorted(docScores.items(), key=lambda item: item[1], reverse=True)
        else:
            bestDocs = heapq.nlargest(ndocs, docScores.items(), key=lambda item: item[1])

        return [self.idMap[docid] for docid, score in bestDocs]


    def areNextToEachOther(self, postingInfo1, postingInfo2):
        if postingInfo1[0] == postingInfo2[0]:    #same doc
            if 0 != abs(postingInfo1[1] - postingInfo2[1]) < 5:
                return True
        return False

    def getScore(self, p):
        if p.isAccounted():
            return 0
        
        p.setAccounted()
        return self.calcScore(p.idf, p.getScore())
    
    def calcScore(self, idf, score):    #defined in children
        pass
    def score(self, query, ndocs=None): #defined in children
        pass
        






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
