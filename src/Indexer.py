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
    def __init__(self, tokenizer, indexFolder):
        self.tokenizer = tokenizer
        self.idMap = {}
        self.invertedIndex = {}
        self.docID = 0
        self.blocks = 0
        self.indexFolder = indexFolder
        self.blocksFolder = indexFolder + "Blocks/"
        self.idMapFile = indexFolder + "idMap.pickle"
        self.ranges = []
        self.currentRange =  ("", "")
    

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

        #Preparation
        print("Indexing...")

        if os.path.exists(self.indexFolder):
            rmtree(self.indexFolder)
        
        print("Creating {} folder".format(self.indexFolder))
        print("Creating {} folder".format(self.blocksFolder))
        os.makedirs(self.indexFolder)
        os.makedirs(self.blocksFolder)
        
        print("Creating {} file".format(self.idMapFile))
        with open(self.idMapFile, 'wb') as f:
            pickle.dump({}, f)

        #Indexing
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
            self.write_to_file(self.indexFolder + "000_zzz.txt")
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
                    os.rename(self.indexFolder + firstWord + ".txt", self.indexFolder + firstWord + "_" + lastWord + ".txt")
                    print("Finished Merging")
                    return
            
            nextword = min(words)   #get next word in alphabetic order

            if currentFile is None: #create file with 1st term
                firstWord = nextword
                currentFile = open(self.indexFolder + firstWord + ".txt", "w")


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
                os.rename(self.indexFolder + firstWord + ".txt", self.indexFolder + firstWord + "_" + lastWord + ".txt")
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



    def read_Index(self, file="../Index.txt"):  # size => numero de linhas que se quer ler, -1 -> ler ficheiro tudo
        
        print("Reading {}".format(file))
        
        f = open(file)

        #self.invertedindex = {}
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

            self.invertedIndex[term] = [idf, postingList]

        f.close()

    
    def loadIndex(self):
        self.loadIDMap()

        for filename in os.listdir(self.indexFolder):
            if filename.endswith(".txt"): 
                filename = filename[:-4]
                wordrange = filename.split('_')

                self.ranges.append((wordrange[0], wordrange[1]))

    
    def proximityBoost(self, index, doc_scores, dmax=4, numberOfTermsWeight=0.8, distanceWeight=0.2 ):

        postList = [Posting_Iterator(term, info[1]) for term, info in index.items() ]

        nterms = len(index.keys())

        currDoc = min(doc_scores.keys())
        currScore = 0.5
        while True:
            
            postings = [p for p in postList if p.getPosting() != None]    #list of next postings for each term

            if len(postings) == 0:
                doc_scores[currDoc] *= currScore
                break
            

            smallestPost = heapq.nsmallest(1, postings, key=lambda item: item.getPosting())[0]
            smallestPosition = smallestPost.getPosting()

            if smallestPosition[0] != currDoc:  #changed doc
                doc_scores[currDoc] *= currScore
                currDoc = smallestPosition[0]
                currScore = 0.5

            documentPositions = sorted([pos.getPosting()[1] for pos in postings if pos.getPosting()[0]==smallestPosition[0]])

            for n in range(len(documentPositions), 1, -1):  #check best score possible for all words, words-1, -2, ... until only 2 words
                
                lastpos = documentPositions[n-1]
                firstpos = documentPositions[0]
                distance = lastpos - firstpos

                if distance > (n-1) * dmax:    #words are not close
                    continue
                
                m = distanceWeight / (-(dmax-1) * (n-1))         #line, when d = n, score = 0.5      when d >= 4n, score = 0
                x = distance - (n-1)            #line points p1 = (n-n, 0.5), p2 = (4n - n, 0)
                distanceScore = m * x + distanceWeight

                ntermScore = n * numberOfTermsWeight / nterms       #0.5 is the max to get if all terms are present

                totalscore = ntermScore + distanceScore

                if totalscore > currScore:
                    currScore = totalscore

                    '''print("number of terms in query: ", nterms)
                    print("number of terms Considered: ", n)
                    print("positions: ", documentPositions)
                    print("distance: ", distance)
                    print("x: ", x)                
                    print("ntermScore: ", ntermScore)
                    print("distanceScore: ", distanceScore)
                    print("totalScore: ", totalscore)
                    print("---------------")'''

            smallestPost.increment()
        
        return doc_scores

    
    def score(self, query, ndocs=None, proxBoost = True): #defined in children

        queryTokens = sorted(self.tokenizer.process(query))
        tokens =  set(queryTokens[:])
        smallIndex = {}

        #Check for words in index in memory
        flag = False
        for term in set(queryTokens):

            if self.currentRange[0] <= term <= self.currentRange[1]:    #if term can be in Index
                if term in self.invertedIndex:
                    smallIndex[term] = self.invertedIndex[term]
                flag = True
                tokens.remove(term)
            else:           
                if flag:    #term out of range
                    break
        
        for term in tokens: #for the rest of the tokens
            if not (self.currentRange[0] <= term <= self.currentRange[1]) : #if cant be in Index
                for wordRange in self.ranges:
                    if wordRange[0] <= term <= wordRange[1]:        #find new index
                        self.invertedIndex = {}
                        self.read_Index(self.indexFolder + wordRange[0] + "_" + wordRange[1] + ".txt")
                        self.currentRange = wordRange
                        break
                    
            if term in self.invertedIndex:      #if could find index before see if is there. If couldnt find index before will return false
                smallIndex[term] = self.invertedIndex[term]
            
            flag = False
        
        #CALC SCORE
        doc_scores = self.calcScore(smallIndex)

        #APPLY PROX BOOST
        if proxBoost and len(smallIndex.keys()) > 1:
            doc_scores =  self.proximityBoost(smallIndex, doc_scores)
        
        #number of docs to return
        if ndocs == None:
            bestDocs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        else:
            bestDocs = heapq.nlargest(ndocs, doc_scores.items(), key=lambda item: item[1])

        return [self.idMap[docid] for docid, score in bestDocs]


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
