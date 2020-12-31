from Indexer import Indexer
from Corpus import CorpusReader
from Tokenizer1 import Tokenizer1
from Tokenizer2 import Tokenizer2
from Posting import Posting
import time
import heapq
import argparse
import os
import psutil
from math import log10, sqrt
import collections

process = psutil.Process(os.getpid())

class BM25_Indexer(Indexer):

    def __init__(self,tokenizer, outputFile, k1=1.2, b=0.75):
        super().__init__(tokenizer, outputFile)
        self.k1 = k1
        self.b = b
        self.avdl = 0


    def calcAvdl(self, corpusreader):
        count=0
        while True:
            data = corpusreader.getNextChunk()
            if data is None:
                break

            for document in data:  # Iterate over Chunk of documents
                doi, title, abstract = document[0], document[1], document[2]
                tokens = self.tokenizer.process(title, abstract)
                self.avdl += len(tokens)
                count+=1

        self.avdl /= count

    def index(self, corpusreader):
        self.calcAvdl(corpusreader)
        super().index(corpusreader)

    def addTokensToIndex(self, tokens):

        #count occurs and positions
        tokensCount = self.extractDocData(tokens)

        #calc length
        dl = sum([info[0] for term, info in tokensCount.items()])

        #add to index
        for token, info in tokensCount.items():
            
            tf = info[0]
            positions = info[1]

            score = ((self.k1 + 1) * tf) / (self.k1 * ((1-self.b) + self.b*(dl / self.avdl)) + tf)

            posting = Posting(self.docID, score, positions)
            if token not in self.invertedIndex:
                self.invertedIndex[token] = [1, [posting]]
            else:
                self.invertedIndex[token][1].append(posting)
                self.invertedIndex[token][0] += 1    


    def score(self, query, ndocs=None):
        queryTokens = self.tokenizer.process(query)
        
        doc_scores = {}
        for term in queryTokens:

            if term not in self.invertedIndex:
                continue

            idf = self.invertedIndex[term][0]
            for doc in self.invertedIndex[term][1]:

                score = idf * doc.score 

                if doc.docID in doc_scores:
                    doc_scores[doc.docID] += score
                else:
                    doc_scores[doc.docID] = score

        if ndocs == None:
            bestDocs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)
        else:
            bestDocs = heapq.nlargest(ndocs, doc_scores.items(), key=lambda item: item[1])

        return [self.idMap[docid] for docid, score in bestDocs]
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-tokenizer", type=int, default=2, choices=[1, 2], help="tokenizer")
    parser.add_argument("-c", type=str, default="../metadata_2020-03-27.csv", help="Corpus file")
    args = parser.parse_args()

    corpusreader = CorpusReader(args.c)
    if args.tokenizer == 1:
        tokenizer = Tokenizer1()
    else:
        tokenizer = Tokenizer2()

    #CREATE INDEXER
    indexer = BM25_Indexer(tokenizer)
    
    #GET RESULTS
    t1 = time.time()
    indexer.index(corpusreader)
    t2 = time.time()

    print('seconds: ', t2-t1)
    print("Total memory used by program: ", process.memory_info().rss)
    
    keyList = list(indexer.invertedIndex.keys())
    print('Vocabulary size: ', len(keyList))

    indexer.write_to_file()

    #QUERY

    query = input("Query: ")
    print(indexer.score(query))

