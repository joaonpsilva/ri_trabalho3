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


class Tf_idf_Indexer(Indexer):

    def __init__(self, tokenizer, indexFolder):
        super().__init__(tokenizer, indexFolder)


    def index(self,corpusreader):
        super().index(corpusreader)

    def addTokensToIndex(self, tokens):

        #count occurs and positions
        tokensCount = self.extractDocData(tokens)

        #update with scores
        for term, info in tokensCount.items():
            tokensCount[term] = (1 + log10(info[0]), info[1])
        
        #doc length
        doc_length = sqrt(sum([info[0] ** 2 for info in tokensCount.values()]))

        #add to index
        for token, info in tokensCount.items():
            
            score = info[0]
            positions = info [1]

            score /= doc_length
            posting = Posting(self.docID, score, positions)

            if token not in self.invertedIndex:
                self.invertedIndex[token] = [1, [posting]]
            else:
                self.invertedIndex[token][1].append(posting)
                self.invertedIndex[token][0] += 1

    def calcScore(self, index, queryTokenss):
        queryTokens = collections.Counter(queryTokenss).most_common()  # [(token, occur)]


        query_weights = {term: ((1 + log10(occur)) * index[term][0])
                         for term, occur in queryTokens if term in index}

        query_length = sqrt(sum([score ** 2 for score in query_weights.values()]))

        doc_scores = {}
        for term, termScore in query_weights.items():
            termScore /= query_length

            for doc in index[term][1]:
                if doc.docID in doc_scores:
                    doc_scores[doc.docID] += doc.score * termScore
                else:
                    doc_scores[doc.docID] = doc.score * termScore  
        
        return doc_scores


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

    # CREATE INDEXER
    indexer = Tf_idf_Indexer(tokenizer)

    # GET RESULTS
    t1 = time.time()
    indexer.index(corpusreader)
    t2 = time.time()

    print('seconds: ', t2 - t1)
    print("Total memory used by program: ", process.memory_info().rss)

    keyList = list(indexer.invertedIndex.keys())
    print('Vocabulary size: ', len(keyList))

    # QUERY
    query = input("Query: ")
    print(indexer.score(query))