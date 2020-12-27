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

process = psutil.Process(os.getpid())


class Indexer():
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.idMap = {}
        self.invertedIndex = {}
        self.docID = 0

    def hasEnoughMemory(self):
        return True

    def idMapToDisk(self, idMapFile):
        with open(idMapFile, 'rb') as f:
            content = pickle.load(f)

        content.update(self.idMap)

        with open(idMapFile, 'wb') as f:
            pickle.dump(content, f)

        self.idMap = {}

    def loadIDMap(self, idMapFile):
        with open(idMapFile, 'rb') as f:
            self.idMap = pickle.load(f)

    def extractDocData(self, tokens):
        # count occurs and positions
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
        while self.hasEnoughMemory():

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

    def write_to_file(self, file="../Index.txt"):
        print("Writing to {}".format(file))
        # Apagar o ficheiro caso ele exista
        try:
            os.remove(file)
        except OSError:
            pass

        # Abrir o ficheiro em "append" para adicionar linha a linha (em vez de uma string enorme)
        f = open(file, "a")

        for term, values in self.invertedIndex.items():
            string = ('{}:{}'.format(term, values[0]))
            for posting in values[1]:
                string += (';{}:{}:'.format(posting.docID, posting.score))  # doc_id:term_weight
                for position in posting.positions:
                    string += ('{},'.format(position))  # pos1,pos2,pos3,…
                string = string[:-1]    # remover a virgula final (para ficar bonito)
            string += "\n"
            f.write(string)

        print("File {} created".format(file))
        f.close()

        # WRITE IDMAP
        idMapFile = os.path.splitext(file)[0] + "_idMapFile.pickle"
        print("Writing idMap to {}".format(idMapFile))
        with open(idMapFile, 'wb') as f:  # Init or clean file
            pickle.dump({}, f)
        self.idMapToDisk(idMapFile)

    def read_file(self, file="../Index.txt"):
        print("Reading {}".format(file))
        try:
            f = open(file)
        except FileNotFoundError:
            print('File {} was not found'.format(file))
            return
        self.invertedIndex = {}

        while True:
            line = f.readline()  # ler linha a linha
            if not line:
                break

            # formato de cada linha:
            # termo:idf ; doc_id:term_weight ; doc_id:term_weight ; ...
            line = line.split(";")
            term = line[0].split(":")[0]
            idf = float(line[0].split(":")[1])

            postingList = [Posting(int(values.split(":")[0]), float(values.split(":")[1])) for values in line[1:]]

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
