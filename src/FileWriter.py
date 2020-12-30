from Posting import Posting
import os
import pickle


class FileWriter:
    def __init__(self, indexer):
        self.indexer = indexer
        self.start = 0  # To save last lines read in read_file

    def write_to_file(self, file="../Index.txt"):
        print("Writing to {}".format(file))
        # Apagar o ficheiro caso ele exista
        try:
            os.remove(file)
        except OSError:
            pass

        # Abrir o ficheiro em "append" para adicionar linha a linha (em vez de uma string enorme)
        f = open(file, "a")

        for term, values in sorted(self.indexer.invertedIndex.items()):
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
        with open(idMapFile, 'wb') as f:  # Init or clean file
            pickle.dump({}, f)
        self.indexer.idMapToDisk(idMapFile)

    def read_file(self, file="../Index.txt", size=-1):  # size => numero de linhas que se quer ler, -1 -> ler ficheiro todo
        print("Reading {}".format(file))
        try:
            f = open(file)
        except FileNotFoundError:
            print('File {} was not found'.format(file))
            return

        for i in range(self.start):  # ignorar linhas já lidas
            f.readline()

        # Inicializar o dicionario
        if self.start == 0:
            self.indexer.invertedIndex = {}
        readLines = 0
        while True:
            line = f.readline()  # ler linha a linha
            readLines += 1

            if not line or (readLines-1 >= size != -1):
                self.start += size
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

            self.indexer.invertedIndex[term] = [idf, postingList]

        f.close()
        print("Index created from file {}".format(file))

        # LOAD IDMAP
        idMapFile = os.path.splitext(file)[0] + "_idMapFile.pickle"
        print("Reading idMap from {}".format(idMapFile))
        self.indexer.loadIDMap(idMapFile)
