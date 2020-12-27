from pandas import read_csv



class CorpusReader():

    def __init__(self, csvfile, chunkSize=10000):
        self.fileIterator = read_csv(csvfile, chunksize=chunkSize, iterator=True)
        self.csvfile = csvfile
        self.chunkSize = chunkSize

    def getNextChunk(self):
        try:
            chunk = self.fileIterator.get_chunk()
        except StopIteration:
            self.fileIterator = read_csv(self.csvfile, chunksize=self.chunkSize, iterator=True)
            return None

        chunk = chunk.dropna(subset=['abstract', 'title'])   #drop empty abstract
        chunk = chunk[['cord_uid', 'title', 'abstract']] #keep only these fields

        return chunk.values
