import argparse
import math
import time

parser = argparse.ArgumentParser()
parser.add_argument("-tokenizer", type=int, default=2, choices=[1, 2], help="tokenizer")
parser.add_argument("-c", type=str, default="../metadata_2020-03-27.csv", help="Corpus file")
parser.add_argument("-l", type=str, help="Load Inverted Index")
parser.add_argument("-i", type=str, choices=['bm25', 'tfidf'], required=True, help="Indexer")
parser.add_argument("-out", default="../models/Indexer.txt", type=str, help="Output file to save Inverted Index")
parser.add_argument("-relevant", type=str, default="../queries.relevance.filtered.txt",
                    help="file with the relevant query result")
parser.add_argument("--query", action="store_true", help="Process Queries")
args = parser.parse_args()


# Retorna um dicionario com formato {numero_da_query : [lista de docs relevantes]}
def getRelevantDocs():
    queries_relevance = open(args.relevant, "r")  # File with the relevant queries
    Lines = queries_relevance.readlines()
    docs_relevance = {}

    for line in Lines:
        line = line.split()
        if line[0] in docs_relevance:
            docs_relevance[line[0]].append(line[1])
        else:
            docs_relevance[line[0]] = [line[1]]

    return docs_relevance


def calculatePrecision(retrieved_docs, relevantList):
    interception = list(set(retrieved_docs) & set(relevantList))
    precision = len(interception) / len(retrieved_docs)
    return precision


def calculateRecall(retrieved_docs, relevantList):
    interception = list(set(retrieved_docs) & set(relevantList))
    recall = len(interception) / len(relevantList)
    return recall


def calculateF_Measure(precision, recall):
    if precision + recall == 0:  # Crash prevention
        return 0.0
    return (2 * precision * recall) / (precision + recall)


def calculateAveragePrecision(retrieved_docs, relevantList):
    averagePrecision = 0
    ret_docs = []
    relevantCount = 0
    for doc in retrieved_docs:
        ret_docs.append(doc)
        if doc in relevantList:
            averagePrecision += calculatePrecision(ret_docs, relevantList)
            relevantCount += 1

    averagePrecision = averagePrecision / relevantCount
    return averagePrecision


def calculateNDCG(retrieved_docs, number):
    queries_relevance = open(args.relevant, "r")  # File with the relevant queries
    Lines = queries_relevance.readlines()
    docs_scores = []
    d = {}  # dicionario com formato {doc: relevance}

    # Criação de um dicionario
    for line in Lines:
        line = line.split()
        if line[0] != number:
            continue
        d[line[1]] = line[2]

    score_list = []  # List with doc scores
    for doc in retrieved_docs:
        if doc in d:
            score_list.append(float(d[doc]))
        else:
            score_list.append(0.0)

    perfectNDCG = list(score_list)
    perfectNDCG.sort(reverse=True)

    if perfectNDCG[0] == 0:  # Se o doc com melhor relevance for 0 então ndcg = 0 (crash prevention)
        return 0

    # NDCG calculation
    realDCG = [score_list[0]]
    idealDCG = [perfectNDCG[0]]
    for i in range(1, len(score_list)):
        realDCG.append(realDCG[i - 1] + (score_list[i] / math.log2(i + 1)))
        idealDCG.append(idealDCG[i - 1] + (perfectNDCG[i] / math.log2(i + 1)))
    # print(realDCG)
    # print(idealDCG)

    NDCG = [x / y for x, y in zip(realDCG, idealDCG)]  # Divide realDCG by idealDCG
    return NDCG[-1]


def calculateMean(valores):
    precision = 0
    recall = 0
    f_measure = 0
    avgPrecision = 0
    latecy = 0
    ndcg = 0
    latencyList = []
    size = len(valores.keys())
    for key in valores.keys():
        precision += valores[key]["precision"]
        recall += valores[key]["recall"]
        f_measure += valores[key]["f-measure"]
        avgPrecision += valores[key]["average Precision"]
        ndcg += valores[key]["ndcg"]
        latecy += valores[key]["latecy"]
        latencyList.append(valores[key]["latecy"])

    precision /= size
    recall /= size
    f_measure /= size
    avgPrecision /= size
    ndcg /= size
    latecy /= size

    # Median Latency
    latencyList = sorted(latencyList)
    lstLen = len(latencyList)
    index = (lstLen - 1) // 2

    if lstLen % 2:
        medianLatency = latencyList[index]
    else:
        medianLatency = (latencyList[index] + latencyList[index + 1]) / 2.0

    return {"precision": precision, "recall": recall, "f-measure": f_measure, "average Precision": avgPrecision,
            "ndcg": ndcg, "latecy": medianLatency}


def writeToCsv(valores10, valores20, valores50, indexername):
    import csv
    with open('../results/results' + indexername + '.csv', mode='w') as file:
        writer = csv.writer(file)
        writer.writerow(
            ["Query", "", "Precision", "", "", "Recall", "", "", "F-Measure", "", "", "Average Precision", "", "",
             "NDCG", "", "", "Latency", "", ])
        writer.writerow(
            ["", "@10", "@20", "@50", "@10", "@20", "@50", "@10", "@20", "@50", "@10", "@20", "@50", "@10", "@20",
             "@50", "@10", "@20", "@50"])
        for key in valores10.keys():
            writer.writerow([key, valores10[key]["precision"], valores20[key]["precision"], valores50[key]["precision"],
                             valores10[key]["recall"], valores20[key]["recall"], valores50[key]["recall"],
                             valores10[key]["f-measure"], valores20[key]["f-measure"], valores50[key]["f-measure"],
                             valores10[key]["average Precision"], valores20[key]["average Precision"],
                             valores50[key]["average Precision"],
                             valores10[key]["ndcg"], valores20[key]["ndcg"], valores50[key]["ndcg"],
                             valores10[key]["latecy"], valores20[key]["latecy"], valores50[key]["latecy"], ])


if args.tokenizer == 1:
    from Tokenizer1 import Tokenizer1

    tokenizer = Tokenizer1()
else:
    from Tokenizer2 import Tokenizer2

    tokenizer = Tokenizer2()

# INDEXER
if args.i == 'bm25':
    from BM25_Indexer import BM25_Indexer
    indexer = BM25_Indexer(tokenizer, args.out)
else:
    from Tf_Idf_Indexer import Tf_idf_Indexer
    indexer = Tf_idf_Indexer(tokenizer, args.out)

# INV_INDEX
if args.l is not None:  # LOAD INV IND FROM FILE
    t1 = time.time()
    indexer.read_file(args.l)
    t2 = time.time()
    print('Loading Time: ', t2 - t1)

else:  # BUILD INV IND FROM CORPUS
    from Corpus import CorpusReader

    corpusreader = CorpusReader(args.c)
    t1 = time.time()
    indexer.index(corpusreader)
    t2 = time.time()
    print('Indexing Time: ', t2 - t1)

# PROCESS QUERIES
if args.query:
    import xml.etree.ElementTree as ET

    root = ET.parse('../queries.txt.xml').getroot()

    relevant_docs = getRelevantDocs()  # dicionario com formato {numero_da_query : [lista de docs relevantes]}

    for size in [10, 20, 50]:
        valores={}                       # dicionario de dicionario com formato {numero_da_query : {precision: valor , recall:valor , ...}
        for entrie in root.findall('topic'):
            number = entrie.get('number')
            query = entrie.find('query').text

            start_time = time.time()
            retrieved_docs = indexer.proximityScore(query, size)
            stop_time = time.time()

            if size == 50:
                print(number, ' - ', query)
                print(retrieved_docs)
                print("\n")
            valores[number] = {}  # inicializar o dicionario nested

            valores[number]["latecy"] = (stop_time - start_time)
            precision = valores[number]["precision"] = calculatePrecision(retrieved_docs, relevant_docs[number])
            recall = valores[number]["recall"] = calculateRecall(retrieved_docs, relevant_docs[number])
            valores[number]["f-measure"] = calculateF_Measure(precision, recall)
            valores[number]["average Precision"] = calculateAveragePrecision(retrieved_docs, relevant_docs[number])
            valores[number]["ndcg"] = calculateNDCG(retrieved_docs, number)

        valores["mean"] = calculateMean(valores)

        # Guardar os valores no respetivo dicionario
        if size == 10:
            valores10 = dict(valores)  # sem o dict ia copiar a referencia
        elif size == 20:
            valores20 = dict(valores)
        else:
            valores50 = dict(valores)

    # Print da tabela
    # print(valores)
    from prettytable import PrettyTable

    for size in [10, 20, 50]:
        if size == 10:
            valores = valores10
        elif size == 20:
            valores = valores20
        elif size == 50:
            valores = valores50
        t = PrettyTable(['Query', 'Precision', 'Recall', 'F-Measure', 'Average Precision', 'NDCG', 'Latency'])
        for key in valores:
            t.add_row([key, valores[key]["precision"], valores[key]["recall"], valores[key]["f-measure"],
                       valores[key]["average Precision"], valores[key]["ndcg"], valores[key]["latecy"]])

        print("Query size: {}".format(size))
        print(t)

    writeToCsv(valores10, valores20, valores50, args.i)
