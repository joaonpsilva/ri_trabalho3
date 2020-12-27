# RI ASSIGNEMT 2

https://github.com/joaonpsilva/RI_TRABALHO2

João Silva 88813  
Bernardo Rodrigues 88835

## Run

Inside src/:

Main File Weighted_Indexer.py  
Main Options:  
 - -i [indexer] -> (tfidf, bm25)
 - -out [outFile] -> save inv. index to file and a Map with the ids to cordId
 - -l [File] -> Load inv. index from file saved with last option (if not provided, inverted index will be indexed on the current run from metadata_2020-03-27.csv, this file can be changed
 with -c)
 - --query -> Flag to process queries.relevance.filtered.txt (-relevance to change queries relevance file)

### EXAMPLES:  

Save index to file:  
$python3 Weighted_Indexer.py -i tfidf --query -out ../models/tfidfIndex.txt

Load index from file:  
$python3 Weighted_Indexer.py -i bm25 --query -l ../models/bm25Index.txt

### Results:

If --query is present the program will output the 50 better results for each query as well as the topics requested in 2.2 for 10, 20, 50 results. This last results can also be found in results/ folder for bm25 and tfidf ( results are rewritten everytime --query is flaged).  
Overall, the average results for the bm25 ranking function were better than the ones present in tf-idf.
time:

TFI-IDF:
 - Indexing Time:  13.400408744812012
 - Loading from file Time:  4.960155725479126

BM25:
 - Indexing Time:  17.63304090499878
 - Loading from file Time:  5.16562032699585


 Query time note:  
 For some reason queries with a index that has been loaded using the -l are faster (~0.002s) than
 queries with a index that has been idexed just now (without -l) (~0.006s)




