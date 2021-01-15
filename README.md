# RI ASSIGNEMT 3

https://github.com/joaonpsilva/ri_trabalho3

João Silva 88813  
Bernardo Rodrigues 88835

## Run

THE CORPUS FILE IS NOT INCLUDED (TESTED WITH 2021-01-01 and 2020-25-12/metadata.csv). Should work with any other.

Inside src/:

Main File Weighted_Indexer.py  
Main Options:  
 - -i [indexer] -> (tfidf, bm25)
 - -f [index folder] -> (default="../model/") Where to store/read index
 - -c [Corpus File] -> File to index (if not provided program assumes index is already indexed)
 - -mem [memory in GB] -> (default all) limit memory to the process (not well tested for values under 2GB)
 - --query -> Flag to process queries
 - --proxBoost -> Flag to apply proximityBoost

### EXAMPLES:  

Index using tfidf, save into ../myfolder/, use only 5gb:  
$python3 Weighted_Indexer.py -i tfidf -c ../2021-01-01/metadata.csv -f ../myfolder/ -mem 5


Query the index saved into ../myfolder/ use proximity boost:  
$python3 Weighted_Indexer.py -i tfidf -f ../myfolder/ --query --proxBoost


Index using bm25, save into ../model/, make queries without proximity Boost:    
$python3 Weighted_Indexer.py -i bm25 -c ../2021-01-01/metadata.csv --query


### Notes

Program will start indexing the corpus. 
Everytime memory gets critical, dump index (alphabetically)to Block/ folder inside the index folder.  

Merge all bocks.txt inside Block/ folder. Read small chunks of all blocks, choose next word and merge all posting lists, write to index file inside index folder. Every time index file has >30mb make new index file. Name of the index files are firtTerm_lastTerm.txt

During Query, the indexer selects from which file to read and loads that part of the index, keeping track of which part of the index is related to which range of terms, and how many times that part of the index has been queried. The index does not get rid of parts of the index until it is needed. When it needs, the indexer chooses parts of the index that have been queried the least.  


The query score is calculated normally.   
If proxBoost is flagged, apply proxBoost Score. Keep pointer to all posting lists and go through all of them increasing the smallest posting (each position of a posting is considered a posting by itself).   
Grab the current posting of each list, choose smallest, keep all postings from same doc. Calculate best score from those postings having into account all n postings, n-1, n-1, ..., 2 (try to give more score to 4 words close than 5 words far appart, etc).   
From experiments, found that for small number of terms in query (<4) it is more important to have all terms than have only 2 terms close. For large number of terms is more important to have part of them close than all terms in doc (4 terms close better than 6 terms far appart). Doesn't really work.

### Results:

If --query is present the program will output the 50 better results for each query as well as the topics requested in 2.2 for 10, 20, 50 results. This last results can also be found in results/ folder for bm25 and tfidf ( results are rewritten everytime --query is flaged).  
Overall, the average results for the bm25 ranking function were better than the ones present in tf-idf.

Proximity Boost takes longer to query.
Increases the results for tdidf but the results for bm25 are more or less the same (without proximity are a bit better).

However, for example (for bm25):

QUERY 26  -  coronavirus early symptoms  
Retrived doc 9bce960n as 5th.

"
**Coronavirus** disease 2019 (COVID-19) is a new infectious disease that spreads very rapidly and therefore, WHO has declared it as a global pandemic disease. 
The main clinical **symptoms** found in COVID-19 patients are cough and fever; however, in some cases, diarrhea can be one of the **early symptoms**. The present
case report describes a patient who came with a complaint of diarrhea withoutfever and she was later confirmed to be positive for COVID-19 during 
hospitalization. The presence of unspecified initial **symptoms** calls for greatervigilance from health workers in establishing diagnosis patients with 
COVID-19.
"

Which has a relevance of 0, but looks like something the indexer should return.


### Python libraries
argparse  
time  
math  
os  
psutil  
heapq  
pickle  
shutil  
xml  
prettytable  
Stemmer  
re  
collections  