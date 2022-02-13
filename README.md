# <b> IR-Project <b>
A search Engine built upon the whole Englsih Wikipedia Pages.
The followings are descriptions of the modules provided in this repository:

## <b>inverted_index_gcp<b>
This module is based on the inverted_index modules from previous home assignments.
It has various fields which contain information about the length of it's documents, the total number of a term's frequency,
document's freqeuncy, etc'.
It is capable of reading and loading an already existing index from local or remote storage. 

## <b>backend<b>
This module provides the <b>CalcNpreps<b> class, which is responsible for parsing queries,
calculating tf,idf and tf-idf values in order to produce a cosine similarity score for a given query.
These calculations are based on formulas learned in class.

## <b>search_fronted<b>
The main module where the magic happens! You can find here the required functions implemented.

### <ins><b>search<b><ins>
This is our best shot. With map@40 score of ~0.5, we follow 3 stages: <br />
1. Query processing: <br />
Query expansion & reduction - Using a special NLTK package, we analyze the query's tokens and distinguish wether it is a verb, noun, etc'.
After carefull examination, we decided to keep only tokens we defined as 'good kinds'(NN, VB,...).
Then, we try to expand the query in the following fashion: If a word is a plural noun (i.e ends with 's' or 'es'), we add it's singular form.<br /> <br />

2. Searching by the title of the document:<br />
We count for each Doc how many distinct words from the query are in it's title. Then, we normalize the count of distinct words from the query in the title, by dividing it by the length of the title
(# of distinct query tokens in the title / # of total tokens in the title)<br />

3. Sorting and normalizing - we sort the results by the normalized values described in stage 2, save only the top 300 results.<br />
Then, we take those top 300 results and assign them with their pageviews, sorting them in decsending order and return the final top 100 results from the list.<br /> <br />

### <ins><b>search_body<b><ins> <br />
Searching  through the body of the articles using the tf-idf to calc cosine similarity score for ranking the results.<br /> <br />

### <ins><b>search_title<b><ins> <br />
We count for each Doc how many distinct words from the query are in it's title and sorting by that value. <br /> <br />

### <ins><b>get_pagerank<b><ins> <br />
Pulling the pre-calculated pagerank value from the storage. <br /> <br />

### <ins><b>get_pageview<b><ins> <br />
Pulling the pre-calculated pageviews value from the storage. <br /> <br />

### <ins><b>search_anchor<b><ins> <br />
Using the same logic of the search_title function.



