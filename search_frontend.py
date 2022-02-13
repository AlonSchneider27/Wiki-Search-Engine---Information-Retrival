from flask import Flask, request, jsonify
import backend
from inverted_index_gcp import *
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

#---------------------------------------------------------------------------------
# LOADING GLOBAL DATA
##########################
#GCP ADDRESS /home/alonshn/postings_gcp/
#COLAB ADDRESS /content/gdrive/MyDrive/postings_gcp/
data_address = '/home/alonshn/postings_gcp/'

global id_title_dict  # ID TITLE DOCT
pv_clean = data_address+'id_title_dict.pickle'
with open(pv_clean, 'rb') as f:
    id_title_dict = dict(pickle.loads(f.read()))

global id_title_len  # ID TITLE LEN DOCT
pv_clean = data_address+'id_title_len.pkl'
with open(pv_clean, 'rb') as f:
    id_title_len = dict(pickle.loads(f.read()))

global text_index  # BODY INVERTED INDEX
inverted_index_text = InvertedIndex()
text_index = inverted_index_text.read_index(data_address, 'text_index_2')

global title_index  # TITLE INVERTED INDEX
inverted_index_title = InvertedIndex()
title_index = inverted_index_title.read_index(data_address, 'title_index_2')

global anchor_index  # ANCHOR INVERTED INDEX
inverted_index_anchor = InvertedIndex()
anchor_index = inverted_index_anchor.read_index(data_address, 'anchor_index_2')

# PAGERANK DATA
pv_clean = data_address+'page_rank_1.pkl'
global pageranks
with open(pv_clean, 'rb') as f:
    pageranks = dict(pickle.loads(f.read()))

# PAGEVIEWS DATA
global pageviews
pv_clean = data_address+'pageviews-202108-user.pkl'
with open(pv_clean, 'rb') as f:
    pageviews = dict(pickle.loads(f.read()))

#---------------------------------------------------------------------------------
# HELPER FUNCTION
##########################

def vals_for_dids(did_list, vals_obj):
    res = []
    for did in did_list:
        res.append(vals_obj[did])
    return res



@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''
    res = []
    utils = backend.CalcsNpreps()
    query_title = defaultdict(int)
    good_kind = ['NN', 'VB', 'NNS', 'VBN', 'JJ']

    query = request.args.get('query', '')
    query = utils.parse_query(query).split()
    for word in query:
        if word.endswith('es'):
            query.append(word[:-1])
            query.append(word[:-2])
        elif word.endswith('s'):
            query.append(word[:-1])
    query = ' '.join(query)
    for inf in [query]:
        result_word = []
        tokens = nltk.word_tokenize(inf)
        pos_tagged_tokens = nltk.pos_tag(tokens)
        for pos in pos_tagged_tokens:
            if pos[1] in good_kind:
                result_word.append(pos[0])
    query = result_word

    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for token in list(set(query)):
        try:
            pl = title_index.read_posting_list(token)
        except:
            continue
        for doc_id, tf in pl:
            query_title[doc_id] += 1

    tmp = {k: float(query_title[k] / id_title_len[k]) for k in query_title}
    tmp = list(tmp.items())
    tmp = sorted(tmp, key=lambda x: x[1], reverse=True)
    tmp = tmp[:300]
    tmp = [x[0] for x in tmp]
    for did in tmp:
        if did in pageviews:
            res.append((did, pageviews[did]))
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = list(map(lambda x: (x[0], id_title_dict[x[0]]), res))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each 
        element is a tuple (wiki_id, title).
    '''

    utils = backend.CalcsNpreps()
    res = []
    query = request.args.get('query', '')
    query = utils.parse_query(query)
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    utils.update_tf_idf(query, text_index, len(text_index.d_len.keys()))
    utils.calc_cosine_similarity(query, text_index)
    res = utils.cosine_sim
    res = list(res.items())[:100]
    res = res[0][1]
    res = list(map(lambda x: (x[0], id_title_dict[x[0]]), res))
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    utils = backend.CalcsNpreps()
    query_title = defaultdict(int)

    res = []
    query = request.args.get('query', '')
    query = utils.parse_query(query).split()
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for token in list(set(query)):
        try:
            pl = title_index.read_posting_list(token)
        except:
            continue
        for doc_id, tf in pl:
            query_title[doc_id] += 1
    res = list(query_title.items())
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = list(map(lambda x: (x[0], id_title_dict[x[0]]), res))
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''

    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    res = vals_for_dids(wiki_ids, pageranks)
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.
        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)

    # BEGIN SOLUTION
    res = vals_for_dids(wiki_ids, pageviews)
    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        For example, a document with a anchor text that matches two of the
        query words will be ranked before a document with anchor text that
        matches only one query term.

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''

    utils = backend.CalcsNpreps()
    query_title = defaultdict(int)

    res = []
    query = request.args.get('query', '')
    query = utils.parse_query(query).split()
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for token in list(set(query)):
        try:
            pl = anchor_index.read_posting_list(token)
        except:
            continue
        for doc_id, tf in pl:
            query_title[doc_id] += 1
    res = list(query_title.items())
    res = sorted(res, key=lambda x: x[1], reverse=True)
    res = list(map(lambda x: (x[0], id_title_dict[x[0]]), res))
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080)
 