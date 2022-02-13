from inverted_index_gcp import *
from collections import defaultdict
import numpy as np
from contextlib import closing
from nltk.corpus import stopwords 
import re


class CalcsNpreps:

    def __init__(self):
        self.TUPLE_SIZE = 6
        self.tf = defaultdict(list)
        self.idf = defaultdict(int)
        self.cosine_sim = defaultdict(list)

    def parse_query(self, query):
        english_stopwords = frozenset(stopwords.words('english'))
        RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        tokens = [token.group() for token in RE_WORD.finditer(query.lower())]
        tokens = [token for token in tokens if token not in english_stopwords]
        new_query = ' '.join(tokens)
        return new_query

    def read_posting_list(self, inverted_index, w):
        with closing(MultiFileReader()) as reader:
            locs = inverted_index.posting_locs[w]
            for i, tup in enumerate(locs):
                locs[i] = tuple(list([tup[0], tup[1]]))
            b = reader.read(locs, inverted_index.df[w] * self.TUPLE_SIZE)
            posting_list = []
            for i in range(inverted_index.df[w]):
                doc_id = int.from_bytes(b[i * self.TUPLE_SIZE:i * self.TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * self.TUPLE_SIZE + 4:(i + 1) * self.TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    def update_tf_idf(self, query, inverted_index, n):
        for token in query.split():
            pl = self.read_posting_list(inverted_index, token)
            df = len(pl)
            self.idf[token] = np.log2((n+1)/(df+1))
            tf_vals = defaultdict(int)
            for id_tf in pl:
                did = id_tf[0]
                tf = id_tf[1]
                tf_vals[did] = tf/inverted_index.d_len[did]
            self.tf[token] = list(tf_vals.items())

    def calc_cosine_similarity(self, query, inverted_index):
        query_len = len(query.split())
        tf_idf = defaultdict(float)
        docs_len = defaultdict(int)
        for word in self.idf:
            for id_tf in self.tf[word]:
                did = id_tf[0]
                tf = id_tf[1]
                idf = self.idf[word]
                dl = inverted_index.d_len[did]
                tf_idf[did] += tf*idf
                docs_len[did] = dl
        for did in docs_len:
            tf_idf[did] = tf_idf[did]/(docs_len[did]*query_len)
        # update the cosine score for the query
        self.cosine_sim[query] = sorted(tf_idf.items(), key=lambda item: item[1], reverse=True)


