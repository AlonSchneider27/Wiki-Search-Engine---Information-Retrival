from contextlib import closing
from pathlib import Path
from collections import Counter, defaultdict
import pickle

#GCP ADDRESS /home/alonshn/postings_gcp/
#COLAB ADDRESS /content/gdrive/MyDrive/postings_gcp/
data_address = '/home/alonshn/postings_gcp/'

class InvertedIndex:
    def __init__(self, docs={}):
        self.d_len = defaultdict(int)  # stores the length of a document
        self.term_total = Counter()  # stores the total terms frequencies
        self.df = Counter()  # stores df value for each term
        self.posting_locs = defaultdict(list)

    def read_posting_list(self, w):
        tuple_size = 6
        with closing(MultiFileReader()) as reader:
            locs = self.posting_locs[w]
            b = reader.read(locs, self.df[w] * tuple_size)
            posting_list = []
            for i in range(self.df[w]):
                doc_id = int.from_bytes(b[i * tuple_size:i * tuple_size + 4], 'big')
                tf = int.from_bytes(b[i * tuple_size + 4:(i + 1) * tuple_size], 'big')
                posting_list.append((doc_id, tf))
            return posting_list

    @staticmethod
    def read_index(base_dir, name):
        with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
            return pickle.load(f)


class MultiFileReader:
    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes):
        block_size = 1999998
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                self._open_files[f_name] = open(data_address+f_name, 'rb')
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, block_size - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False





