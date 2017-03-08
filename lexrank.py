def tf(word, sentence):
    n = len(sentence)
    count = 0
    for w in sentence:
        if w == word:
            count += 1
    return count / n


def idf_document(word, cluster):
    N = len(cluster)
    count = 0
    for doc in cluster:
        if word in doc:
            count += 1
    import math
    return math.log(N / count)


def idf_cosine_sim(x, y, world):
    # x and y are N dimensional vector where N is size of language (big)
    big_n = len(x)
    d = 0
    for word in [word for word in x if word in y]:
        a = tf(word, x)
        b = tf(word, y)
        c = idf_document(word, world)
        d += a * b * (c * c)

    import math
    e = 0
    f = 0
    for word in x:
        e0 = tf(word, x) * idf_document(word, world)
        e += e0 * e0
    for word in y:
        f0 = tf(word, y) * idf_document(word, world)
        f += f0 * f0

    g = math.sqrt(e) * math.sqrt(f)
    return d / g


def lexrank_world(world):

    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words

    LANGUAGE = 'english'
    SENTENCES_COUNT = 5
    parser = PlaintextParser.from_string(world, Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    for sent in summarizer(parser.document, SENTENCES_COUNT):
        print(sent)

# preliminaries

from collections import *
from dataframe import *

import os
import re
import nltk
import pickle
import math
    
from nltk.corpus import stopwords
import numpy as np
from scipy import sparse

# %load_ext memory_profiler
# %load_ext line_profiler


# language lookup table
knowledge = {}
knowledge["haven't"] = "have not"
knowledge["hasn't"] = "has not"
knowledge["hadn't"] = "had not"
knowledge["doesn't"] = "does not"
knowledge["don't"] = "do not"
knowledge["didn't"] = "did not"
knowledge["couldn't"] = "could not"
knowledge["mustn't"] = "must not"
knowledge["can't"] = "can not"
knowledge["hadn't"] = "had not"
knowledge["won't"] = "will not"
knowledge["wouldn't"] = "would not"
knowledge["i'm"] = "i am"
knowledge["it's"] = "it is"
knowledge["let's"] = "let us"

# custom regex tokenizer pattern
# caveat: orginal inclues
"""
| [][.,;"'?():_`-]    # these are separate tokens; includes ], [
"""
pattern = r'''(?x)          # set flag to allow verbose regexps
    (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
  | \w+(?:-\w+)*        # words with optional internal hyphens
  | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
  | \.\.\.              # ellipsis
  | \w+(?:'\w+)*        # words that have ' in between
'''

class TSBase(object):
    """
    """

    
    def build_internal(self, document_set, remove_stopwords=False):
        """
        document_set is set or list of (review_id, review_text) tuple.
        """
        # internal document mapping, implicitly index is the internal id, same for ism (sentences)
        self._idm = [document for document in document_set]
        self.save_attr(self._idm, "idm")
        self._ism = [(doc_id, sentence) for doc_id, doc in enumerate(self._idm) for sentence in nltk.sent_tokenize(doc[1]) ]
        self.save_attr(self._ism, "ism")
        # build world (entire genre), and tf for genre
        self.build_world()
        self.save_attr(self._world, "world")
        
        self.build_world_tf(remove_stopwords)
        self.save_attr(self._world_words, "world_words")
        self.save_attr(self._world_tf, "world_tf")

        # build world document matrix based on built words
        self.build_idf(remove_stopwords)
        self.save_attr(self._idf, "idf")

    def save_attr(self, attr, attr_str):
        file = open("dataset/{0}.pkl".format(attr_str), 'wb')
        pickle.dump(attr, file)
        file.close()
        
    def load_attr(self, attr_str):
        obj = open("dataset/{0}.pkl".format(attr_str), 'rb')
        attr = pickle.load(obj)
        obj.close()
        return attr
        
    def load_internal():
        self._idm = self.load_attr("idm")
        
        self._ism = self.load_attr("ism")
        
        self._world = self.load_attr("world")
        
        self._world_words = self.load_attr("world_words")
        
        self._world_tf = self.load_attr("world_tf")
        
        self._world_words_document_matrix = self.load_attr("world_words_document_matrix")
        
        self._idf = self.load_attr("idf")


    """
    getter methods
    """
    def world(self):
        return self._world
    
    
    def world_words(self):
        return self._world_words
    
    
    def world_words_set(self):
        if hasattr(self, '_world_words_set') and len(self._world_words_set) != 0:
            return self._world_words_set
        else:
            self._world_words_set = sorted(list(set(self.world_words())))
            return self._world_words_set
        
        
    def world_tf(self):
        return self._world_tf
    
    def idf(self):
        if hasattr(self, '_idf') and len(self._idf) != 0:
            return self._idf
        return None


    """
    builder methods
    """
    def build_world(self):
        self._world = " ".join([text[1] for text in self._idm])
        self._world = " ".join(self._world.split())
        
        preprocessed = self.preprocess(self._world)
        
        self._world = preprocessed
    
    def build_world_words(self, remove_stopwords):
        self._world_words = self.tokenize(self.world())

        if remove_stopwords:
            self._world_words = self.nonstop(self._world_words)
    
    
    def build_world_tf(self, remove_stopwords):
        self.build_world_words(remove_stopwords)
        """
        this is worlds word hash
        """
        from collections import Counter
        self._world_tf = Counter(self.world_words())

    
    def build_idf(self, remove_stopwords):
        N = len(self.world_words_set())
        m = len(self._idm)
        
        word_map2index = defaultdict()
        
        for i in range(N):
            word = self.world_words_set()[i]
            word_map2index[word] = i
        
        self._world_words_document_matrix = sparse.dok_matrix((m,N),dtype=np.int)
        
        for doc_index,doc_tuple in enumerate(self._idm):
            tokens = self.tokenize(self.preprocess(doc_tuple[1]))
            if remove_stopwords:
                tokens = self.nonstop(tokens)
            local_tf = Counter(tokens)

            for token in set(tokens):
                word_index = word_map2index[token]
                freq = local_tf[token]
                self._world_words_document_matrix[doc_index, word_index] = freq
                
        self.save_attr(self._world_words_document_matrix, "world_words_document_matrix")

        self._idf = defaultdict()
        for i in range(N):
            word = self.world_words_set()[i]
            n_i = self._world_words_document_matrix.getcol(i).count_nonzero()
            self._idf[word] = math.log( m / float(n_i) , 10)

    
    
    """
    utility methods
    """
    def tokenize(self,text):
        return nltk.regexp_tokenize(text, pattern)

    
    def preprocess(self, text):
        text = text.lower()
        
        # global static look up table for contraction must be present
        replace_contraction = re.compile(r'\b(' + '|'.join(knowledge.keys()) + r')\b')
        return replace_contraction.sub(lambda x: knowledge[x.group()], text)

    
    def nonstop(self, tokens):
        cachedStopWords = stopwords.words("english")
        return [token for token in tokens if token not in cachedStopWords]
    
        
    def hashing_vectorizer(self, text, N):
        """term frequency (local)"""
        x = np.zeros(N, dtype=np.int32)
        words = self.tokenize(text)
        for w in words:
            h = hash(w)
            x[h % N] += 1
        return x

    
def build_tsbase(small_test = False, value_test = False):
    # test for base

    DOC_PREFIX = 'dataset/text/documents/raw'
    if small_test:
        txts = os.listdir(DOC_PREFIX)[100000:100100]
    else:
        txts = os.listdir(DOC_PREFIX) # all, caution, should use parallelism to speed up

    docs = deque()
    for t in txts:
        with open(os.path.join(DOC_PREFIX,t), 'r') as f:
            raw = f.read()
            doc_id = os.path.splitext(os.path.basename(f.name))[0]
            if small_test and value_test:
                print(raw,"\n\n")
            docs.append((doc_id,raw))
    if small_test:
        import cProfile
        cProfile.run( 'tsbase = TSBase(); tsbase.build_internal(docs, True)' )
    else:
        tsbase = TSBase()
        tsbase.build_internal(docs, True)


if __name__ == '__main__':
    build_tsbase()

