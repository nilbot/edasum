from collections import *
from dataframe import *

import os
import re
import nltk
import pickle
import math
import glob

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

    def __init__(self):
        self._idm = None
        self._ism = None
        self._world = None
        self._world_words = None
        self._world_words_set = None
        self._world_tf = None
        self._wmd = None
        self._wmd_bycol = None
        self._wmd_byrow = None
        self._idf = None

    def build_internal(self, document_set, remove_stopwords=False):
        """document_set is set or list of (review_id, review_text) tuple.
        """
        # internal document mapping, implicitly index is the internal id, same for ism (sentences)
        self._idm = [document for document in document_set]
        self.save_attr(self._idm, "idm")
        self._ism = [(doc_id, sentence)
                     for doc_id, doc in enumerate(self._idm)
                     for sentence in nltk.sent_tokenize(doc[1])]
        self.save_attr(self._ism, "ism")
        # build world (entire genre), and tf for genre
        self.build_world()
        self.save_attr(self._world, "world")

        self.build_world_tf(remove_stopwords)
        self.save_attr(self._world_words, "world_words")
        self.save_attr(self._world_tf, "world_tf")

        # build world document matrix based on built words
        self.prepare_for_idf()
        self.build_idf_only()
        self.save_attr(self._idf, "idf")

    def save_attr(self, attr, attr_str):
        file_to_be_saved = open("dataset/{0}.pkl".format(attr_str), 'wb')
        if any(x in ["matrix", "wmd"] for x in attr_str):
            np.save(file_to_be_saved, attr)
        else:
            pickle.dump(attr, file_to_be_saved)
        file_to_be_saved.close()

    def load_attr(self, attr_str):
        if attr_str in self.valid_attr_names():
            filename = "dataset/{0}.pkl".format(attr_str)
            if not os.path.isfile(filename):
                raise ValueError('{0}.pkl file not found!'.format(attr_str))
            obj = open(filename, 'rb')
            if any(x in ["wmd"] for x in attr_str):
                setattr(self, "_" + attr_str, np.load(obj))
            else:
                setattr(self, "_" + attr_str, pickle.load(obj))
            obj.close()

    def load_internal(self):
        """
        local internal dataset for further production uses
        """
        self.load_attr("idm")

        self.load_attr("ism")

        self.load_attr("world")

        self.load_attr("world_words")

        self.load_attr("world_tf")

        self.load_attr("wmd")

        self.load_attr("wmd_bycol")

        self.load_attr("wmd_byrow")

        self.load_attr("idf")

    """
    getter methods
    """

    def valid_attr_names(self):
        "return set of names of valid class attributes"
        return [
            "idm",
            "ism",
            "world",
            "world_words",
            "world_tf",
            "wmd",
            "wmd_bycol",
            "wmd_byrow",
            "idf",
        ]

    def get_attr(self, string):
        if string in self.valid_attr_names():
            return getattr(self, '_' + string)

    def world_words_set(self):
        if self._world_words_set is not None and len(
                self._world_words_set) != 0:
            return self._world_words_set
        elif self._world_words is not None and len(self._world_words) > 1:
            self._world_words_set = sorted(list(set(self._world_words)))
            return self._world_words_set
        else:
            self.load_attr("world_words")

    def wmd(self):
        return self._wmd

    def wmd_bycol(self):
        return self._wmd_bycol

    def wmd_byrow(self):
        return self._wmd_byrow

    """
    builder methods
    """

    def build_world(self):
        self._world = " ".join([text[1] for text in self._idm])
        self._world = " ".join(self._world.split())

        preprocessed = self.preprocess(self._world)

        self._world = preprocessed

    def build_world_words(self, remove_stopwords):
        self._world_words = self.tokenize(self.get_attr('word'))

        if remove_stopwords:
            self._world_words = self.nonstop(self._world_words)

    def build_world_tf(self, remove_stopwords):
        self.build_world_words(remove_stopwords)
        """
        this is worlds word hash
        """
        from collections import Counter
        self._world_tf = Counter(self.get_attr('world_words'))

    def build_idf(self, remove_stopwords):
        print("under construction.")
        # self._idf = defaultdict()
        # for i in range(N):
        #     word = self.world_words_set()[i]
        #     n_i = self._world_words_document_matrix.getcol(i).count_nonzero()
        #     self._idf[word] = math.log(m / float(n_i), 10)

    """
    utility methods
    """

    def tokenize(self, text):
        return nltk.regexp_tokenize(text, pattern)

    def preprocess(self, text):
        text = text.lower()

        # global static look up table for contraction must be present
        replace_contraction = re.compile(r'\b(' + '|'.join(knowledge.keys()) +
                                         r')\b')
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

    def prepare_for_idf(self, remove_stopwords=True):
        print("at least the rest works")
        n = len(self.world_words_set())
        m = len(self._idm)

        word_map2index = defaultdict()

        for i in range(n):
            word = self.world_words_set()[i]
            word_map2index[word] = i

        self._wmd = sparse.dok_matrix((m, n), dtype=np.int)

        for doc_index, doc_tuple in enumerate(self._idm):
            tokens = self.tokenize(self.preprocess(doc_tuple[1]))
            if remove_stopwords:
                tokens = self.nonstop(tokens)
            local_tf = Counter(tokens)

            for token in set(tokens):
                word_index = word_map2index[token]
                freq = local_tf[token]
                self._wmd[doc_index, word_index] = freq

        self.save_attr(self._wmd, "wmd")

        self._wmd_bycol = self._wmd.tocsc(True)
        self._wmd_byrow = self._wmd.tocsr(True)

        print("at least by col and by row is generated")

        self.save_attr(self._wmd_bycol, "wmd_bycol")
        self.save_attr(self._wmd_byrow, "wmd_byrow")

        print("saving done.")

    def build_idf_only(self):
        n = len(self.world_words_set())
        m = len(self._idm)
        self._idf = defaultdict()
        for i in range(n):
            word = self.world_words_set()[i]
            n_i = self._wmd_bycol.getcol(i).count_nonzero()
            self._idf[word] = math.log(m / float(n_i), 10)


def sample_result(hotel_id, period):
    "sample summarisation result using lexrank"
    tb = TSBase()
    tb.load_internal()


def load_existing():
    DOC_PREFIX = 'dataset/text/documents/raw'
    PICKLE_PREFIX = 'dataset'
    PICKLE_EXT = '.pkl'
    res = TSBase()
    pickles = glob.glob(PICKLE_PREFIX + "/*" + PICKLE_EXT)

    for p in pickles:
        attr = os.path.splitext(os.path.basename(p))[0]
        print("attr name read from dataset folder: {0}".format(attr))

        if attr in res.valid_attr_names():
            print("attr validated inside dict: {0}".format(attr))
            res.load_attr(attr)
            print("obj loaded with length {0}".format(len(res.get_attr(attr))))

    error = False
    for name in res.valid_attr_names():
        if res.get_attr(attr) is None:
            print("attr {0} is None, check constructor".format(name))
            error = True
        if len(res.get_attr(attr)) == 0:
            print("attr {0} not loaded, check load_existing".format(name))
            error = True

    print("world_words: len{0}".format(len(res._world_words)))

    if error:
        print("failed")
        exit(1)
    print("all done.")
    return res


if __name__ == '__main__':
    tsbase = load_existing()
    # if not hasattr(tsbase, "_world_words_document_matrix"):
    #     print('need matrix to calculate idf')
    #     exit(1)
    if tsbase._idf is None:
        tsbase.prepare_for_idf()
