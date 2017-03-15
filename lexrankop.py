from collections import defaultdict, Counter

import os
import nltk
import pickle
import math
import glob

from nltk.corpus import stopwords
import numpy as np
from scipy import sparse


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
        print('processed internal document mapping')

        self._ism = [(doc_id, sentence)
                     for doc_id, doc in enumerate(self._idm)
                     for sentence in nltk.sent_tokenize(doc[1])]
        self.save_attr(self._ism, "ism")
        print('processed internal sentence document mapping')

        self.build_world_words(remove_stopwords)
        self.save_attr(self._world_words, "world_words")
        print('processed internal lemmatised word tokens for our genre')

        self.build_world_tf()
        self.save_attr(self._world_tf, "world_tf")
        print('processed genre term frequency')

        # build world document matrix based on built words
        print('preparing datastructure for genre idf calculation')
        self.prepare_for_idf()
        self.build_idf_only()
        self.save_attr(self._idf, "idf")

    def save_attr(self, attr, attr_str):
        file_to_be_saved = open("dataset/{0}.pkl".format(attr_str), 'wb')
        if any(x in ["wmd"] for x in attr_str):
            if "by" in attr_str:
                # csr and csc
                np.savez(
                    file_to_be_saved,
                    data=attr.data,
                    indices=attr.indices,
                    indptr=attr.indptr,
                    shape=attr.shape)
            else:
                # coo (dok converted)
                np.savez(
                    file_to_be_saved,
                    data=attr.data,
                    row=attr.row,
                    col=attr.col,
                    shape=attr.shape)
        else:
            pickle.dump(attr, file_to_be_saved)
        file_to_be_saved.close()

    def load_attr(self, attr_str):
        if attr_str in self.valid_attr_names():
            filename = "dataset/{0}.pkl".format(attr_str)
            if not os.path.isfile(filename):
                raise ValueError('{0}.pkl file not found!'.format(attr_str))
            file_obj = open(filename, 'rb')
            if any(x in ["wmd"] for x in attr_str):
                loader = np.load(filename)
                if "bycol" in filename:
                    temp = sparse.csc_matrix(
                        (loader['data'], loader['indices'], loader['indptr']),
                        shape=loader['shape'])
                elif "byrow" in filename:
                    temp = sparse.csr_matrix(
                        (loader['data'], loader['indices'], loader['indptr']),
                        shape=loader['shape'])
                else:
                    temp = sparse.coo_matrix(
                        (loader['data'], (loader['row'], loader['col'])),
                        shape=loader['shape'])
                setattr(self, "_" + attr_str, temp)
            else:
                setattr(self, "_" + attr_str, pickle.load(file_obj))
            file_obj.close()

    def load_internal(self):
        """
        local internal dataset for further production uses
        """
        self.load_attr("idm")

        self.load_attr("ism")

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

    def build_world_words(self, remove_stopwords=False):
        listo = [self.tokenize(doc[1]) for doc in self._idm]
        wnl = nltk.WordNetLemmatizer()
        self._world_words = [
            wnl.lemmatize(item) for lst in listo for item in lst
        ]

        if remove_stopwords:
            self._world_words = self.nonstop(self._world_words)

    def build_world_tf(self):
        """
        this is worlds word hash
        """
        self._world_tf = Counter(self.get_attr('world_words'))

    """
    utility methods
    """

    def tokenize(self, text):
        return nltk.word_tokenize(text)

    def nonstop(self, tokens):
        cached_stopwords = stopwords.words("english")
        return [token for token in tokens if token not in cached_stopwords]

    def hashing_vectorizer(self, text, N):
        """term frequency (local)"""
        x = np.zeros(N, dtype=np.int32)
        words = self.tokenize(text)
        for w in words:
            h = hash(w)
            x[h % N] += 1
        return x

    def prepare_for_idf(self, remove_stopwords=False):
        print("at least the rest works")
        n = len(self.world_words_set())
        m = len(self._idm)

        wnl = nltk.WordNetLemmatizer()

        word_map2index = defaultdict()

        for i in range(n):
            word = self.world_words_set()[i]
            word_map2index[word] = i

        temp_wmd = sparse.dok_matrix((m, n), dtype=np.int)

        for doc_index, doc_tuple in enumerate(self._idm):
            tokens = [wnl.lemmatize(t) for t in self.tokenize(doc_tuple[1])]
            if remove_stopwords:
                tokens = self.nonstop(tokens)
            local_tf = Counter(tokens)

            for token in set(list(sorted(tokens))):
                try:
                    word_index = word_map2index[token]
                except KeyError:
                    print(
                        "key: {0} from doc_tuple {1}".format(token, doc_tuple))
                    exit(1)
                freq = local_tf[token]
                temp_wmd[doc_index, word_index] = freq

        self._wmd = temp_wmd.tocoo()
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
    # tb = TSBase()
    # tb.load_internal()
    # hotel_database = load_hotel_database()
    # docs_ids = hotel_database.get_docs(hotel_id).subset(period)
    # return tb.summarise(docs_ids)


def load_existing():
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
            if "wmd" in attr:
                print("obj loaded with length {0}".format(
                    res.get_attr(attr).shape[0]))
            else:
                print("obj loaded with length {0}".format(
                    len(res.get_attr(attr))))

    error = False
    for name in res.valid_attr_names():
        if res.get_attr(attr) is None:
            print("attr {0} is None, check constructor".format(name))
            error = True
        if len(res.get_attr(attr)) == 0:
            print("attr {0} not loaded, check load_existing".format(name))
            error = True
    if error:
        print("failed")
        exit(1)
    else:
        print("all done.")
        print("world_words len: {0}".format(len(res.get_attr('world_words'))))
        print("world_words_set len: {0}".format(len(res.world_words_set())))
        return res
