import sys
import getopt
import codecs
import collections
import numpy
import networkx
import nltk
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import pairwise_distances



def lexrank(sentences, continuous=False, sim_threshold=0.1, alpha=0.9,
            use_divrank=False, divrank_alpha=0.25):
    ranker_params = {'max_iter': 1000}

    ranker = networkx.pagerank_scipy
    ranker_params['alpha'] = alpha

    graph = networkx.DiGraph()

    # sentence -> tf
    sent_tf_list = []
    wnl = nltk.WordNetLemmatizer()
    for sent in sentences:
        words = [ wnl.lemmatize(w) for w in nltk.word_tokenize(sent)]
        tf = collections.Counter(words)
        sent_tf_list.append(tf)

    sent_vectorizer = DictVectorizer(sparse=True)
    sent_vecs = sent_vectorizer.fit_transform(sent_tf_list)

    # compute similarities between senteces
    sim_mat = 1 - pairwise_distances(sent_vecs, sent_vecs, metric='cosine')

    if continuous:
        linked_rows, linked_cols = numpy.where(sim_mat > 0)
    else:
        linked_rows, linked_cols = numpy.where(sim_mat >= sim_threshold)

    # create similarity graph
    graph.add_nodes_from(range(sent_vecs.shape[0]))
    for i, j in zip(linked_rows, linked_cols):
        if i == j:
            continue
        weight = sim_mat[i,j] if continuous else 1.0
        graph.add_edge(i, j, {'weight': weight})

    scores = ranker(graph, **ranker_params)
    return scores, sim_mat


def summarize(text, sent_limit=None, char_limit=None, imp_require=None,
              debug=False, **lexrank_params):

    debug_info = {}
    
    sentences = list( nltk.sent_tokenize( " ".join( text )  )  )
    scores, sim_mat = lexrank(sentences, **lexrank_params)
    sum_scores = sum(scores.values())
    acc_scores = 0.0
    indexes = set()
    num_sent, num_char = 0, 0
    for i in sorted(scores, key=lambda i: scores[i], reverse=True):
        num_sent += 1
        num_char += len(sentences[i])
        if sent_limit is not None and num_sent > sent_limit:
            break
        if char_limit is not None and num_char > char_limit:
            break
        if imp_require is not None and acc_scores / sum_scores >= imp_require:
            break
        indexes.add(i)
        acc_scores += scores[i]

    if len(indexes) > 0:
        summary_sents = [sentences[i] for i in sorted(indexes)]
    else:
        summary_sents = sentences

    if debug:
        debug_info.update({
            'sentences': sentences, 'scores': scores
        })

    return summary_sents, debug_info





# from sumy.parsers.plaintext import PlaintextParser
# from sumy.nlp.tokenizers import Tokenizer
# from sumy.summarizers.lex_rank import LexRankSummarizer as Summarizer
# from sumy.nlp.stemmers import Stemmer
# from sumy.utils import get_stop_words


# def lexrank_sumy(docs):
#     LANGUAGE = 'english'
#     SENTENCES_COUNT = 5
#     parser = PlaintextParser.from_string(docs, Tokenizer(LANGUAGE))
#     stemmer = Stemmer(LANGUAGE)

#     summarizer = Summarizer(stemmer)
#     summarizer.stop_words = get_stop_words(LANGUAGE)

#     for sent in summarizer(parser.document, SENTENCES_COUNT):
#         print(sent)






# some implementations

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

