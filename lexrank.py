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
