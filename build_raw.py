import os
from collections import deque
from lexrankop import TSBase


def build_tsbase(small_test=False, value_test=False):
    # test for base
    DOC_PREFIX = 'dataset/text/documents/raw'
    if small_test:
        txts = os.listdir(DOC_PREFIX)[100000:100100]
    else:
        txts = os.listdir(
            DOC_PREFIX)  # all, caution, should use parallelism to speed up

    docs = deque()
    n = len(txts)
    print('collecting {0} documents...'.format(n))
    counter = 1
    for t in txts:
        with open(os.path.join(DOC_PREFIX, t), 'r') as f:
            raw = f.read()
            doc_id = os.path.splitext(os.path.basename(f.name))[0]
            if small_test and value_test:
                print(raw, "\n\n")
            docs.append((doc_id, raw))
        counter = counter + 1
        if counter % 2000 == 0:
            print("read in number {0}th document.".format(counter))
    if small_test:
        import cProfile
        cProfile.run('tsbase = TSBase(); tsbase.build_internal(docs, True)')
    else:
        tb = TSBase()
        tb.build_internal(docs)


if __name__ == '__main__':
    build_tsbase()
