import glob
import os
from lexrankop import TSBase
from collections import deque

docs = deque()
for doc_name in glob.glob('dataset/text/documents/raw/review_14938*'):
	with open(doc_name, 'r') as f:
		raw = f.read()
		doc_id = os.path.splitext(os.path.basename(f.name))[0]
		docs.append((doc_id, raw))




t_obj = TSBase()

t_obj.build_internal(docs)

print(t_obj.get_attr('world_words'))
print(t_obj.world_words_set())
print(t_obj.get_attr('world_tf'))
print(t_obj.get_attr('idf'))