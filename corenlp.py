text = (
'Pusheen and Smitha walked along the beach. '
  'Pusheen wanted to surf, but fell off the surfboard.'
)
from pycorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP('http://localhost:51241')

output = nlp.annotate(text, properties={
    'annotators':'tokenize,ssplit,pos,lemma,ner,parse,mention,coref',
    'coref.algorithm':'statistical',
    'outputFormat':'json'
})
import pprint
pp = pprint.PrettyPrinter(depth=6)
pp.pprint(output)
