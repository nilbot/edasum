## LexRank
### Sentence Centrality and Centroid
given a cluster (e.g. multi-document), a the main theme of this cluster can be
captured using inverse-document-frequency

idf of word $i$ is the log scale of 
number of documents over number of docs that has word $i$

$$ idf_i = log \frac{N}{n_i} $$

This formula implies that when if every document has too many occurances of a
certain word $i$, that means either the word is a stop word, the value of the 
fraction will be close to zero and the log scale will be a very very small 
value. On the other hand if every document has roughly exactly 1 occurance of 
certain word, that word might be the main theme of the cluster and a not a 
common word, perhaps even a professional word or a noun that has special 
meaning, then the value is going to be close to 1, the log scale is going to be
0. A "stand-out" word would be only occurred few times in the entire cluster, 
these words will give $idf$ value a high score.

One way to define centrality of a sentence is to calculate the centrality of 
the words it contains, and centrality of words is going to be looking at the 
center of _**pseudo document**_ consists of all important words.

I present a such construction in steps:

1. $tf(i)$ is the frequency of word $i$ in cluster $\mathbf{C}$
2. $idf(i)$ is the $idf$ value of word $i$ in the **genre** $\mathbf{G}$ that
 wraps the cluster $\mathbf{C}$
3. put all words $w$ that has $tf\times idf$ score higher than a threshold $t$
together as a pseudo document, and call it _**Centroid**_
4. examine sentences, count words that are from _**centroid**_, higher count 
sentences are considered _central_. The degree (count) of denotes the distance
towards _centroid_.

_LexRank_ extends with _prestige_ (centrality for undirected graphs) measuring.
Through defining a similarity between sentences in the bag-of-words model. 
($N$ stands for number of words in the entire genre, sometimes even language).
$$ \text{idf-modified-cosine}(x,y) = \frac{
	\sum_{w\in x,y}\text{tf}_{w,x}
	\text{tf}_{w,y}(\text{idf}_w)^2}
	{
		\sqrt{\sum_{x_i\in x}{(\text{tf}_{x_i,x}\text{idf}_{x_i})^2}}
	\times
		\sqrt{\sum_{y_i\in y}{(\text{tf}_{y_i,y}\text{idf}_{y_i})^2}}
	}
$$
where $\text{tf}_{w,s}$ is the number of occurrences of the word $w$ in the 
sentence $s$.