from scipy.spatial import distance
import numpy as np

def analogy(w1, w2, w3, vecs, vocab):
    u = vecs[vocab[w1]] - vecs[vocab[w2]] + vecs[vocab[w3]]
    sim = []
    for v in vecs:
        sim.append(distance.cosine(u, v))
    sim = np.array(sim)
    ind = np.argpartition(sim, 3)[:3 + 1]
    ind = ind[np.argsort(sim[ind])][1:]
    analogy = [(i, vocab.lookup_token(i), sim[i]) for i in ind]
    return analogy

def get_distance_matrix(wordvecs, metric):
    dist_matrix = distance.squareform(distance.pdist(wordvecs, metric))
    return dist_matrix

def get_k_similar_words(vocab, word, dist_matrix, k=10):
    idx = vocab[word]
    dists = dist_matrix[idx]
    ind = np.argpartition(dists, k)[:k+1]
    ind = ind[np.argsort(dists[ind])][1:]
    out = [(i, vocab.lookup_token(i), dists[i]) for i in ind]
    return out

def evaluate(vocab, vecs, tokens):
    dmat = get_distance_matrix(vecs, 'cosine')
    for word in tokens:
        print(word, [t[1] for t in get_k_similar_words(vocab, word, dmat)], "\n")