import random
from itertools import chain

from gensim.models import doc2vec
import numpy as np


def seq_to_kmers(seq, k=3, overlap=False, **kwargs):
    """ Divide a string into a list of kmer strings.

    Parameters:
        seq (string)
        k (int), default 3
        overlap (Boolean), default False

    Returns:
        List containing 1 list of kmers (overlap=True) or k lists of
            kmers (overlap=False)
    """
    N = len(seq)
    if overlap:
        return [[seq[i:i+k] for i in range(N - k + 1)]]
    else:
        return [[seq[i:i+k] for i in range(j, N - k + 1, k)]
                for j in range(k)]


def seqs_to_kmers(seqs, k=3, overlap=False, **kwargs):
    """Divide a list of sequences into kmers.

    Parameters:
        seqs (iterable) containing strings
        k (int), default 3
        overlap (Boolean), default False

    Returns:
        List of lists of kmers
    """
    as_kmers = []
    for seq in seqs:
        as_kmers += seq_to_kmers(seq, k=k, overlap=overlap)
    return as_kmers


def randomize_seqs(seqs, method=0):
    """ Randomize a set of sequences.

    Parameters:
        seqs (iterable) containing sequences
        method (int)
            0: Scramble each individual sequence.
            1: Generate sequences that match original lengths by drawing
                uniformly from alphabet.
            2: Generate sequences that match original lengths by drawing
                from distribution of characters in seqs.

    Returns:
        List containing randomized sequences
    """
    if method == 0:
        return [''.join(random.sample(seq, k=len(seq))) for seq in seqs]
    else:
        alpha = []
        for seq in seqs:
            alpha += list(seq)
        alpha = list(set(alpha))
        if method == 1:
                    return [''.join(np.random.choice(alpha, size=len(seq)))
                            for seq in seqs]
        elif method == 2:
            P = np.zeros(25)
            alpha_dict = {a:i for i, a in enumerate(alpha)}
            for seq in seqs:
                for s in seq:
                    P[alpha_dict[s]] += 1
            return [''.join(np.random.choice(alpha, size=len(seq), p=P))
                    for seq in seqs]


def _combine(vectors, k):
    """ Combine vectors. """
    embeds = np.zeros((vectors.shape[0] // k, vectors.shape[1]))
    for i in range(k):
        embeds += vectors[i::k, :]
    return embeds


def _normalize(vectors):
    """ Normalize vectors (in rows) to length 1. """
    norms = np.sqrt(np.sum(vectors ** 2, axis=1))
    vectors /= norms.reshape((len(norms), 1))
    return vectors


def get_embeddings(doc2vec_file, seqs, k=3, overlap=False, norm=True, steps=5):
    """ Infer embeddings in one pass using a gensim doc2vec model.

    Parameters:
        doc2vec_file (str): file pointing to saved doc2vec model
        seqs (iterable): sequences to infer
        k (int) default 3
        overlap (Boolean) default False
        norm (Boolean) default True
        steps (int): number of steps during inference. Default 5.

    Returns:
        numpy ndarray where each row is the embedding for one sequence.
    """
    model = doc2vec.Doc2Vec.load(doc2vec_file)
    as_kmers = seqs_to_kmers(seqs, k=k, overlap=overlap)
    vectors = np.array([model.infer_vector(doc, steps=steps)
                        for doc in as_kmers])
    if overlap:
        embeds = vectors
    else:
        embeds = _combine(vectors, k)
    if norm:
        embeds = _normalize(embeds)
    return embeds


def get_embeddings_new(doc2vec_file, seqs, k=3, overlap=False, passes=100):
    """ Infer embeddings by averaging passes using a gensim doc2vec model.

    Make passes through the sequences, normalizing and averaging the results
    after every pass.

    Parameters:
        doc2vec_file (str): file pointing to saved doc2vec model
        seqs (iterable): sequences to infer
        k (int) default 3
        overlap (Boolean) default False
        passes (int): number of passes during inference. Default 100.

    Returns:
        numpy ndarray where each row is the embedding for one sequence.
    """
    model = doc2vec.Doc2Vec.load(doc2vec_file)
    as_kmers = seqs_to_kmers(seqs, k=k, overlap=overlap)
    old_embeds = None
    order = [i for i in range(len(seqs))]
    for p in range(passes):
        random.shuffle(order)
        if not overlap and k > 1:
            shuffled_inds = list(chain.from_iterable(([k*i + kk for
                                                       kk in range(k)]
                                 for i in order)))
        else:
            shuffled_inds = order
        shuffled_kmers = [as_kmers[i] for i in shuffled_inds]
        vectors = np.array([model.infer_vector(doc, steps=1)
                            for doc in shuffled_kmers])
        if overlap:
            embeds = vectors
        else:
            embeds = _combine(vectors, k)
        embeds = _normalize(embeds)
        unshuffle_order = [order.index(i) for i in range(len(order))]
        embeds = embeds[unshuffle_order]
        if old_embeds is not None:
            old_embeds = (p * old_embeds + embeds) / (p + 1)
            old_embeds = _normalize(old_embeds)
        else:
            old_embeds = embeds[:]
    return old_embeds


def get_seqs(df):
    """Extract items in df.sequence without gap characters ('-'). """
    seqs = df.sequence.values
    return [''.join([s for s in seq if s != '-']) for seq in seqs]


class Corpus(object):
    """ An iteratable for training seq2vec models. """

    def __init__(self, df, kmer_hypers):
        self.df = df
        self.kmer_hypers = kmer_hypers

    def __iter__(self):
        for doc in self.get_documents():
            yield doc

    def df_to_kmers(self):
        for seq in self.df.sequence.values:
            kmers = seq_to_kmers(seq, **self.kmer_hypers)
            if self.kmer_hypers['overlap']:
                yield kmers
            else:
                for km in kmers:
                    yield km

    def get_documents(self):
        if self.kmer_hypers['merge']:
            return (doc2vec.TaggedDocument(doc, [i // self.kmer_hypers['k']])
                    for i, doc in enumerate(self.df_to_kmers()))
        return (doc2vec.TaggedDocument(doc, [i]) for i,
                doc in enumerate(self.df_to_kmers()))


if __name__ == '__main__':
    get_embeddings_new('scrambled_3_64_20_5_doc2vec.pkl', ['ABCFFFFFFFFFFFF',
                                                           'EFGHQWERRTTUIIO'])
