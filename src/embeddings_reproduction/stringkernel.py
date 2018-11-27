import itertools
from collections import Counter
from embeddings_reproduction.gpk import BaseKernel
import numpy as np


class MismatchKernel(BaseKernel):

    """
    A mismatch kmer kernel as described by Leslie et al. here:
    https://academic.oup.com/bioinformatics/article/20/4/467/192308

    Attributes:
        k (int): size of the kmers
        A (list): allowed alphabet. Elements must be strings.
        m (int): allowed mismatches
    """

    def __init__(self, k, A, m):
        """ Instantiate a MismatchKernel.

        Parameters:
            k (int): size of the kmers
            A (list): allowed alphabet. Elements must be strings.
            m (int): allowed mismatches
        """
        self.k = k
        self.A = A
        self.m = m
        self.A_to_num = {a:i for i, a in enumerate(A)}
        self.num_to_A = {i:a for i, a in enumerate(A)}
        self.nums = list(range(len(A)))
        self.nodes = self.make_kmer_tree(k, self.nums)
        self._n_hypers = 1
        self._saved = None
        return

    def make_kmer_tree(self, k, nums):
        """ Return a list representing the kmer tree."""
        nodes = [(np.array([]), [])]
        for it in range(k):
            new_nodes = []
            count = 0
            for i, node in enumerate(nodes):
                n, e = node
                if len(n) < it:
                    continue
                for a in nums:
                    count += 1
                    new_node = (np.append(n, a), [])
                    new_nodes.append(new_node)
                    nodes[i][1].append(len(nodes) + count - 1)
            nodes += new_nodes
        return nodes

    def prune(self, candidates, mutations, prefix):
        """
        Candidates are indices to kmer candidates,
        mutations are corresponding mutation counts
        prefix is kmer as vector
        """
        L = len(prefix)
        if L == 0:
            return candidates, mutations
        mutant = self.observed[candidates][:, L - 1] != prefix[-1]
        mutations[candidates] += mutant
        keep_me = mutations <= self.m
        candidates = candidates[keep_me[candidates]]
        return candidates, mutations

    def fit(self, seqs):
        """ Precompute the kernel for a set of sequences."""
        self._saved = self.cov(seqs1=seqs, seqs2=seqs)
        return self._n_hypers

    def cov(self, seqs1=None, seqs2=None, hypers=(1.0,)):
        """Calculate the mismatch string kernel.

        If no sequences given, then uses precomputed kernel.

        Parameters:
            seqs1 (list): list of strings
            seqs2 (list): list of strings
            hypers (iterable): the sigma value

        Returns:
            K (np.ndarray): n1 x n2 normalized mismatch string kernel
        """
        if seqs1 is None and seqs2 is None:
            return self._saved * hypers[0]
        if not isinstance(seqs1, list):
            seqs1 = list(seqs1)
        if not isinstance(seqs2, list):
            seqs2 = list(seqs2)
        # Break seqs into kmers
        kmers1 = [[seq[i:i + self.k] for i in range(len(seq) - self.k + 1)] for seq in seqs1]
        kmers2 = [[seq[i:i + self.k] for i in range(len(seq) - self.k + 1)] for seq in seqs2]
        # Get all observed kmers
        self.observed = sorted(set(itertools.chain.from_iterable(kmers1 + kmers2)))
        # Get count of each observed kmer for each sequence
        self.X1 = np.zeros((len(seqs1), len(self.observed)))
        self.X2 = np.zeros((len(seqs2), len(self.observed)))
        kmer_counts1 = [Counter(kmer) for kmer in kmers1]
        kmer_counts2 = [Counter(kmer) for kmer in kmers2]
        for j, obs in enumerate(self.observed):
            for i, counts in enumerate(kmer_counts1):
                self.X1[i, j] = counts[obs]
            for i, counts in enumerate(kmer_counts2):
                self.X2[i, j] = counts[obs]
        # Convert observed kmers to an array
        self.observed = np.array([[self.A_to_num[a] for a in obs] for obs in self.observed])
        # Create the covariance matrix
        self.K = np.zeros((len(seqs1), len(seqs2)))
        # Create the variance matrices
        self.K11 = np.zeros((len(seqs1), 1))
        self.K22 = np.zeros((len(seqs2), 1))
        # Initialize the mutation counts
        mutations = np.zeros(len(self.observed))
        # Initialize the candidate indices
        candidates = np.arange(len(mutations))
        # Populate K
        self.dft(candidates.copy(), mutations.copy(), 0)
        # Normalize K
        self.K /= np.sqrt(self.K11)
        self.K /= np.sqrt(self.K22.T)
        self.K *= hypers[0]
        return self.K

    def dft(self, candidates, mutations, ind):
        """ Depth first traversal of kmer tree to calculate K."""
        kmer = self.nodes[ind][0]
        candidates, mutations = self.prune(candidates, mutations, kmer)
        if len(candidates) == 0:
            return
        if len(self.nodes[ind][1]) == 0:
            Y = np.zeros((len(self.observed), 1))
            Y[candidates, 0] = 1
            n_alphas1 = self.X1 @ Y
            n_alphas2 = self.X2 @ Y
            self.K += n_alphas1 @ n_alphas2.T
            self.K11 += n_alphas1 ** 2
            self.K22 += n_alphas2 ** 2
        for e in self.nodes[ind][1]:
            self.dft(candidates.copy(), mutations.copy(), e)
