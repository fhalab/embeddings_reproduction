"""Kernel functions that calculate the covariance between two inputs."""

import numpy as np
import abc

from scipy.spatial import distance


class BaseKernel(abc.ABC):

    """ A Gaussian Process kernel.

       Attributes:
           hypers (list)
    """

    @abc.abstractmethod
    def __init__(self):
        """ Create a GPKernel. """
        self._n_hypers = 0

    @abc.abstractmethod
    def cov(self, X1, X2, hypers=None):
        """ Calculate the covariance. """
        return np.zeros((len(X1), len(X2)))

    @abc.abstractmethod
    def fit(self, X):
        return self._n_hypers


class PolynomialKernel(BaseKernel):

    """ A Polynomial kernel of the form (s0 + sp * x.T*x)^d

    Attributes:
    hypers (list): names of the hyperparameters required
    _deg (integer): degree of polynomial
    """

    def __init__(self, d):
        """ Initiate a polynomial kernel.

        Parameters:
            d (integer): degree of the polynomial
        """
        if not isinstance(d, int):
            raise TypeError('d must be an integer.')
        if d < 1:
            raise ValueError('d must be greater than or equal to 1.')
        self._n_hypers = 2
        self._deg = d
        self._saved = None
        self._X = None

    def fit(self, X):
        """ Remember an input. """
        self._saved = X @ X.T
        return self._n_hypers

    def cov(self, X1=None, X2=None, hypers=(1.0, 1.0)):
        """ Calculate the polynomial covariance matrix between X1 and X2.

        If neither X1 nor X2 is given, they are assumed to be saved from fit.

        Parameters:
            X1 (np.ndarray):
            X2 (np.ndarray):
            hypers (iterable):

        Returns:
            K (np.ndarray)
        """
        sigma_0, sigma_p = hypers
        if X1 is None and X2 is None:
            return np.power(sigma_0 ** 2 + sigma_p ** 2 * self._saved,
                            self._deg)
        return np.power(sigma_0 ** 2 + sigma_p ** 2 * X1 @ X2.T, self._deg)


class BaseRadialKernel(BaseKernel):

    """ Base class for radial kernel functions. """

    def __init__(self):
        self._n_hypers = 1
        return

    def cov(self, X1, X2, hypers=None):
        return BaseKernel.cov(X1, X2)

    def fit(self, X):
        self._saved = self._distance(X, X)
        return self._n_hypers

    def _distance(self, X1, X2):
        """ Calculates the squared distances between rows of X1 and X2.

        Each row of X1 and X2 represents one measurement. Each column
        represents a dimension.

        Parameters:
            X1 (np.ndarray): n x d
            X2 (np.ndarray): m x d

        Returns:
            D (np.ndarray): n x m
        """
        return distance.cdist(X1, X2, metric='sqeuclidean')


class MaternKernel(BaseRadialKernel):

    """ A Matern kernel with nu = 5/2 or 3/2.

    Attributes:
        hypers (list): names of the hyperparameters required
        _saved_X (dict): dict of saved index:X pairs
        nu (string): '3/2' or '5/2'
    """

    def __init__(self, nu):
        """ Initiate a Matern kernel.

        Parameters:
            nu (string): '3/2' or '5/2'
        """
        if nu not in ['3/2', '5/2']:
            raise ValueError("nu must be '3/2' or '5/2'")
        self._n_hypers = 1
        self.nu = nu

    def fit(self, X):
        BaseRadialKernel.fit(self, X)
        self._saved = np.sqrt(self._saved)
        return self._n_hypers

    def cov(self, X1=None, X2=None, hypers=(1.0, )):
        """ Calculate the Matern kernel between X1 and X2.

        Parameters:
            X1 (np.ndarray):
            X2 (np.ndarray)
            hypers (iterable): default is ell=1.0.

        Returns:
            K (np.ndarray)
        """
        if X1 is None and X2 is None:
            D = self._saved
        else:
            D = np.sqrt(self._distance(X1, X2))
        L = hypers[0]
        D_L = D / L
        if self.nu == '3/2':
            return self._m32(D_L)
        elif self.nu == '5/2':
            return self._m52(D_L)

    def _m32(self, D_L):
        return (1.0 + np.sqrt(3.0) * D_L) * np.exp(-np.sqrt(3) * D_L)

    def _m52(self, D_L):
        first = (1.0 + np.sqrt(5.0)*D_L) + 5.0*np.power(D_L, 2)/3.0
        second = np.exp(-np.sqrt(5.0) * D_L)
        return first * second


class SEKernel(BaseRadialKernel):

    """ A squared exponential kernel.

    Attributes:
        hypers (list)
        _d_squared (np.ndarray)
        _saved_X (dict)
    """

    def __init__(self):
        """ Initiate a SEKernel. """
        self._n_hypers = 2

    def cov(self, X1=None, X2=None, hypers=(1.0, 1.0)):
        """ Calculate the squared exponential kernel between x1 and x2.

        Parameters:
            X1 (np.ndarray):
            X2 (np.ndarray)
            hypers (iterable): default is (1.0, 1.0)

        Returns:
            K (np.ndarray)
        """
        sigma_f, ell = hypers
        if X1 is None and X2 is None:
            D = self._saved
        else:
            D = self._distance(X1, X2)
        sigma_f, L = hypers
        D_L2 = D / L ** 2
        return self._se(D_L2, sigma_f)

    def _se(self, D_L2, sigma_f):
        return sigma_f ** 2 * np.exp(-0.5 * D_L2)


class SumKernel(BaseKernel):

    """
    A kernel that sums over other kernels.

    Attributes:
        _kernels (list): list of member kernels
        hypers (list): the names of the hyperparameters
    """

    def __init__(self, kernels):
        '''
        Initiate a SumKernel containing a list of other kernels.

        Parameters:
            kernels(list): list of member kernels
            hypers (list): list of hyperparameter names
        '''
        self._kernels = kernels

    def fit(self, X):
        n_hypers = 0
        for kernel in self._kernels:
            n_hypers += kernel.fit(X)
        return n_hypers

    def cov(self, X1=None, X2=None, hypers=None):
        """ Calculate the sum kernel between two inputs.

        Parameters:
            X1 (np.ndarray):
            X2
            hypers (iterable): the hyperparameters. Default is to use
                the defaults for each kernel.

        Returns:
            K (np.ndarray)
        """
        if hypers is None:
            Ks = [kern.cov(X1, X2) for kern in self._kernels]
        else:
            hypers_inds = [k._n_hypers for k in self._kernels]
            hypers_inds = np.cumsum(np.array(hypers_inds))
            hypers_inds = np.insert(hypers_inds, 0, 0)
            Ks = [kern.cov(X1, X2, hypers[hypers_inds[i]:
                                          hypers_inds[i+1]])
                  for i, kern in enumerate(self._kernels)]
        return sum(Ks)


class LinearKernel(BaseKernel):

    """ The linear (dot product) kernel for two inputs. """

    def __init__(self):
        """ Initiates a LinearKernel. """
        self._n_hypers = 1

    def fit(self, X):
        self._saved = X @ X.T
        return self._n_hypers

    def cov(self, X1=None, X2=None, hypers=(1.0, )):
        """ Calculate the linear kernel between x1 and x2.

        The linear kernel is the dot product of x1 and x2 multiplied
        by a scaling variable var_p.

        Parameters:
            x1 (np.ndarray)
            x2 (np.ndarray)
            hypers (iterable): default is var_p=1.0.

        Returns:
            K (np.ndarray)
        """
        vp = hypers[0]
        if X1 is None and X2 is None:
            return vp * self._saved
        return vp * X1 @ X2.T
