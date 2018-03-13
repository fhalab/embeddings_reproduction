import pickle
import abc

import numpy as np
from scipy.optimize import minimize
from scipy import linalg
import pandas as pd

import gpk

class BaseGPModel(abc.ABC):

    """ Base class for Gaussian process models. """

    @abc.abstractmethod
    def __init__(self, kernel):
        self.kernel = kernel

    @abc.abstractmethod
    def predict(self, X):
        return

    @abc.abstractmethod
    def fit(self, X, Y):
        return

    def _set_params(self, **kwargs):
        ''' Sets parameters for the model.

        This function can be used to set the value of any or all
        attributes for the model. However, it does not necessarily
        update dependencies, so use with caution.
        '''
        for key, value in kwargs.items():
            setattr(self, key, value)

    @classmethod
    def load(cls, model):
        ''' Load a saved model.

        Use pickle to load the saved model.

        Parameters:
            model (string): path to saved model
        '''
        with open(model, 'rb') as m_file:
            attributes = pickle.load(m_file, encoding='latin1')
        model = cls(attributes['kernel'])
        del attributes['kernel']
        if attributes['objective'] == 'LOO_log_p':
            model.objective = model._LOO_log_p
        else:
            model.objective = model._log_ML
        del attributes['objective']
        model._set_params(**attributes)
        return model

    def dump(self, f):
        ''' Save the model.

        Use pickle to save a dict containing the model's
        attributes.

        Parameters:
            f (string): path to where model should be saved
        '''
        save_me = {k: self.__dict__[k] for k in list(self.__dict__.keys())}
        if self.objective == self._log_ML:
            save_me['objective'] = 'log_ML'
        else:
            save_me['objective'] = 'LOO_log_p'
        save_me['guesses'] = self.guesses
        try:
            save_me['hypers'] = list(self.hypers)
            # names = self.hypers._fields
            # hypers = {n: h for n, h in zip(names, self.hypers)}
            # save_me['hypers'] = hypers
        except AttributeError:
            pass
        with open(f, 'wb') as f:
            pickle.dump(save_me, f)


class GPRegressor(BaseGPModel):

    """ A Gaussian process regression model for proteins. """

    def __init__(self, kernel, **kwargs):
        BaseGPModel.__init__(self, kernel)
        self.guesses = None
        if 'objective' not in list(kwargs.keys()):
            kwargs['objective'] = 'log_ML'
        self.variances = None
        self._set_objective(kwargs['objective'])
        del kwargs['objective']
        self._set_params(**kwargs)

    def _set_objective(self, objective):
        """ Set objective function for model. """
        if objective is not None:
            if objective == 'log_ML':
                self.objective = self._log_ML
            else:
                raise AttributeError(objective + ' is not a valid objective')
        else:
            self.objective = self._log_ML

    def fit(self, X, Y, variances=None, bounds=None):
        ''' Fit the model to the given data.

        Set the hyperparameters by training on the given data.
        Update all dependent values.

        Measurement variances can be given, or
        a global measurement variance will be estimated.

        Parameters:
            X (np.ndarray): n x d
            Y (np.ndarray): n.
            variances (np.ndarray): n. Optional.
        '''
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(Y, pd.Series):
            Y = Y.values
        self.X = X
        self.Y = Y
        self._ell = len(Y)
        self._n_hypers = self.kernel.fit(X)
        self.mean, self.std, self.normed_Y = self._normalize(self.Y)
        if variances is not None:
            if not len(variances) != len(Y):
                raise ValueError('len(variances must match len(Y))')
            self.variances = variances / self.std**2
        else:
            self.variances = None
            self._n_hypers += 1
        if self.guesses is None:
            guesses = [0.9 for _ in range(self._n_hypers)]
        else:
            guesses = self.guesses
            if len(guesses) != self._n_hypers:
                raise AttributeError(('Length of guesses does not match '
                                      'number of hyperparameters'))
        if bounds is None:
            bounds = [(1e-5, None) for _ in guesses]
        minimize_res = minimize(self.objective,
                                guesses,
                                method='L-BFGS-B',
                                bounds=bounds)
        self.hypers = minimize_res['x']


    def _make_Ks(self, hypers):
        """ Make covariance matrix (K) and noisy covariance matrix (Ky)."""
        if self.variances is not None:
            K = self.kernel.cov(hypers=hypers)
            Ky = K + np.diag(self.variances)
        else:
            K = self.kernel.cov(hypers=hypers[1::])
            Ky = K + np.identity(len(K)) * hypers[0]
        return K, Ky

    def _normalize(self, data):
        """ Normalize the given data.

        Normalizes the elements in data by subtracting the mean and
        dividing by the standard deviation.

        Parameters:
            data (pd.Series)

        Returns:
            mean, standard_deviation, normed
        """
        m = data.mean()
        s = data.std()
        return m, s, (data-m) / s

    def unnormalize(self, normed):
        """ Inverse of _normalize, but works on single values or arrays.

        Parameters:
            normed

        Returns:
            normed*self.std * self.mean
        """
        return normed*self.std + self.mean

    def predict(self, X):
        """ Make predictions for each sequence in new_seqs.

        Predictions are scaled as the original outputs (not normalized)

        Uses Equations 2.23 and 2.24 of RW
        Parameters:
            new_seqs (pd.DataFrame or np.ndarray): sequences to predict.

         Returns:
            means, cov as np.ndarrays. means.shape is (n,), cov.shape is (n,n)
        """
        if isinstance(X, pd.DataFrame):
            X = X.values
        h = self.hypers[1::]
        k_star = self.kernel.cov(X, self.X, hypers=h)
        k_star_star = self.kernel.cov(X, X, hypers=h)
        E = k_star @ self._alpha
        v = linalg.solve_triangular(self._L, k_star.T, lower=True)
        var = k_star_star - v.T @ v
        if self.variances is None:
            np.fill_diagonal(var, np.diag(var) + self.hypers[0])
        E = self.unnormalize(E)
        E = E[:, 0]
        var *= self.std ** 2
        return E, var

    def _log_ML(self, hypers):
        """ Returns the negative log marginal likelihood for the model.

        Uses RW Equation 5.8.

        Parameters:
            log_hypers (iterable): the hyperparameters

        Returns:
            log_ML (float)
        """
        self._K, self._Ky = self._make_Ks(hypers)
        self._L = np.linalg.cholesky(self._Ky)
        self._alpha = linalg.solve_triangular(self._L, self.normed_Y, lower=True)
        self._alpha = linalg.solve_triangular(self._L.T, self._alpha,
                                              lower=False)
        self._alpha = np.expand_dims(self._alpha, 1)

        first = 0.5 * np.dot(self.normed_Y, self._alpha)
        second = np.sum(np.log(np.diag(self._L)))
        third = len(self._K) / 2. * np.log(2 * np.pi)
        self.ML = (first + second + third).item()
        return self.ML
