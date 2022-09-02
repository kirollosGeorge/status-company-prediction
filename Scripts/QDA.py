"""
Linear Discriminant Analysis and Quadratic Discriminant Analysis
"""


import warnings
import numpy as np
from scipy import linalg
from scipy.special import expit
from numbers import Real

from  sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from  sklearn.base import _ClassNamePrefixFeaturesOutMixin
from  sklearn.linear_model._base import LinearClassifierMixin
from  sklearn.covariance import ledoit_wolf, empirical_covariance, shrunk_covariance
from  sklearn.utils.multiclass import unique_labels
from  sklearn.utils.validation import check_is_fitted
from  sklearn.utils.multiclass import check_classification_targets
from  sklearn.utils.extmath import softmax
from  sklearn.preprocessing import StandardScaler
import joblib


#__all__ = ["LinearDiscriminantAnalysis", "QuadraticDiscriminantAnalysis"]


def _cov(X, shrinkage=None, covariance_estimator=None):
    
    if covariance_estimator is None:
        shrinkage = "empirical" if shrinkage is None else shrinkage
        if isinstance(shrinkage, str):
            if shrinkage == "auto":
                sc = StandardScaler()  # standardize features
                X = sc.fit_transform(X)
                s = ledoit_wolf(X)[0]
                # rescale
                s = sc.scale_[:, np.newaxis] * s * sc.scale_[np.newaxis, :]
            elif shrinkage == "empirical":
                s = empirical_covariance(X)
            else:
                raise ValueError("unknown shrinkage parameter")
        elif isinstance(shrinkage, Real):
            if shrinkage < 0 or shrinkage > 1:
                raise ValueError("shrinkage parameter must be between 0 and 1")
            s = shrunk_covariance(empirical_covariance(X), shrinkage)
        else:
            raise TypeError("shrinkage must be a float or a string")
    else:
        if shrinkage is not None and shrinkage != 0:
            raise ValueError(
                "covariance_estimator and shrinkage parameters "
                "are not None. Only one of the two can be set."
            )
        covariance_estimator.fit(X)
        if not hasattr(covariance_estimator, "covariance_"):
            raise ValueError(
                "%s does not have a covariance_ attribute"
                % covariance_estimator.__class__.__name__
            )
        s = covariance_estimator.covariance_
    return s


def _class_means(X, y):
    
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    return means


def _class_cov(X, y, priors, shrinkage=None, covariance_estimator=None):

    classes = np.unique(y)
    cov = np.zeros(shape=(X.shape[1], X.shape[1]))
    for idx, group in enumerate(classes):
        Xg = X[y == group, :]
        cov += priors[idx] * np.atleast_2d(_cov(Xg, shrinkage, covariance_estimator))
    return cov


class LinearDiscriminantAnalysis(
    _ClassNamePrefixFeaturesOutMixin,
    LinearClassifierMixin,
    TransformerMixin,
    BaseEstimator,
):
    

    def __init__(
        self,
        solver="svd",
        shrinkage=None,
        priors=None,
        n_components=None,
        store_covariance=False,
        tol=1e-4,
        covariance_estimator=None,
    ):
        self.solver = solver
        self.shrinkage = shrinkage
        self.priors = priors
        self.n_components = n_components
        self.store_covariance = store_covariance  # used only in svd solver
        self.tol = tol  # used only in svd solver
        self.covariance_estimator = covariance_estimator

    def _solve_lsqr(self, X, y, shrinkage, covariance_estimator):
       
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )
        self.coef_ = linalg.lstsq(self.covariance_, self.means_.T)[0].T
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def _solve_eigen(self, X, y, shrinkage, covariance_estimator):
        
        self.means_ = _class_means(X, y)
        self.covariance_ = _class_cov(
            X, y, self.priors_, shrinkage, covariance_estimator
        )

        Sw = self.covariance_  # within scatter
        St = _cov(X, shrinkage, covariance_estimator)  # total scatter
        Sb = St - Sw  # between scatter

        evals, evecs = linalg.eigh(Sb, Sw)
        self.explained_variance_ratio_ = np.sort(evals / np.sum(evals))[::-1][
            : self._max_components
        ]
        evecs = evecs[:, np.argsort(evals)[::-1]]  # sort eigenvectors

        self.scalings_ = evecs
        self.coef_ = np.dot(self.means_, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(self.means_, self.coef_.T)) + np.log(
            self.priors_
        )

    def _solve_svd(self, X, y):
        
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)

        self.means_ = _class_means(X, y)
        if self.store_covariance:
            self.covariance_ = _class_cov(X, y, self.priors_)

        Xc = []
        for idx, group in enumerate(self.classes_):
            Xg = X[y == group, :]
            Xc.append(Xg - self.means_[idx])

        self.xbar_ = np.dot(self.priors_, self.means_)

        Xc = np.concatenate(Xc, axis=0)

        # 1) within (univariate) scaling by with classes std-dev
        std = Xc.std(axis=0)
        # avoid division by zero in normalization
        std[std == 0] = 1.0
        fac = 1.0 / (n_samples - n_classes)

        # 2) Within variance scaling
        X = np.sqrt(fac) * (Xc / std)
        # SVD of centered (within)scaled data
        U, S, Vt = linalg.svd(X, full_matrices=False)

        rank = np.sum(S > self.tol)
        # Scaling of within covariance is: V' 1/S
        scalings = (Vt[:rank] / std).T / S[:rank]
        fac = 1.0 if n_classes == 1 else 1.0 / (n_classes - 1)

        # 3) Between variance scaling
        # Scale weighted centers
        X = np.dot(
            (
                (np.sqrt((n_samples * self.priors_) * fac))
                * (self.means_ - self.xbar_).T
            ).T,
            scalings,
        )
        # Centers are living in a space with n_classes-1 dim (maximum)
        # Use SVD to find projection in the space spanned by the
        # (n_classes) centers
        _, S, Vt = linalg.svd(X, full_matrices=0)

        if self._max_components == 0:
            self.explained_variance_ratio_ = np.empty((0,), dtype=S.dtype)
        else:
            self.explained_variance_ratio_ = (S**2 / np.sum(S**2))[
                : self._max_components
            ]

        rank = np.sum(S > self.tol * S[0])
        self.scalings_ = np.dot(scalings, Vt.T[:, :rank])
        coef = np.dot(self.means_ - self.xbar_, self.scalings_)
        self.intercept_ = -0.5 * np.sum(coef**2, axis=1) + np.log(self.priors_)
        self.coef_ = np.dot(coef, self.scalings_.T)
        self.intercept_ -= np.dot(self.xbar_, self.coef_.T)

    def fit(self, X, y):
        
        X, y = self._validate_data(
            X, y, ensure_min_samples=2, dtype=[np.float64, np.float32]
        )
        self.classes_ = unique_labels(y)
        n_samples, _ = X.shape
        n_classes = len(self.classes_)

        if n_samples == n_classes:
            raise ValueError(
                "The number of samples must be more than the number of classes."
            )

        if self.priors is None:  # estimate priors from sample
            _, y_t = np.unique(y, return_inverse=True)  # non-negative ints
            self.priors_ = np.bincount(y_t) / float(len(y))
        else:
            self.priors_ = np.asarray(self.priors)

        if (self.priors_ < 0).any():
            raise ValueError("priors must be non-negative")
        if not np.isclose(self.priors_.sum(), 1.0):
            warnings.warn("The priors do not sum to 1. Renormalizing", UserWarning)
            self.priors_ = self.priors_ / self.priors_.sum()

        # Maximum number of components no matter what n_components is
        # specified:
        max_components = min(len(self.classes_) - 1, X.shape[1])

        if self.n_components is None:
            self._max_components = max_components
        else:
            if self.n_components > max_components:
                raise ValueError(
                    "n_components cannot be larger than min(n_features, n_classes - 1)."
                )
            self._max_components = self.n_components

        if self.solver == "svd":
            if self.shrinkage is not None:
                raise NotImplementedError("shrinkage not supported")
            if self.covariance_estimator is not None:
                raise ValueError(
                    "covariance estimator "
                    "is not supported "
                    "with svd solver. Try another solver"
                )
            self._solve_svd(X, y)
        elif self.solver == "lsqr":
            self._solve_lsqr(
                X,
                y,
                shrinkage=self.shrinkage,
                covariance_estimator=self.covariance_estimator,
            )
        elif self.solver == "eigen":
            self._solve_eigen(
                X,
                y,
                shrinkage=self.shrinkage,
                covariance_estimator=self.covariance_estimator,
            )
        else:
            raise ValueError(
                "unknown solver {} (valid solvers are 'svd', "
                "'lsqr', and 'eigen').".format(self.solver)
            )
        if self.classes_.size == 2:  # treat binary case as a special case
            self.coef_ = np.array(
                self.coef_[1, :] - self.coef_[0, :], ndmin=2, dtype=X.dtype
            )
            self.intercept_ = np.array(
                self.intercept_[1] - self.intercept_[0], ndmin=1, dtype=X.dtype
            )
        self._n_features_out = self._max_components
        return self

    def transform(self, X):
        
        if self.solver == "lsqr":
            raise NotImplementedError(
                "transform not implemented for 'lsqr' solver (use 'svd' or 'eigen')."
            )
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)
        if self.solver == "svd":
            X_new = np.dot(X - self.xbar_, self.scalings_)
        elif self.solver == "eigen":
            X_new = np.dot(X, self.scalings_)

        return X_new[:, : self._max_components]

    def predict_proba(self, X):
        
        check_is_fitted(self)

        decision = self.decision_function(X)
        if self.classes_.size == 2:
            proba = expit(decision)
            return np.vstack([1 - proba, proba]).T
        else:
            return softmax(decision)

    def predict_log_proba(self, X):
       
        prediction = self.predict_proba(X)
        prediction[prediction == 0.0] += np.finfo(prediction.dtype).tiny
        return np.log(prediction)

    def decision_function(self, X):
        
        # Only override for the doc
        return super().decision_function(X)


class QuadraticDiscriminantAnalysis_Random_Forest(ClassifierMixin, BaseEstimator):

    def __init__(
        self, *, priors=None, reg_param=0.0, store_covariance=False, tol=1.0e-4
    ):
        self.priors = np.asarray(priors) if priors is not None else None
        self.reg_param = reg_param
        self.store_covariance = store_covariance
        self.tol = tol

    def fit(self, X, y):
       
        X, y = self._validate_data(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        n_samples, n_features = X.shape
        n_classes = len(self.classes_)
        if n_classes < 2:
            raise ValueError(
                "The number of classes has to be greater than one; got %d class"
                % (n_classes)
            )
        if self.priors is None:
            self.priors_ = np.bincount(y) / float(n_samples)
        else:
            self.priors_ = self.priors

        cov = None
        store_covariance = self.store_covariance
        if store_covariance:
            cov = []
        means = []
        scalings = []
        rotations = []
        for ind in range(n_classes):
            Xg = X[y == ind, :]
            meang = Xg.mean(0)
            means.append(meang)
            if len(Xg) == 1:
                raise ValueError(
                    "y has only 1 sample in class %s, covariance is ill defined."
                    % str(self.classes_[ind])
                )
            Xgc = Xg - meang
            # Xgc = U * S * V.T
            _, S, Vt = np.linalg.svd(Xgc, full_matrices=False)
            rank = np.sum(S > self.tol)
            if rank < n_features:
                warnings.warn("Variables are collinear")
            S2 = (S**2) / (len(Xg) - 1)
            S2 = ((1 - self.reg_param) * S2) + self.reg_param
            if self.store_covariance or store_covariance:
                # cov = V * (S^2 / (n-1)) * V.T
                cov.append(np.dot(S2 * Vt.T, Vt))
            scalings.append(S2)
            rotations.append(Vt.T)
        if self.store_covariance or store_covariance:
            self.covariance_ = cov
        self.means_ = np.asarray(means)
        self.scalings_ = scalings
        self.rotations_ = rotations
        return self

    def _decision_function(self, X):
        # return log posterior, see eq (4.12) p. 110 of the ESL.
        check_is_fitted(self)

        X = self._validate_data(X, reset=False)
        norm2 = []
        for i in range(len(self.classes_)):
            R = self.rotations_[i]
            S = self.scalings_[i]
            Xm = X - self.means_[i]
            X2 = np.dot(Xm, R * (S ** (-0.5)))
            norm2.append(np.sum(X2**2, axis=1))
        norm2 = np.array(norm2).T  # shape = [len(X), n_classes]
        u = np.asarray([np.sum(np.log(s)) for s in self.scalings_])
        return -0.5 * (norm2 + u) + np.log(self.priors_)

    def decision_function(self, X):
       
        dec_func = self._decision_function(X)
        # handle special case of two classes
        if len(self.classes_) == 2:
            return dec_func[:, 1] - dec_func[:, 0]
        return dec_func

    def predict(self, X):
        rf_model = joblib.load('Rf_selected_91.h5')
        
        d = self._decision_function(X)
        y_pred = self.classes_.take(d.argmax(1))
        prediction= []
        i=0
        for n in y_pred :
            if n == 0 :
                prediction.append(n)
                i = i+1
            else : 
                pred = rf_model.predict(X[i].reshape(1,-1))
                pred = list(pred)
                prediction = prediction + pred
                i = i+1
            
        return np.array(prediction)

    def predict_proba(self, X):
        
        values = self._decision_function(X)
        # compute the likelihood of the underlying gaussian models
        # up to a multiplicative constant.
        likelihood = np.exp(values - values.max(axis=1)[:, np.newaxis])
        # compute posterior probabilities
        return likelihood / likelihood.sum(axis=1)[:, np.newaxis]

    def predict_log_proba(self, X):
        
        # XXX : can do better to avoid precision overflows
        probas_ = self.predict_proba(X)
        return np.log(probas_)