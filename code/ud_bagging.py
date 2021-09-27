import itertools
import numbers
from warnings import warn

import numpy as np
from joblib import Parallel
from pandas import DataFrame
from scipy.stats import entropy
from sklearn.ensemble._bagging import _generate_indices, BaggingClassifier, MAX_INT
from sklearn.ensemble._base import _partition_estimators
from sklearn.utils import check_random_state, indices_to_mask
from sklearn.utils.fixes import delayed
from sklearn.utils.validation import has_fit_parameter, _deprecate_positional_args, _check_sample_weight


def generate_biased_indices(random_state, bootstrap, n_population, n_samples, probabilities):

    return random_state.choice(range(n_population), size=n_samples, replace=bootstrap, p=probabilities)


def generate_biased_bagging_indices(random_state, bootstrap_features,
                                    bootstrap_samples, n_features, n_samples,
                                    max_features, max_samples, feature_bias=None):
    """Randomly draw feature and sample indices."""
    # Get valid random state
    random_state = check_random_state(random_state)

    # Draw indices
    if feature_bias is None:
        feature_indices = _generate_indices(random_state, bootstrap_features,
                                            n_features, max_features)
    else:
        feature_indices = generate_biased_indices(random_state, bootstrap_features,
                                                  n_features, max_features, feature_bias)

    sample_indices = _generate_indices(random_state, bootstrap_samples, n_samples, max_samples)

    return feature_indices, sample_indices


def parallel_build_estimators(n_estimators, ensemble, X, y, sample_weight,
                              seeds, total_n_estimators, verbose, feature_bias=None):
    """Private function used to build a batch of estimators within a job."""
    # Retrieve settings
    n_samples, n_features = X.shape
    max_features = ensemble._max_features
    max_samples = ensemble._max_samples
    bootstrap = ensemble.bootstrap
    bootstrap_features = ensemble.bootstrap_features
    support_sample_weight = has_fit_parameter(ensemble.base_estimator_,
                                              "sample_weight")
    if not support_sample_weight and sample_weight is not None:
        raise ValueError("The base estimator doesn't support sample weight")

    # Build estimators
    estimators = []
    estimators_features = []

    for i in range(n_estimators):
        if verbose > 1:
            print("Building estimator %d of %d for this parallel run "
                  "(total %d)..." % (i + 1, n_estimators, total_n_estimators))

        random_state = seeds[i]
        estimator = ensemble._make_estimator(append=False,
                                             random_state=random_state)

        # Draw random feature, sample indices
        features, indices = generate_biased_bagging_indices(random_state,
                                                            bootstrap_features,
                                                            bootstrap, n_features,
                                                            n_samples, max_features,
                                                            max_samples,
                                                            feature_bias=feature_bias)

        # Draw samples, using sample weights, and then fit
        if support_sample_weight:
            if sample_weight is None:
                curr_sample_weight = np.ones((n_samples,))
            else:
                curr_sample_weight = sample_weight.copy()

            if bootstrap:
                sample_counts = np.bincount(indices, minlength=n_samples)
                curr_sample_weight *= sample_counts
            else:
                not_indices_mask = ~indices_to_mask(indices, n_samples)
                curr_sample_weight[not_indices_mask] = 0

            estimator.fit(X[:, features], y, sample_weight=curr_sample_weight)

        else:
            estimator.fit((X[indices])[:, features], y[indices])

        estimators.append(estimator)
        estimators_features.append(features)

    return estimators, estimators_features


def compute_feature_bias(x, uncertain_features, sample_indices=None):
    if sample_indices is None:
        sample_indices = range(x.shape[0])

    bias_sum = 0.0
    feature_bias = np.empty((x.shape[1],), dtype=np.float64)
    for f in range(x.shape[1]):
        entropy_sum = 0.0
        missing_values = 0
        if uncertain_features[f]:
            for s in sample_indices:
                v = x.iloc[s, f] if isinstance(x, DataFrame) else x[s, f]
                if v > 0:
                    entropy_sum += entropy([v, 1.0 - v])
                else:
                    missing_values += 1
        nb_indices = sample_indices.stop if isinstance(sample_indices, range) else sample_indices.shape[0]
        if nb_indices == missing_values:
            feature_bias[f] = 0.0001
        else:
            feature_bias[f] = 1.0 - entropy_sum / (nb_indices - missing_values)
            feature_bias[f] *= (nb_indices - missing_values) / nb_indices    # Known values rate
        bias_sum += feature_bias[f]

    return feature_bias / bias_sum


class UDBaggingClassifier(BaggingClassifier):
    @_deprecate_positional_args
    def __init__(self,
                 base_estimator=None,
                 n_estimators=10, *,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap=True,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0,
                 uncertain_features=None,
                 biased_subspaces=False):
        super().__init__(
            base_estimator,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)
        self.uncertain_features = uncertain_features
        self.biased_subspaces = biased_subspaces

    def _fit(self, X, y, max_samples=None, max_depth=None, sample_weight=None):
        random_state = check_random_state(self.random_state)

        # Convert data (X is required to be 2d and indexable)
        X, y = self._validate_data(
            X, y, accept_sparse=['csr', 'csc'], dtype=None,
            force_all_finite=False, multi_output=True
        )
        if sample_weight is not None:
            sample_weight = _check_sample_weight(sample_weight, X, dtype=None)

        # Remap output
        n_samples, self.n_features_ = X.shape
        self._n_samples = n_samples
        y = self._validate_y(y)

        # Check parameters
        self._validate_estimator()

        if max_depth is not None:
            self.base_estimator_.max_depth = max_depth

        # Validate max_samples
        if max_samples is None:
            max_samples = self.max_samples
        elif not isinstance(max_samples, numbers.Integral):
            max_samples = int(max_samples * X.shape[0])

        if not (0 < max_samples <= X.shape[0]):
            raise ValueError("max_samples must be in (0, n_samples]")

        # Store validated integer row sampling value
        self._max_samples = max_samples

        # Validate max_features
        if isinstance(self.max_features, numbers.Integral):
            max_features = self.max_features
        elif isinstance(self.max_features, float):
            max_features = self.max_features * self.n_features_
        else:
            raise ValueError("max_features must be int or float")

        if not (0 < max_features <= self.n_features_):
            raise ValueError("max_features must be in (0, n_features]")

        max_features = max(1, int(max_features))

        # Store validated integer feature sampling value
        self._max_features = max_features

        # Other checks
        if not self.bootstrap and self.oob_score:
            raise ValueError("Out of bag estimation only available"
                             " if bootstrap=True")

        if self.warm_start and self.oob_score:
            raise ValueError("Out of bag estimate only available"
                             " if warm_start=False")

        if hasattr(self, "oob_score_") and self.warm_start:
            del self.oob_score_

        if not self.warm_start or not hasattr(self, 'estimators_'):
            # Free allocated memory, if any
            self.estimators_ = []
            self.estimators_features_ = []

        n_more_estimators = self.n_estimators - len(self.estimators_)

        if n_more_estimators < 0:
            raise ValueError('n_estimators=%d must be larger or equal to '
                             'len(estimators_)=%d when warm_start==True'
                             % (self.n_estimators, len(self.estimators_)))

        elif n_more_estimators == 0:
            warn("Warm-start fitting without increasing n_estimators does not "
                 "fit new trees.")
            return self

        # Parallel loop
        n_jobs, n_estimators, starts = _partition_estimators(n_more_estimators,
                                                             self.n_jobs)
        total_n_estimators = sum(n_estimators)

        # Advance random state to state after training
        # the first n_estimators
        if self.warm_start and len(self.estimators_) > 0:
            random_state.randint(MAX_INT, size=len(self.estimators_))

        seeds = random_state.randint(MAX_INT, size=n_more_estimators)
        self._seeds = seeds

        feature_bias = None
        if self.biased_subspaces:
            feature_bias = compute_feature_bias(X, self.uncertain_features)

        all_results = Parallel(n_jobs=n_jobs, verbose=self.verbose,
                               **self._parallel_args())(
            delayed(parallel_build_estimators)(
                n_estimators[i],
                self,
                X,
                y,
                sample_weight,
                seeds[starts[i]:starts[i + 1]],
                total_n_estimators,
                verbose=self.verbose,
                feature_bias=feature_bias)
            for i in range(n_jobs))

        # Reduce
        self.estimators_ += list(itertools.chain.from_iterable(
            t[0] for t in all_results))
        self.estimators_features_ += list(itertools.chain.from_iterable(
            t[1] for t in all_results))

        if self.oob_score:
            self._set_oob_score(X, y)

        return self
