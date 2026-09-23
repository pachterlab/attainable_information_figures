"""attainable_information -- finite-sample ceilings on cross-modal recoverability.

Given two paired matrices ``G`` (n x p) and ``X`` (n x d), the package answers
two questions about how much of ``G`` is linearly recoverable from ``X``:

1. **Channel recoverability.** :func:`fit_recoverability` fits the
   linear--Gaussian model by regularized canonical correlation analysis and
   returns the recoverability spectrum ``R_i = rho_i^2``, the identifiable
   directions of ``G`` and ``X``, and the Gaussian mutual information
   ``I(G; X) = -1/2 sum_i log(1 - rho_i^2)``. This is the infinite-data limit.

2. **Attainable information.** :func:`attainable_recoverability` and
   :func:`attainable_information` give the closed-form ceiling on what any
   learning algorithm trained on ``n`` samples with a ``d_star``-dimensional
   representation of ``X`` can recover out of sample::

       R_n = rho^2 * n / (n + nu),   nu = d_star * (1 - rho^2) / rho^2
       I_n = -1/2 * sum_i log(1 - R_n,i)

   ``nu`` is the learning cost of a direction: the sample size at which half of
   its channel recoverability is attainable, and also the Bayes-optimal ridge
   penalty in whitened coordinates.

The variable names ``G`` and ``X`` follow the radiogenomic application in which
the method was developed (genomics and imaging), but nothing in the code
depends on that interpretation. Any two patient-aligned, sample-aligned, or
row-aligned modalities can be used.
"""

from attainable_information.recoverability import (
    RecoverabilityFit,
    cross_validated_recoverability,
    cv_permutation_test,
    direction_recoverability,
    distance_correlation,
    distance_correlation_test,
    fit_recoverability,
    gaussian_rank_transform,
    imaging_variance_explained,
    make_synthetic_radiogenomics,
    mutual_information,
    permutation_test,
    posterior,
    subspace_alignment,
    to_dense,
    true_recoverability,
)
from attainable_information.bounds import (
    anchor_decomposition,
    attainable_information,
    attainable_recoverability,
    auc_ceiling,
    channel_information,
    learning_cost,
    max_information_given_fourth_moment,
    max_total_information,
    optimal_working_dimension,
    residualize,
    sample_size_for_fraction,
    weak_direction_bound,
)

__version__ = "0.1.0"

__all__ = [
    # channel recoverability (infinite-data limit)
    "RecoverabilityFit",
    "fit_recoverability",
    "posterior",
    "mutual_information",
    "direction_recoverability",
    "imaging_variance_explained",
    "cross_validated_recoverability",
    "permutation_test",
    "cv_permutation_test",
    "distance_correlation",
    "distance_correlation_test",
    "gaussian_rank_transform",
    "subspace_alignment",
    "to_dense",
    "make_synthetic_radiogenomics",
    "true_recoverability",
    # attainable information (finite-n ceiling)
    "attainable_recoverability",
    "attainable_information",
    "channel_information",
    "learning_cost",
    "sample_size_for_fraction",
    "optimal_working_dimension",
    "weak_direction_bound",
    "max_total_information",
    "max_information_given_fourth_moment",
    "auc_ceiling",
    "anchor_decomposition",
    "residualize",
]
