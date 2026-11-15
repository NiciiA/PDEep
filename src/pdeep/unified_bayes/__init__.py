"""Geometry-aware Bayesian controls; no neural architectures or training."""
from .gaussian import GaussianPosterior, condition
from .operators import Coordinates, ForwardOperators, prior_specification
