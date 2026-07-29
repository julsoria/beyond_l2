r"""A one-property subclass of the public `cabrnet` ProtoPNetClassifier that exposes the
similarity layer's covariance matrices, as expected by
`explain.extended_explainers.IsotropicGaussianFormalExplanation` /
`IsotropicLogDistanceFormalExplanation`.

The public `cabrnet.archs.protopnet.decision.ProtoPNetClassifier` has no notion of
covariances (it was written for point-vector prototypes); this proxy property is not
otherwise a code change to ProtoPNet's forward pass, classification, or training logic.
"""

import torch

from cabrnet.archs.protopnet.decision import ProtoPNetClassifier


class GaussianProtoPNetClassifier(ProtoPNetClassifier):
    r"""ProtoPNetClassifier variant that exposes `covariances`/`precisions` properties
    forwarded from its similarity layer, for use with `beyond_l2_ext.similarities`."""

    @property
    def covariances(self) -> torch.Tensor:
        r"""Dynamically exposes the covariance matrices of the similarity layer."""
        if hasattr(self.similarity_layer, "covariances"):
            return self.similarity_layer.covariances
        raise AttributeError("Current similarity layer does not track covariances.")

    @property
    def precisions(self) -> torch.Tensor:
        r"""Dynamically exposes the precision matrices of the similarity layer, if available."""
        if hasattr(self.similarity_layer, "precisions"):
            return self.similarity_layer.precisions
        raise AttributeError("Current similarity layer does not track precisions.")
