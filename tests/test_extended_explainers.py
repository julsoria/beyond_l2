"""Unit tests for the ALE explainer classes in `explain.extended_explainers`.

These tests exercise construction and the basic control flow of each explainer
(paradigm validation, dict-shaped explanation output, ProtoPool weight compression)
using lightweight mock/real objects. They are not a substitute for the end-to-end
verification described in the README (running `run_formal_exp.py` against a real
trained checkpoint and comparing bound/size outputs) -- the geometric correctness of
the bounds themselves is only meaningfully exercised there, since it depends on real
prototype/feature geometry that is impractical to hand-construct in a unit test.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "explain"))

from cabrnet.core.utils.similarities import CosineSimilarity, ProtoPNetSimilarity
from cabrnet.archs.protopool.decision import ProtoPoolClassifier

from extended_explainers import (
    CosineSimFormalExplanation,
    IsotropicGaussianFormalExplanation,
    IsotropicLogDistanceFormalExplanation,
    PIPSparseFormalExplanation,
    PIPSimplexFormalExplanation,
)


class MockModel(nn.Module):
    """Minimal stand-in for a CaBRNet model, exposing exactly what
    `FormalExplanationBase.__init__` and the explainer subclasses need:
    `classifier.{num_classes,num_prototypes,prototypes,last_layer,similarity_layer}`,
    plus top-level `similarities(x)` and `forward(x)`.
    """

    def __init__(self, num_classes=3, num_prototypes=6, num_features=4, height=2, width=2, similarity_layer=None):
        super().__init__()
        self.classifier = MagicMock()
        self.classifier.num_classes = num_classes
        self.classifier.num_prototypes = num_prototypes
        self.classifier.prototypes = nn.Parameter(torch.randn(num_prototypes, num_features, 1, 1))
        self.classifier.last_layer = nn.Linear(num_prototypes, num_classes, bias=False)
        self.classifier.similarity_layer = similarity_layer
        # No ProtoPool-specific attributes on this mock -> base class takes the
        # "standard" (non-ProtoPool) branch in FormalExplanationBase.__init__.
        del self.classifier.proto_slot_map

        self.num_prototypes = num_prototypes
        self.num_features = num_features
        self.H, self.W = height, width

        def _classifier_similarities(x):
            return similarity_layer.similarities(x, self.classifier.prototypes)

        self.classifier.similarities = _classifier_similarities

    def similarities(self, x):
        return self.classifier.similarities(x)

    def features(self, x):
        return torch.randn(x.size(0), self.num_features, self.H, self.W)

    def forward(self, x):
        return torch.randn(x.size(0), self.classifier.num_classes), None


@pytest.fixture
def x_y():
    return torch.randn(1, 3, 8, 8), torch.tensor([0])


class TestCosineSimFormalExplanation:
    def _model(self):
        return MockModel(similarity_layer=CosineSimilarity())

    def test_rejects_invalid_paradigm(self):
        model = self._model()
        with pytest.raises(AssertionError):
            CosineSimFormalExplanation(model, device="cpu", paradigm="not_a_paradigm")

    def test_instantiation_triangle(self):
        model = self._model()
        explainer = CosineSimFormalExplanation(model, device="cpu", paradigm="triangle")
        assert explainer.paradigm == "triangle"
        assert explainer.prototype_similarities.shape == (model.num_prototypes, model.num_prototypes)

    def test_instantiation_hypersphere(self):
        model = self._model()
        explainer = CosineSimFormalExplanation(model, device="cpu", paradigm="hypersphere")
        assert explainer.paradigm == "hypersphere"


class TestIsotropicExplanations:
    def _model_with_covariances(self, covariances):
        model = MockModel(similarity_layer=ProtoPNetSimilarity())
        model.classifier.covariances = covariances
        model.classifier.similarity_layer.stability_factor = 1e-4
        return model

    def test_isotropic_gaussian_instantiation(self):
        num_prototypes = 6
        model = self._model_with_covariances(torch.ones(num_prototypes))
        explainer = IsotropicGaussianFormalExplanation(model, device="cpu")
        assert explainer.covs.shape == (num_prototypes,)
        assert explainer.log_Z.shape == (num_prototypes,)

    def test_isotropic_log_distance_instantiation(self):
        model = self._model_with_covariances(torch.ones(6))
        explainer = IsotropicLogDistanceFormalExplanation(model, device="cpu")
        assert explainer.epsilon == 1e-4
        assert explainer.covs.shape == (6,)

    def test_isotropic_log_distance_round_trip(self):
        """The inverse-mapping (similarity -> Euclidean distance) and the base
        LogDistance.distances_to_similarities forward mapping should be inverses
        of one another, since IsotropicLogDistanceFormalExplanation relies on this
        to reuse the base Euclidean HIA/TI machinery.
        """
        model = self._model_with_covariances(torch.ones(6))
        explainer = IsotropicLogDistanceFormalExplanation(model, device="cpu")

        d_true = torch.tensor([0.0, 0.5, 1.0, 2.0])
        sim = torch.log((d_true**2 + 1.0) / (d_true**2 + explainer.epsilon))

        sigma_sq = torch.ones(4)
        exp_s = torch.exp(sim)
        denom = torch.clamp(exp_s - 1.0, min=1e-7)
        d_sigma_sq = torch.clamp((1.0 - explainer.epsilon * exp_s) / denom, min=0.0)
        d_recovered = torch.sqrt(sigma_sq * d_sigma_sq)

        assert torch.allclose(d_recovered, d_true, atol=1e-2)


class TestPIPExplanations:
    """PIP-Net's expansion loops (PIPSimplexFormalExplanation/PIPSparseFormalExplanation)
    only terminate once every competing class is verifiably beaten. With arbitrary
    random weights/logits that isn't guaranteed to happen (the "predicted" class from a
    random forward pass need not match the true argmax of the random weights), so tests
    use a deterministic, single-spatial-location scenario where class 0 dominates every
    other class through prototype 0 alone -- this converges in a single greedy step.
    """

    def _pipnet_like_model(self, num_prototypes=3, num_classes=3):
        model = MockModel(num_prototypes=num_prototypes, num_classes=num_classes, similarity_layer=MagicMock())
        model.H, model.W = 1, 1

        weights = torch.zeros(num_classes, num_prototypes)
        weights[0, 0] = 10.0  # Class 0 wins via prototype 0 alone; all else is 0.
        model.classifier.last_layer.weight.data = weights

        probs = torch.zeros(1, num_prototypes, 1, 1)
        probs[0, 0, 0, 0] = 0.9
        probs[0, 1:, 0, 0] = 0.1 / (num_prototypes - 1)
        model.similarities = lambda x: probs

        logits = torch.zeros(1, num_classes)
        logits[0, 0] = 10.0
        model.forward = lambda x: (logits, None)
        return model

    def test_pip_simplex_explain_one_returns_patch_prototype_dict(self, x_y):
        model = self._pipnet_like_model()
        explainer = PIPSimplexFormalExplanation(model, device="cpu")
        x, y = x_y
        explanation = explainer.explain_one(x, y, verbose=False)
        assert isinstance(explanation, dict)
        assert len(explanation) >= 1
        for key in explanation:
            assert isinstance(key, tuple) and len(key) == 2

    def test_pip_sparse_explain_one_returns_prototype_dict(self, x_y):
        model = self._pipnet_like_model()
        explainer = PIPSparseFormalExplanation(model, device="cpu")
        x, y = x_y
        explanation = explainer.explain_one(x, y, verbose=False)
        assert isinstance(explanation, dict)
        assert len(explanation) >= 1
        for key in explanation:
            assert isinstance(key, int)


class TestProtoPoolWeightCompression:
    def test_effective_weights_shape_and_sum(self):
        """FormalExplanationBase.__init__ must compress ProtoPool's (C, num_slots)
        decision head into an effective (C, num_prototypes) weight matrix."""
        num_classes, num_prototypes, num_slots_per_class, num_features = 3, 6, 2, 4
        classifier = ProtoPoolClassifier(
            similarity_config={"name": "ProtoPNetSimilarity"},
            num_classes=num_classes,
            num_features=num_features,
            num_prototypes=num_prototypes,
            num_slots_per_class=num_slots_per_class,
        )

        model = MagicMock()
        model.classifier = classifier
        model.eval = MagicMock(return_value=None)
        model.to = MagicMock(return_value=model)

        from subset_minimal_axp import FormalExplanationBase

        explainer = FormalExplanationBase(model, device="cpu", prototype_filepath="/tmp/unused_protos.pth")

        assert explainer.weights.shape == (num_classes, num_prototypes)
        # Every slot's weight must be attributed to exactly one prototype, so summing
        # the compressed weights must equal summing the raw per-slot weights.
        assert torch.allclose(explainer.weights.sum(), classifier.last_layer.weight.data.sum(), atol=1e-4)
