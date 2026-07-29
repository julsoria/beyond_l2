from __future__ import annotations
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from cabrnet.archs.protopnet.decision import ProtoPNetClassifier
from cabrnet.core.utils.prototypes import init_prototypes


class TesNetClassifier(ProtoPNetClassifier):
    r"""Classification pipeline for TesNet architecture.
    
    This classifier is structurally identical to the ProtoPNetClassifier but is
    designed to be used with Cosine Similarity, as proposed in the TesNet paper.
    The core logic remains the same: compute similarities between features and prototypes,
    find the max similarity for each prototype, and pass these to a final linear layer.
    """

    def __init__(
        self,
        similarity_config: dict[str, Any],
        num_classes: int,
        num_features: int,
        num_proto_per_class: int,
        proto_init_mode: str = "UNIFORM",  # TesNet paper initializes with Uniform[0,1]
        incorrect_class_penalty: float = -0.5,
        **kwargs,
    ) -> None:
        super().__init__(num_classes=num_classes, num_features=num_features, similarity_config=similarity_config, num_proto_per_class=num_proto_per_class)

        assert num_proto_per_class > 0, f"Invalid number of prototypes per class: {num_proto_per_class}"
        self.num_proto_per_class = num_proto_per_class

        # Init prototypes (basis vectors in TesNet)
        self.prototypes = nn.Parameter(
            init_prototypes(
                num_prototypes=num_proto_per_class * num_classes,
                num_features=self.num_features,
                init_mode=proto_init_mode,
            )
        )
        # In TesNet, basis vectors are constrained to be unit vectors.

        # Init similarity layer (expected to be CosineSimilarity)
        self.build_similarity(similarity_config)

        # Initialize last layer
        proto_class_map = torch.zeros(self.num_prototypes, self.num_classes)
        for j in range(self.num_prototypes):
            proto_class_map[j, j // self.num_proto_per_class] = 1
        self.register_buffer("proto_class_map", proto_class_map, persistent=True)
        self.last_layer = nn.Linear(in_features=self.num_prototypes, out_features=self.num_classes, bias=False)
        correct_locations = torch.t(self.proto_class_map)
        incorrect_locations = 1 - correct_locations
        self.last_layer.weight.data.copy_(correct_locations + incorrect_class_penalty * incorrect_locations)

    def forward(self, features: Tensor, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        r"""Performs classification using a linear layer based on projection scores,
        while returning cosine distances for the loss function.
        """
        # Ensure prototypes are unit vectors for both calculations
        normalized_prototypes = torch.nn.functional.normalize(self.prototypes, p=2, dim=1)

        # 1. Calculate Cosine Distances (for the loss function)
        # This uses the existing CosineSimilarity layer.
        cosine_distances_map = self.similarity_layer.distances(features, prototypes=normalized_prototypes)
        min_cosine_distances = torch.min(cosine_distances_map.flatten(start_dim=2), dim=2).values

        # 2. Calculate Projection Scores (for the final logits)
        # This is a simple dot product, implemented with a 2D convolution.
        # Note: We do not normalize the 'features' tensor, only the prototypes.
        projection_scores_map = self.similarity_layer.similarities(features, prototypes=normalized_prototypes)
        
        # # Before — fails under torch.compile with dynamic shapes
        # max_projection_scores = torch.nn.functional.max_pool2d(
        #     projection_scores_map, kernel_size=projection_scores_map.shape[2:]
        # ).flatten(start_dim=1)

        # # After - compiler-safe
        max_projection_scores = torch.nn.functional.adaptive_max_pool2d(
            projection_scores_map, output_size=(1, 1)
        ).flatten(start_dim=1)
        
        # 3. Compute the final prediction using the projection scores
        prediction = self.last_layer(max_projection_scores)

        # Return logits, max projection scores, and min cosine distances
        return prediction, max_projection_scores, min_cosine_distances
