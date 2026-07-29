import copy
import os
from typing import Any, Callable
from collections import deque

import graphviz
import numpy as np
import torch
import torch.nn as nn
from loguru import logger
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from tqdm import tqdm
import time

from cabrnet.archs.generic.decision import CaBRNetClassifier
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.image import open_image
from cabrnet.core.utils.optimizers import OptimizerManager
from cabrnet.core.visualization.explainer import ExplanationGraph
from cabrnet.core.visualization.visualizer import SimilarityVisualizer

from cabrnet.archs.protopnet.model import ProtoPNet


class TesNet(ProtoPNet):
    r"""CaBRNet model implementing the TesNet architecture.

    This model inherits from ProtoPNet and modifies the loss function to include
    the Orthonormality and Subspace Separation losses as described in the TesNet paper.
    It is intended to be used with a CosineSimilarity layer.
    
    ---
    Reference: Wang et al., "Interpretable Image Recognition by Constructing 
    Transparent Embedding Space", ICCV 2021
    
    Key improvements over ProtoPNet:
    1. Orthogonality constraint (L_orth) for within-class prototypes
    2. Subspace separation (L_ss) on Grassmann manifold for between-class discrimination
    3. Dynamic stage switching based on convergence condition
    
    Loss formula (from original TesNet code):
        loss = CE + 0.8 * compactness - 0.08 * separation + 1e-4 * l1 + 1e-4 * orth - 1e-7 * subspace_sep
    
    Convergence condition (Algorithm 1, line 5):
        L_compactness < -L_separation  ⟺  min(d_correct) < min(d_wrong)

    """

    def __init__(self, extractor: nn.Module, classifier: CaBRNetClassifier, **kwargs):
        super().__init__(extractor, classifier, **kwargs)

        # Overwrite default loss coefficients with TesNet-specific ones
        self.loss_coefficients = {
            "clustering": 0.8,  # Corresponds to L_cs's compactness term
            "separability": - 0.08,   # Corresponds to L_cs's separation term (mu in paper)
            # "orthonormality": 1e-4,  # Lambda_1 in the paper - REMOVED
            "subspace_separation": - 1e-7,  # Lambda_2 in the paper
            "regularization": 1e-4,  # L1 regularization on the last layer
            "volume": 0.0,  # 
        }
        
        # Convergence tracking for dynamic stage switching
        self.loss_history = deque(maxlen=10)
        self.stage_switch_triggered = False

    def is_converged(
        self, 
        current_loss: float,
        patience: int = 5,
        threshold: float = 1e-4
    ) -> bool:
        r"""Detect convergence based on loss plateau."""
        self.loss_history.append(current_loss)
        
        if len(self.loss_history) < patience:
            return False
        
        recent_losses = list(self.loss_history)[-patience:]
        loss_std = np.std(recent_losses)
        loss_change = abs(recent_losses[-1] - recent_losses[0])
        
        converged: bool = (loss_std < threshold) and (loss_change < threshold)
        
        if converged:
            logger.debug(
                f"Convergence detected: std={loss_std:.6f}, "
                f"change={loss_change:.6f}"
            )
        
        return converged

    def check_stage_switch_condition(
        self, clustering_loss: float, separation_loss: float, current_loss: float,
        epoch_idx: int, require_convergence: bool = True, min_epoch: int = 10,
    ) -> bool:
        r"""Check if we should trigger Stage 2 (projection) and Stage 3 (fine-tuning)."""
        if epoch_idx < min_epoch or self.stage_switch_triggered:
            return False

        condition_met = clustering_loss < separation_loss
        
        logger.debug(
            f"Stage switch check: L_cluster={clustering_loss:.4f} < L_sep={separation_loss:.4f}?  Condition={condition_met}"
        )
        if not condition_met:
            return False
        
        if not require_convergence:
            logger.info(f"Stage switch condition met at epoch {epoch_idx} (convergence not required)")
            return True
            
        converged = self.is_converged(current_loss)
        if converged:
            logger.success(
                f"Stage switch triggered at epoch {epoch_idx}: "
                f"Converged AND L_cluster={clustering_loss:.4f} < L_sep={separation_loss:.4f}"
            )
            return True
        return False

    def loss(self, model_output: Any, label: torch.Tensor, **kwargs) -> tuple[torch.Tensor, dict[str, float]]:
        r"""TesNet loss function aligned with the original paper and implementation."""
        output, _, min_distances = model_output

        #  Cross-entropy loss
        cross_entropy = torch.nn.functional.cross_entropy(output, label)

        #  Clustering and Separation Loss
        prototypes_of_correct_class = torch.t(torch.index_select(self.classifier.proto_class_map, 1, label))
        # clustering_loss = torch.mean(torch.min(min_distances * prototypes_of_correct_class + (1 - prototypes_of_correct_class) * 1e5, dim=1).values)
        clustering_loss = torch.mean(torch.min(min_distances + (1 - prototypes_of_correct_class) * 1e5, dim=1).values)
        
        # separation_loss = torch.mean(torch.min(min_distances * prototypes_of_wrong_class + prototypes_of_wrong_class * 1e5, dim=1).values)
        separation_loss = torch.mean(torch.min(min_distances + prototypes_of_correct_class * 1e5, dim=1).values)
        
        # Reshape prototypes from (P, D, 1, 1) to a 2D tensor (P, D)
        protos_2d = self.classifier.prototypes.view(self.classifier.prototypes.size(0), -1)

        #  Orthonormality Loss
        M = self.classifier.num_proto_per_class
        C = self.classifier.num_classes
        D = self.classifier.num_features
        protos_by_class = protos_2d.view(C, M, D)
        identity = torch.eye(M, device=protos_2d.device).expand(C, M, M)
        ortho_matrices = torch.bmm(protos_by_class, protos_by_class.transpose(1, 2))

        #  Subspace Separation Loss (Vectorized)
        projection_matrices = torch.bmm(protos_by_class.transpose(1, 2), protos_by_class)
        
        # Flatten from (C, D, D) to (C, D*D)
        proj_flat = projection_matrices.view(C, -1)
        
        # Compute all pairwise L2 distances simultaneously (equivalent to Frobenius norms of differences)
        pairwise_distances = torch.cdist(proj_flat, proj_flat, p=2.0)
        
        # Extract the sum of the upper triangle (ignoring the diagonal and symmetric lower half)
        subspace_separation_loss = pairwise_distances.triu(diagonal=1).sum()
        
        # Number of unique pairs is C * (C - 1) / 2
        num_pairs = (C * (C - 1)) / 2
        
        if num_pairs > 0:
            subspace_separation_loss = subspace_separation_loss / (num_pairs * np.sqrt(2))
        else:
            subspace_separation_loss = torch.tensor(0.0, device=protos_2d.device)

        #  L1 Regularization
        l1_mask = 1 - torch.t(self.classifier.proto_class_map)
        l1 = (self.classifier.last_layer.weight * l1_mask).norm(p=1)

        # =====================================================================
        # --- Inject Volume Regularization (Metric Collapse Prevention) ---
        # =====================================================================
        volume_reg = torch.tensor(0.0, device=output.device)
        lambda_vol = self.loss_coefficients.get("volume", 0.0)

        sim_layer = self.classifier.similarity_layer
        if lambda_vol > 0.0 and hasattr(sim_layer, "cov_type") and not hasattr(sim_layer, "regularization_loss"):
            if sim_layer.cov_type in ["isotropic", "diagonal"]:
                # log det(Lambda) = -sum(log_var) for isotropic/diagonal
                volume_reg = -sim_layer.log_var.mean()
            elif sim_layer.cov_type == "general":
                # log det(Lambda) = 2 * sum(L_diag) [because diag is exp-parameterised]
                volume_reg = -sim_layer.L_diag.mean()
        # =====================================================================

        # Ask the layer cleanly for its own regularization penalty
        if lambda_vol > 0.0 and hasattr(sim_layer, "regularization_loss"):
            volume_reg = sim_layer.regularization_loss()
        # =====================================================================
        
        # Final Loss Combination
        # Separation and Subspace Separation are maximization objectives, so their coefficients should be negative.
        loss = (
            cross_entropy
            + self.loss_coefficients["clustering"] * clustering_loss
            + self.loss_coefficients["separability"] * separation_loss
            # + self.loss_coefficients["orthonormality"] * orthonormality_loss  # REMOVED
            + self.loss_coefficients["subspace_separation"] * subspace_separation_loss
            + self.loss_coefficients["regularization"] * l1
            + lambda_vol * volume_reg
        )

        batch_accuracy = torch.sum(torch.eq(torch.argmax(output, dim=1), label)).item() / len(label)
        stats = {
            "loss": loss.item(),
            "accuracy": batch_accuracy,
            "cross_entropy": cross_entropy.item(),
            "clustering": clustering_loss.item(),
            "separation": separation_loss.item(),
            # "orthonormality": removed,
            "subspace_separation": subspace_separation_loss.item(),
            "l1": l1.item(),
        }

        return loss, stats

    def tesnet_train_epoch(
        self,
        dataloaders: dict[str, DataLoader],
        optimizer_mngr,
        device: str | torch.device = "cuda:0",
        tqdm_position: int = 0,
        epoch_idx: int = 0,
        verbose: bool = False,
    ) -> dict[str, float]:
        r"""Train for one epoch with dynamic stage switching."""
        
        # Stage 1: Embedding space learning
        train_info = self._train_epoch(
            dataloaders,
            optimizer_mngr,
            "train_set",
            device,
            tqdm_position,
            epoch_idx,
            verbose,
            tqdm_title=f"Training epoch {epoch_idx}"
        )
        
        # Check for dynamic stage switching
        stage_switch_config = getattr(self, "dynamic_stage_switching", None)
        
        if stage_switch_config and stage_switch_config.get("enable", False):
            clustering_loss = train_info.get("train_set/clustering", 0.0)
            separation_loss = train_info.get("train_set/separation", 0.0)
            total_loss = train_info.get("train_set/loss", 0.0)
            
            require_convergence = stage_switch_config.get("require_convergence", True)
            min_epoch = stage_switch_config.get("min_epoch", 10)
            
            should_switch = self.check_stage_switch_condition(
                clustering_loss=clustering_loss,
                separation_loss=separation_loss,
                current_loss=total_loss,
                epoch_idx=epoch_idx,
                require_convergence=require_convergence,
                min_epoch=min_epoch,
            )
            
            if should_switch:
                logger.info(f"Triggering Stage 2 & 3 at epoch {epoch_idx}")
                self.stage_switch_triggered = True
                
                # Stage 2: Projection
                logger.info("Stage 2: Projection")
                self.project(
                    dataloader=dataloaders['projection_set'],
                    device=device,
                    verbose=verbose,
                    tqdm_position=tqdm_position,
                )
                
                # Stage 3: Fine-tuning
                logger.info("Stage 3: Fine-tuning")
                finetune_epochs = stage_switch_config.get("finetune_epochs", 20)
                finetune_period = stage_switch_config.get(
                    "finetune_period_name", "projection_finetune"
                )
                
                optimizer_mngr.freeze(period_name=finetune_period)
                
                for ft_epoch in tqdm(
                    range(finetune_epochs),
                    desc="Fine-tuning",
                    leave=False,
                    position=tqdm_position,
                    disable=not verbose,
                ):
                    self._train_epoch(
                        dataloaders,
                        optimizer_mngr,
                        "train_set",
                        device,
                        tqdm_position + 1,
                        epoch_idx,
                        verbose,
                        tqdm_title=f"Fine-tuning {ft_epoch+1}/{finetune_epochs}"
                    )
                
                logger.success("Stage 2 & 3 completed")
        
        return train_info

    def register_training_params(self, training_config: dict[str, Any]) -> None:
        r"""Registers additional training parameters for dynamic stage switching."""
        super().register_training_params(training_config)
        aux_info = training_config.get("auxiliary_info", {})
        if "dynamic_stage_switching" in aux_info:
            setattr(self, "dynamic_stage_switching", aux_info["dynamic_stage_switching"])