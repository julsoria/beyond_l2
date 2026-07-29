import os
# import sys
from pathlib import Path
import time
from typing import Optional, Type, Callable, Any, List

import psutil
import logging
import numpy as np
import torch
import torch.nn as nn
import torchinfo
from tqdm import tqdm
from cabrnet.archs.protopool.decision import ProtoPoolClassifier
from cabrnet.archs.generic.model import CaBRNet
from cabrnet.core.utils.similarities import SquaredEuclideanDistance, CosineSimilarity
from cabrnet.core.utils.data import DatasetManager as DM
from extra_utils import _euclidean_distances

MAX_EXPLANATIONS = 100_000


def check_memory_usage(threshold_mb=500):
    """Check current memory usage and raise an error if it exceeds the threshold."""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_used_mb = memory_info.rss / (1024 * 1024)  # Convert bytes to MB

    if memory_used_mb > threshold_mb:
        print(f"Memory usage exceeded! Current usage: {memory_used_mb:.2f} MB")
        sys.exit("Terminating script due to high memory usage.")


def load_model(final_model_path: Path, seed:int, device: str = "cuda:0", test_set: bool = True, val_set: bool = False):
    model_path = final_model_path / "model_state.pth"
    model_config_path = final_model_path / "model_arch.yml"
    data_config_path = final_model_path / "dataset.yml"
    
    # Load the model
    print("Loading the model...")
    print("Model config path: ", model_config_path)
    model = CaBRNet.build_from_config(config=model_config_path, state_dict_path=model_path, seed=seed)
    
    # --- SPARSITY ---
    if hasattr(model.classifier, 'set_sparsity_enabled'):
        print("Enabling Sparsity for Explanations...")
        model.classifier.set_sparsity_enabled(True)
        print(f"Sparsity mode: {model.classifier.sparsity_config.get('mode', 'unknown')}")
        print(f"Mask during inference: {model.classifier.sparsity_config.get('mask_inference', 'unknown')}")
        print(f"TopK value (if applicable): {model.classifier.sparsity_config.get('k', 'unknown')}")
    # --------------------
    
    # Load the test data
    print("Loading the test data...")
    dataloaders = DM.get_dataloaders(config=(data_config_path))
    print(dataloaders.keys())
    # print('Evaluating the model...')
    # res = model.evaluate(dataloaders['test_set'], device='cuda:0', verbose=True)
    # print('Results:', res)
    test_loader = dataloaders["test_set"]
    
    if val_set:
        try:
            val_loader = dataloaders["val_set"]
        except KeyError:
            print("Validation set not found, using test set as validation set.")
            val_loader = test_loader
    
    
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print("Device: ", device)
    model.to(device)
    if test_set:
        model.eval()
        return model, test_loader
    elif val_set:
        model.eval()
        return model, val_loader
    else:
        return model

# --- Wrapper Function ---
def _create_sqrt_wrapper(original_distance_func: Callable, eps: float = 1e-12) -> Callable:
    """Creates a wrapper that applies sqrt to the output of a distance function."""
    def sqrt_distance_wrapper(*args, **kwargs):
        # Calculate original squared distance
        dist_sq = original_distance_func(*args, **kwargs)
        # Clamp slightly above zero before sqrt for numerical stability
        # (prevents NaN gradients if ever used, and handles potential tiny negatives)
        dist = torch.sqrt(torch.clamp(dist_sq, min=eps))
        # dist = torch.sqrt(dist_sq)
        return dist
    sqrt_distance_wrapper.is_sqrt_wrapper = True  # Tag to identify wrapped functions
    return sqrt_distance_wrapper

class FormalExplanationBase:
    """
    FormalExplainer focusing on initialization and verification.
    Takes a model and device, handles prototype distances loading/saving,
    and provides a vectorized triangle inequality check.
    """

    def __init__(
        self,
        model: CaBRNet,
        device: Optional[str] = None,
        prototype_filepath: Optional[str] = None,  # Configurable path
        save_proto: bool = False,
        load_proto: bool = True,
        max_explanations: int = MAX_EXPLANATIONS,
        epsilon: float = 1e-6,  # Ideally should be small
    ):
        """
        Initializes the FormalExplainer.

        Args:
            model: The CaBRNet model to explain.
            device: The device to use ('cuda:0', 'cpu', etc.). Autodetects if None.
            prototype_filepath: Path to load/save prototype distances. Required if save_proto=True or load_proto=True.
            save_proto: Whether to calculate and save prototype distances.
            load_proto: Whether to load pre-calculated prototype distances.
            max_explanations: Maximum number of explanations to generate (used by explanation logic).
        """
        print("Initializing Refactored Formal Explanation...")
        
        self.prototype_filepath = prototype_filepath
        self.save_proto = save_proto
        self.load_proto = load_proto
        
        # --- Argument Validation ---
        if save_proto and load_proto:
            raise ValueError("Cannot save and load prototypes simultaneously.")
        if (save_proto or load_proto) and not prototype_filepath:
            prototype_filepath = os.path.join(os.getcwd(), "prototype_distances.pth")
            print("prototype_filepath must be provided if save_proto or load_proto is True.")

        # --- Device Setup ---
        if device is not None:
            self.device = device
        else:
            self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # --- Model Setup ---
        self.model = model
        self.model.eval()
        self.model.to(self.device)

        # --- Model Properties ---
        if not hasattr(model, 'classifier') or \
           not hasattr(model.classifier, 'num_classes') or \
           not hasattr(model.classifier, 'num_prototypes') or \
           not hasattr(model.classifier, 'prototypes') or \
           not hasattr(model.classifier, 'last_layer'):
        #    not hasattr(model.classifier, 'similarity_layer') or \
        #    not hasattr(model.classifier.similarity_layer, 'distances'):
            raise AttributeError("Model does not have the expected structure (classifier, prototypes, similarity_layer.distances, etc.)")

        self.K = self.num_classes = model.classifier.num_classes
        self.P = self.num_prototypes = model.classifier.num_prototypes
        self.prototypes = model.classifier.prototypes.data  # Get data, ensure it's not a parameter requiring grad here
        self.D = None  # Will be set after checking distance function output shape
        if self.prototypes.ndim > 1:
            self.D = self.latent_dim = self.prototypes.shape[1]
        else:
            print("Prototypes do not have a channel dimension. D is set to None.")
            
        # --- ADAPTATION FOR PROTOPOOL WEIGHTS ---
        # ProtoPool has a linear layer of shape (C, Slots) but we explain (C, Prototypes).
        # We must condense the slot weights into effective prototype weights.
        # if isinstance(self.model.classifier, ProtoPool)
        if hasattr(self.model.classifier, 'proto_slot_map') and hasattr(self.model.classifier, 'num_slots_per_class'):
            print("ProtoPool detected. Calculating effective weights for active prototypes...")
            
            # 1. Get architectural constants
            num_classes = self.model.classifier.num_classes
            num_slots_per_class = self.model.classifier.num_slots_per_class
            
            # 2. Get the mapping: Which prototype feeds which slot?
            # Returns shape (C, S_per_class) containing indices of active prototypes (0..201)
            # We ensure it's on CPU/Numpy for easy iteration
            slot_to_proto_map = self.model.classifier.class_mapping 
            if isinstance(slot_to_proto_map, torch.Tensor):
                slot_to_proto_map = slot_to_proto_map.cpu().numpy()

            # 3. Get raw weights (C, Total_Slots), e.g., (200, 2000)
            raw_weights = self.model.classifier.last_layer.weight.data

            # 4. Initialize effective weights (C, P_active), e.g., (200, 202)
            # self.P is the number of active prototypes (e.g. 202)
            effective_weights = torch.zeros((num_classes, self.P), device=self.device)

            # 5. Aggregate weights
            # The linear layer input is flattened: [Class 0 Slots | Class 1 Slots | ... ]
            current_slot_linear_idx = 0
            
            for c in range(num_classes):
                for s in range(num_slots_per_class):
                    # Find which prototype is assigned to this specific slot
                    p_idx = slot_to_proto_map[c, s]
                    
                    # Add the weight of this slot to the prototype's effective weight
                    # We add the entire column (for all classes) because the linear layer is fully connected
                    effective_weights[:, p_idx] += raw_weights[:, current_slot_linear_idx]
                    
                    current_slot_linear_idx += 1
            
            self.weights = effective_weights
            print(f"ProtoPool Adaptation: Compressed weights from {raw_weights.shape} to {self.weights.shape}")
        
        else:
            # Standard ProtoPNet behavior (1-to-1 mapping)
            self.weights = self.model.classifier.last_layer.weight.data
            
        # Ensure weights are on the correct device
        self.weights = self.weights.to(self.device)
        
        
        # Ensure prototypes are on the correct device for calculations
        self.prototypes = self.prototypes.to(self.device)
        self.epsilon = epsilon
        
        # if model does not have `similarity_layer` we add it as None:
        if not hasattr(self.model.classifier, 'similarity_layer'):
            self.model.classifier.similarity_layer = None
        
        if isinstance(self.model.classifier.similarity_layer, SquaredEuclideanDistance):
            self.init_prototype_distances(save_proto=save_proto, load_proto=load_proto, prototype_filepath=prototype_filepath)
        elif isinstance(self.model.classifier.similarity_layer, CosineSimilarity):
            # no need to use distances, we operate directly on similarities. We can set prototype_distances to None or skip initialization.
            self.prototype_distances = None
            print("Model uses CosineSimilarity. Skipping prototype distance initialization.")
        else:
            self.prototype_distances = None
            print("Model does not use SquaredEuclideanDistance. Skipping prototype distance initialization.")

        # --- Other Attributes ---
        self.max_explanations = max_explanations
        self.explanation: list = []  # Use a single clear name
        self.decision_head = self.model.classifier.last_layer  # h (in the original paper)
        self.batch_size = 0
        self.batch_idx = -1
        self.explanations_size: list = []
        self.correct_explanations: list = []
        self.incorrect_explanations: list = []
        self.counter_step = 100
        self.H: Optional[int] = None  # Initialize spatial dimensions as None
        self.W: Optional[int] = None  # Will be set when processing first batch

        # Potentially useful, depending on explanation methods
        # self.distance_fn = self.model.distances

        print(f"Initialized with {self.P} prototypes, {self.K} classes. Latent dim: {self.D if self.D else 0}.")
        print("Ready for explanation generation.")


    # def init_prototype_distances(self, save_proto: bool = False, load_proto: bool = True, prototype_filepath: Optional[str] = None):
    #     # --- ROBUST DISTANCE FUNCTION SETUP ---
    #     dist_layer = self.model.classifier.similarity_layer
    #     current_method = dist_layer.distances
    #     self.use_wrapper = True # Set to True if you prefer wrapping over class replacement
        
    #     print(dist_layer)
    #     print(current_method)
    #     print((hasattr(current_method, '__func__') and current_method.__func__ is SquaredEuclideanDistance.distances))
        
    #     # 1. Check: Already Wrapped? (Wrapper Tag)
    #     if hasattr(current_method, 'is_sqrt_wrapper') and current_method.is_sqrt_wrapper:
    #         print("INFO: Distance function is already wrapped (L2). Reusing existing configuration.")
    #         proto_dist_calculator = current_method
    #         self.feature_distance_fn = self.model.distances


    #     # 2. Check: Needs Adaptation? (SquaredEuclideanDistance)
    #     elif (hasattr(current_method, '__func__') and 
    #         current_method.__func__ is SquaredEuclideanDistance.distances):
            
    #         if self.use_wrapper:
    #             print("INFO: Detected SquaredEuclideanDistance. Applying sqrt wrapper.")
    #             # Create and persist the wrapper so next pass detects it
    #             wrapped_dist = _create_sqrt_wrapper(current_method, self.epsilon)
    #             dist_layer.distances = wrapped_dist # Monkey-patch the object
    #             proto_dist_calculator = wrapped_dist
                
    #             # Wrap feature distances similarly
    #             original_feat_dist = self.model.distances
    #             def sqrt_feature_distance_fn(x):
    #                 return torch.sqrt(torch.clamp(original_feat_dist(x), min=self.epsilon))
    #             self.feature_distance_fn = sqrt_feature_distance_fn
                
    #         else:
    #             print("INFO: Replacing prototype distance method with L2 distance.")
    #             # Replace the layer entirely
    #             self.model.classifier.similarity_layer = _euclidean_distances  # Assuming this is the correct L2 function
    #             # Update references
    #             proto_dist_calculator = self.model.classifier.similarity_layer.distances
    #             self.feature_distance_fn = self.model.distances
        
    #     # 3. Fallback: Unknown / Custom
    #     else:
    #         print(f"INFO: Unknown distance method ({type(current_method)}). Assuming appropriate L2/Cosine output.")
    #         proto_dist_calculator = current_method
    #         self.feature_distance_fn = self.model.distances
    #         print(self.model.classifier.similarity_layer)
    #         print(self.model.classifier.similarity_layer.distances)
    #         # print(self.model.distances)


    #     # --- PROTOTYPE DISTANCE CALCULATION ---
    #     self.prototype_distances = None
    #     loaded_successfully = False
        
    #     if load_proto and prototype_filepath and os.path.exists(prototype_filepath):
    #         try:
    #             loaded_data = torch.load(prototype_filepath, map_location=self.device)
    #             print(f"Loaded prototype distances from {prototype_filepath}")
    #             self.prototype_distances = loaded_data
    #             if self.prototype_distances.shape == (self.P, self.P):
    #                 loaded_successfully = True
    #             else:
    #                 print(f"Shape mismatch in loaded file. Recalculating.")
    #         except Exception as e:
    #             print(f"Failed to load prototype distances: {e}")

    #     if not loaded_successfully:
    #         print("Calculating prototype distances...")
    #         with torch.no_grad():
    #             protos_for_calc = self.model.classifier.prototypes.to(self.device)
    #             # Compute
    #             dists = proto_dist_calculator(protos_for_calc, protos_for_calc)
    #             self.prototype_distances = dists.detach().squeeze()

    #             # --- FIX: ENFORCE ZERO DIAGONAL ---
    #             # Numerical noise in float32 cdist/sqrt can result in epsilon values (e.g., 0.001)
    #             # We strictly enforce d(p,p) = 0.0
    #             self.prototype_distances.fill_diagonal_(0.0)
    #             # ----------------------------------

    #             # Sanity Check
    #             wrong_prototypes = torch.nonzero(torch.diag(self.prototype_distances) != 0)
    #             if wrong_prototypes.numel() > 0:
    #                 print(f"Warning: {len(wrong_prototypes)} non-zero diagonal elements remaining.")
                
    #             assert torch.all(torch.diag(self.prototype_distances) == 0), "Diagonal must be zero."

    #             # Handle negatives
    #             if torch.any(self.prototype_distances < 0):
    #                 self.prototype_distances = torch.clamp(self.prototype_distances, min=0.0)

    #         if save_proto and prototype_filepath:
    #             try:
    #                 os.makedirs(os.path.dirname(prototype_filepath), exist_ok=True)
    #                 torch.save(self.prototype_distances.cpu(), prototype_filepath)
    #                 print(f"Saved prototype distances to {prototype_filepath}")
    #             except Exception as e:
    #                 print(f"Failed to save prototype distances: {e}")

    #     self.prototype_distances = self.prototype_distances.to(self.device)
        
    #     #--- Final Sanity Checks ---
    #     assert self.prototype_distances.shape == (self.P, self.P), "Final prototype distance shape check failed."
    #     assert torch.all(self.prototype_distances >= 0), "Negative distances detected after clamping!"


    def init_prototype_distances(self, save_proto: bool = False, load_proto: bool = True, prototype_filepath: Optional[str] = None):
        # --- ROBUST DISTANCE FUNCTION SETUP ---
        dist_layer = self.model.classifier.similarity_layer
        
        # We MUST NOT monkey-patch the model's distance function, otherwise we corrupt 
        # the neural network's forward pass (causing cascading square roots).
        # We create isolated functions exclusively for the explainer's geometry math.
        
        def safe_sqrt(tensor):
            return torch.sqrt(torch.clamp(tensor, min=self.epsilon))
            
        if isinstance(dist_layer, SquaredEuclideanDistance):
            self.use_wrapper = True
            print("INFO: Detected SquaredEuclideanDistance. Explainer will compute spatial bounds in true L2 space.")
            
            def proto_dist_calculator(p1, p2):
                return safe_sqrt(dist_layer.distances(features=p1, prototypes=p2))
                
            def feature_distance_fn(x):
                # The model natively returns d^2, we isolate the sqrt here
                return safe_sqrt(self.model.distances(x))
                
            self.feature_distance_fn = feature_distance_fn
            
        elif isinstance(dist_layer, CosineSimilarity):
            self.use_wrapper = False
            print("INFO: Detected CosineSimilarity. Using native metric.")
            
            def proto_dist_calculator(p1, p2):
                return dist_layer.distances(features=p1, prototypes=p2)
                
            def feature_distance_fn(x):
                return self.model.distances(x)
                
            self.feature_distance_fn = feature_distance_fn
            
        else:
            self.use_wrapper = False
            print(f"INFO: Unknown distance method ({type(dist_layer)}). Assuming native metric.")
            
            def proto_dist_calculator(p1, p2):
                return dist_layer.distances(features=p1, prototypes=p2)
                
            def feature_distance_fn(x):
                return self.model.distances(x)
                
            self.feature_distance_fn = feature_distance_fn

        # --- PROTOTYPE DISTANCE CALCULATION ---
        self.prototype_distances = None
        loaded_successfully = False
        
        if load_proto and prototype_filepath and os.path.exists(prototype_filepath):
            try:
                loaded_data = torch.load(prototype_filepath, map_location=self.device)
                print(f"Loaded prototype distances from {prototype_filepath}")
                self.prototype_distances = loaded_data
                if self.prototype_distances.shape == (self.P, self.P):
                    loaded_successfully = True
                else:
                    print("Shape mismatch in loaded file. Recalculating.")
            except Exception as e:
                print(f"Failed to load prototype distances: {e}")

        if not loaded_successfully:
            print("Calculating prototype distances...")
            with torch.no_grad():
                # We must ensure shape is (P, D, 1, 1) for the distance functions
                protos_for_calc = self.model.classifier.prototypes.to(self.device)
                
                # Compute using our isolated calculator
                dists = proto_dist_calculator(protos_for_calc, protos_for_calc)
                self.prototype_distances = dists.detach().squeeze()

                # ENFORCE ZERO DIAGONAL
                self.prototype_distances.fill_diagonal_(0.0)

                # Handle negatives
                if torch.any(self.prototype_distances < 0):
                    self.prototype_distances = torch.clamp(self.prototype_distances, min=0.0)

            if save_proto and prototype_filepath:
                try:
                    os.makedirs(os.path.dirname(prototype_filepath), exist_ok=True)
                    torch.save(self.prototype_distances.cpu(), prototype_filepath)
                    print(f"Saved prototype distances to {prototype_filepath}")
                except Exception as e:
                    print(f"Failed to save prototype distances: {e}")

        self.prototype_distances = self.prototype_distances.to(self.device)
        
        #--- Final Sanity Checks ---
        assert self.prototype_distances.shape == (self.P, self.P), "Final prototype distance shape check failed."
        assert torch.all(self.prototype_distances >= 0), "Negative distances detected after clamping!"
    
    def explain(self, x: torch.Tensor, y: torch.Tensor, verbose: bool = False) -> dict:
        """
        Generate spatial formal explanations for a batch of inputs x with labels y.
        Args:
            x: Input tensor of shape (N, C, H, W) where N is the batch size, C is the number of channels, H is height, and W is width.
            y: True labels of the inputs of shape (N).
            verbose: If True, print additional information.
        
        Returns:
            dict: Explanation dictionary containing pairs of (feature vector, prototype) and their distances.
        """
        # start_batch = time.time()
        # for x_i, y_i in zip(x, y):
        self.explanation = []
        self.explanations_size = []
        self.correct_explanations = []
        self.incorrect_explanations = []
        y_pred = self.model(x)[0].argmax(dim=1)
        prediction_corr = (y_pred == y)
        explanation = self.explain_one(x, y, verbose=verbose)
        
        self.explanation = []
        if isinstance(explanation, dict):
            # If the explanation is a dictionary, we need to change it to a list
            first_key = list(explanation.keys())[0]  # Get the first key
            if isinstance(first_key, tuple):
                for (l, j), d in explanation.items():
                    self.explanation.append((l, j, d))
            else:
                # If the keys are not tuples, we can just append the values
                self.explanation.extend(explanation.values())
        elif isinstance(explanation, list):
            # If it's a list of explanations, extend the current explanation list
            self.explanation = explanation
        else:
            raise TypeError(f"Unexpected type for explanation: {type(explanation)}. Expected dict or list.")
        # self.explanation.append(explanation)
        # end_batch = time.time()
        # if verbose:
        #     print(f"Generated explanations for {len(x)} inputs.")
        #     print(f"Total time taken for batch: {end_batch - start_batch:.2f} seconds")
        
        # deal with the 'explanations_size' attributes
        exp_size = len(self.explanation)
        self.explanations_size.append(exp_size)
        if prediction_corr:
            self.correct_explanations.append(exp_size)
        else:
            self.incorrect_explanations.append(exp_size)
        
        return self.explanation
    
    def explain_one(self, x: torch.Tensor, y: torch.Tensor, verbose: bool = False) -> dict:
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def verify_triangle_inequality(self, x: torch.Tensor, tolerance: float = 1e-6) -> bool:
        """
        Verifies that the triangle inequality holds for distances between
        prototypes and feature vectors using vectorized opnormalized_scalar_prodserations.

        Checks |d(A, C) - d(B, C)| <= d(A, B) <= d(A, C) + d(B, C) for:
            - A, B: Prototypes (from self.prototypes)
            - C: Feature vector (derived from x)

        Args:
            x: Batch of input images, shape (N, C, H_in, W_in).
            tolerance: Numerical tolerance for floating-point comparisons.

        Returns:
            True if the triangle inequality holds for all triplets within
            tolerance, False otherwise.
        """
        # print("Verifying triangle inequality (vectorized)...")
        if self.prototype_distances is None:
            raise RuntimeError("Prototype distances have not been initialized.")

        x = x.to(self.device)
        N = x.size(0)
        # print(f"Input batch size: {N}")
        # --- Check if model is in eval mode ---
        if self.model.training:
            raise RuntimeError("Model must be in eval mode to verify triangle inequality.")
        # --- Check what distance function is used ---
        # print(f"Model distance function: {self.model.distances.__class__}")

        with torch.no_grad():
            # Calculate distances from features to prototypes
            # Expected shape: (N, P, H, W)
            distances_to_features = self.feature_distance_fn(x)  # 
            # distances_to_features = self.model.distances(x)

            # Infer spatial dimensions H, W from the distance map
            _, P_check, H, W = distances_to_features.shape

            # --- Initialization/Validation of H, W ---
            if self.H is None or self.W is None:
                self.H, self.W = H, W
                # print(f"Inferred spatial dimensions H={self.H}, W={self.W}")
            elif self.H != H or self.W != W:
                # This case might occur if the model produces variable output sizes
                print(f"Input batch resulted in different spatial dims ({H}x{W}) than previous ({self.H}x{self.W}). Using current batch's dimensions for this check.")
                # Optionally update self.H, self.W or raise error depending on expected model behavior
                # Sticking with current H, W for this check only:
                # self.H, self.W = H, W

            if P_check != self.P:
                raise ValueError(f"Model distance function returned {P_check} prototype distances, but expected {self.P}")

            # Reshape feature distances for broadcasting: (N, P, H*W)
            d_feat_proto = distances_to_features.reshape(N, self.P, -1)
            num_spatial_locations = H * W

            # Prepare distances for broadcasting:
            # d(p1, p2): (P, P) -> (1, P, P, 1)
            d_p1_p2 = self.prototype_distances.view(1, self.P, self.P, 1)

            # d(p1, z): (N, P, H*W) -> (N, P, 1, H*W)
            d_p1_z = d_feat_proto.unsqueeze(2)

            # d(p2, z): (N, P, H*W) -> (N, 1, P, H*W)
            d_p2_z = d_feat_proto.unsqueeze(1)

            # --- Perform Triangle Inequality Checks ---

            # Check 1: d(p1, p2) <= d(p1, z) + d(p2, z)
            # Broadcasting d_p1_z and d_p2_z yields (N, P, P, H*W)
            # Broadcasting d_p1_p2 yields (N, P, P, H*W)
            check1_valid = (d_p1_p2 <= d_p1_z + d_p2_z + tolerance)

            # Check 2: |d(p1, z) - d(p2, z)| <= d(p1, p2)
            check2_valid = (torch.abs(d_p1_z - d_p2_z) <= d_p1_p2 + tolerance)

            # Combine checks: All must hold for a triplet (n, p1, p2, hw) to be valid
            # Result shape: (N, P, P, H*W)
            all_checks_valid_tensor = check1_valid & check2_valid

            # Check if *all* elements across all dimensions are True
            all_valid = torch.all(all_checks_valid_tensor).item()  # .item() converts 0-dim tensor to Python bool

            # --- Reporting ---
            if all_valid:
                # print("Triangle inequality holds for all checked triplets within tolerance.")
                pass
            else:
                # Count total checks and violations
                # Note: includes p1=p2 cases, which should always be True if tolerance >= 0
                total_checks = N * self.P * self.P * num_spatial_locations
                num_violations = total_checks - torch.sum(all_checks_valid_tensor).item()
                print(f"Triangle inequality VIOLATIONS DETECTED: {num_violations} violations out of {total_checks} checks.")
                print(f"Percentage of violations: {100 * num_violations / total_checks:.2f}%")
                
                # --- Extended Debugging: Find and print first violation details ---
                violating_indices = torch.nonzero(~all_checks_valid_tensor)
                if violating_indices.numel() > 0:
                    # Get indices of the first violation found
                    n_idx, p1_idx, p2_idx, hw_idx = violating_indices[0].tolist()
                    print("-" * 30)
                    print(f"Details for first violation found at:")
                    print(f"  Image index (N): {n_idx}")
                    print(f"  Prototype 1 index (P1): {p1_idx}")
                    print(f"  Prototype 2 index (P2): {p2_idx}")
                    # Calculate H/W coordinates from flattened hw_idx
                    h_idx = hw_idx // W
                    w_idx = hw_idx % W
                    print(f"  Spatial location (H, W): ({h_idx}, {w_idx}) (Flattened index: {hw_idx})")
                    print("-" * 30)

                    # Retrieve the specific scalar distance values involved
                    d_p1_p2_val = self.prototype_distances[p1_idx, p2_idx].item()
                    d_p1_z_val = d_feat_proto[n_idx, p1_idx, hw_idx].item()
                    d_p2_z_val = d_feat_proto[n_idx, p2_idx, hw_idx].item()

                    # Recalculate the bounds for clarity
                    sum_d_pz = d_p1_z_val + d_p2_z_val
                    abs_diff_d_pz = abs(d_p1_z_val - d_p2_z_val)

                    print(f"Values:")
                    print(f"  d(P{p1_idx}, P{p2_idx})       = {d_p1_p2_val:.6f}")
                    print(f"  d(P{p1_idx}, Z)          = {d_p1_z_val:.6f}")
                    print(f"  d(P{p2_idx}, Z)          = {d_p2_z_val:.6f}")
                    print(f"  d(P{p1_idx}, Z) + d(P{p2_idx}, Z) = {sum_d_pz:.6f}")
                    print(f"  |d(P{p1_idx}, Z) - d(P{p2_idx}, Z)| = {abs_diff_d_pz:.6f}")
                    print(f"  Tolerance             = {tolerance}")
                    print("-" * 30)

                    # Check which inequality failed
                    inequality1_holds = (d_p1_p2_val <= sum_d_pz + tolerance)
                    inequality2_holds = (abs_diff_d_pz <= d_p1_p2_val + tolerance)

                    print("Check Results:")
                    if not inequality1_holds:
                        print(f"  FAILED: d(P{p1_idx}, P{p2_idx}) <= d(P{p1_idx}, Z) + d(P{p2_idx}, Z)")
                        print(f"          {d_p1_p2_val:.6f}  >  {sum_d_pz:.6f} + {tolerance}")
                    else:
                        print(f"  PASSED: d(P{p1_idx}, P{p2_idx}) <= d(P{p1_idx}, Z) + d(P{p2_idx}, Z)")

                    if not inequality2_holds:
                        print(f"  FAILED: |d(P{p1_idx}, Z) - d(P{p2_idx}, Z)| <= d(P{p1_idx}, P{p2_idx})")
                        print(f"          {abs_diff_d_pz:.6f}  >  {d_p1_p2_val:.6f} + {tolerance}")
                    else:
                        print(f"  PASSED: |d(P{p1_idx}, Z) - d(P{p2_idx}, Z)| <= d(P{p1_idx}, P{p2_idx})")
                    print("-" * 30)
                # --- End of Extended Debugging ---

        return bool(all_valid)
        # Note: This function does not return the distances, as the main goal is to verify the triangle inequality.
    
    def forward(
        self,
        x: torch.Tensor,  # (N, C, H, W)
        y: torch.Tensor,  # (N)
        top_k: bool = False,
        max_only: bool = False,
        triangle_inequality: bool = False,
        hypersphere_approximation: bool = False,
        verbose: bool = False,
        chkpt_iter: Optional[int] = None,
        check_memory: bool = True,
    ):
        """
        x: torch.Tensor - batch of images
        y: torch.Tensor - batch of labels
        """
        if triangle_inequality or hypersphere_approximation:
            verbose = True  # Enable verbose mode if triangle inequality or hypersphere approximation is used
        
        # Verify triangle inequality before proceeding
        if verbose:
            pass
            # print(f"Verifying triangle inequality for input distances with tolerance {self.epsilon}...")
        if not self.verify_triangle_inequality(x, tolerance=self.epsilon):
            raise ValueError("Triangle inequality does not hold for input distances. Check distance computation.")
            
        # check_memory_usage()
        if check_memory:
            check_memory_usage(threshold_mb=5000)
        
        # XOR on the four booleans
        assert (
            sum([top_k, triangle_inequality, hypersphere_approximation]) == 1
        ), f"Exactly one of the three booleans must be True. \n top_k: {top_k}, triangle_inequality: {triangle_inequality}, hypersphere_approximation: {hypersphere_approximation}"
        x = x.to(self.device)  # (N, C, H, W) # images
        y = y.to(self.device)  # (N) # labels
        if chkpt_iter is not None:
            self.counter_step = chkpt_iter
        else:
            self.counter_step = 100_000  # high number to avoid checkpointing
        self.batch_size = x.size(0)  # N
        self.batch_idx += 1
        with torch.no_grad():
            # distances_with_proto = self.model.distances(x)  # (N, P, H, W)
            distances_with_proto = self.feature_distance_fn(x)  # (N, P, H, W)
            _, self.num_prototypes, self.H, self.W = distances_with_proto.shape
            similarities = self.model.similarities(x)  # (N, P, H, W)

            # similarities_with_proto = self.model.classifier.similarity_layer.distances_to_similarities(distances_with_proto)
            logits = self.model(x)[0]  # (N, K)
            # print(logits.shape)

        # similarities_with_proto = similarities_with_proto.view(batch_size, self.num_prototypes, -1)  # (N, P, H, W) -> (N, P, H*W)
        distances_with_proto = distances_with_proto.view(
            self.batch_size, self.num_prototypes, -1
        )  # (N, P, H, W) -> (N, P, H*W)
        self.distances_with_proto = distances_with_proto
        min_dis, min_dis_idx = torch.min(distances_with_proto, dim=-1)  # (N, P, H*W) -> (N, P)

        z = self.model.features(x)
        # torch.stack([min_dis, min_dis_idx], dim=-1) # (N, P, 2)

        # predicted class
        predicted_class = torch.argmax(logits, dim=1)  # (N)
        predicted_class.squeeze()  # (N)
        # if N == 1
        self.c = predicted_class.item()  # (1)
        # next step:
        self.explanations_size = []
        self.correct_explanations = []
        self.incorrect_explanations = []
        for iidx in range(self.batch_size):  # [1, ..., N]
            self.iidx = iidx
            # print(f"Image {iidx} in the batch")
            # if iidx > 0:
            #     break
            true_label = y[iidx].item()
            # try:
            #     z = self.model.features(x[iidx]).unsqueeze(0) # (1, D, H, W)
            # except ValueError:
            #     z = self.model.features(x[iidx].unsqueeze(0)) # (1, D, H, W)
            
            dis_img, dis_img_idx = distances_with_proto[iidx], min_dis_idx[iidx]  # (P, H*W), (P)
        
            min_dis_img = min_dis[iidx].detach()  # (P)
            min_dis_img_idx = min_dis_idx[iidx].detach()  # (P)
            real_distances = min_dis_img.clone()
            
            # initialize bounds
            feature_lower_bound_distances = torch.zeros((self.H * self.W, self.P), device=self.device)  # (H*W, P)
            feature_upper_bound_distances = torch.ones((self.H * self.W, self.P), device=self.device) * float(
                "inf"
            )  # (H*W, P)
            lower_bound_distances = torch.zeros((self.P), device=self.device)  # (P)
            upper_bound_distances = torch.ones((self.P), device=self.device) * float("inf")  # (P)

            max_similarities = torch.max(similarities[iidx].view(self.num_prototypes, -1), dim=1).values  # (P)
            
            # verification init
            unverified = list(range(self.num_classes))
            selected_class = int(torch.argmax(logits[iidx]).item())  # class predicted
            confidence_scores = logits[iidx]
            self.selected_class = selected_class  # for helper functions (Strategy 2)
            # print("predicted class ", selected_class)
            unverified.remove(selected_class)
            counter = 0
            self.explanation = []  # empty explanation
            strategy_2 = False
            if verbose:
                t = tqdm(total=MAX_EXPLANATIONS)
                t.n = 0
            prediction_corr = (selected_class == true_label)
            prediction_conf = confidence_scores[selected_class]
            warning_counter = 0
            while unverified:  # while unverified != []:
                if verbose:
                    t.n += 1
                # t.refresh()
                # The next prototype is the one that represents the shortest distance (highest similarity) between a feature vector and a prototype among the prototypes not in the explanation.
                # print("Testing:", not (top_k) and not (max_only))
                # --- Strategy 2 (Verification Driven) ---
                # --- Determine next distance ---
                if (triangle_inequality or hypersphere_approximation) and strategy_2:
                    total_num_feature_vectors = self.H * self.W
                    num_explained_pairs = len(self.explanation)

                    if num_explained_pairs < total_num_feature_vectors:
                        # --- Phase 1 Logic ---
                        explained_vectors = [el[0] for el in self.explanation]
                        found_in_phase1 = False
                        for hw in tqdm(range(total_num_feature_vectors)):
                            if hw not in explained_vectors:
                                # Initialize centers/radii if first time and hypersphere
                                if hypersphere_approximation and counter == 0 and hw == 0: # Approx condition
                                    estimated_centers = torch.zeros( (total_num_feature_vectors, self.D), device=self.device)
                                    estimated_radii = torch.zeros( (total_num_feature_vectors, 1), device=self.device)

                                next_proto_dis, next_proto_idx = torch.min(self.distances_with_proto[self.iidx, :, hw], dim=0)
                                next_proto_dis, next_proto_idx = next_proto_dis.item(), next_proto_idx.item()
                                self.explanation.append((hw, next_proto_idx, next_proto_dis))
                                hwref = hw
                                found_in_phase1 = True
                                break
                        if not found_in_phase1:
                            print("Error: Failed to find next pair in Phase 1.")
                            print(self.explanation)
                            break  # Should not happen if H*W > 0

                    else:
                        # --- Phase 2 Logic ---
                        if (num_explained_pairs == total_num_feature_vectors) and verbose:
                            print(f"Phase 2: Selecting next distance via Verification Driven Strategy")

                        # Use Strategy 2
                        next_distance_info = self._find_next_distance_verification_driven(unverified, feature_lower_bound_distances, feature_upper_bound_distances, distances_with_proto)

                        if next_distance_info[0] is None:
                            print("Error: Could not find any next distance to add (Verification Driven Fallback Failed).")
                            unverified = []  # Stop loop gracefully
                            continue      # Skip rest of loop iteration

                        hwref, next_proto_idx, next_proto_dis = next_distance_info
                        # Append to explanation (if not already handled by the function itself - check function design)
                        # Assuming the function only *returns* the choice, we append here
                        self.explanation.append((hwref, next_proto_idx, next_proto_dis))
                # --- Strategy 2 (End) ---
                if (top_k) or (max_only):
                    next_proto_dis, next_proto_idx = torch.min(min_dis_img, dim=0)  # (2)
                    next_proto_dis = next_proto_dis.item()
                    next_proto_idx = next_proto_idx.item()
                    # self.explanation.append(next_proto_idx)
                    self.explanation.append((next_proto_idx, max_similarities[next_proto_idx].item(), next_proto_dis))
                    # important renaming
                    hwref = min_dis_img_idx[next_proto_idx]
                    # dense_explanation.append((hwref, next_proto_idx, next_proto_dis))
                    # update bounds
                    # fix values for that prototype and that feature vector
                    feature_lower_bound_distances[hwref, next_proto_idx] = next_proto_dis  # (1) [d]
                    feature_upper_bound_distances[hwref, next_proto_idx] = next_proto_dis  # (1) [d]
                # update bounds for prototype p and all h,w (all feature vectors)

                # top-k update
                if top_k:
                    idx_to_keep = torch.logical_or(
                        (feature_lower_bound_distances >= next_proto_dis),
                        (feature_upper_bound_distances <= next_proto_dis),
                    )
                    feature_lower_bound_distances = (
                        feature_lower_bound_distances * idx_to_keep + next_proto_dis * torch.logical_not(idx_to_keep)
                    )
                elif max_only:
                    print("MAX_ONLY")
                    # feature_lower_bound_distances[..., next_proto_idx] = torch.maximum(feature_lower_bound_distances[..., next_proto_idx], torch.Tensor(next_proto_dis)) # (H*W, P)
                    feature_lower_bound_distances[..., next_proto_idx] = torch.clamp(
                        feature_lower_bound_distances[..., next_proto_idx], min=next_proto_dis
                    )  # (H*W, P)
                    # triangle inequality :
                    # ||d(p, i) - d(p,z_{h,w})|| <= d(z_{h,w}, i) <= d(p, i) + d(p, z_{h,w})
                    # d(p,z_{h,w}) = next_proto_dis
                    # d(p, i) = self.prototype_distances[next_proto_idx, i]
                    # d_pi = self.prototype_distances[next_proto_idx, :]
                    # feature_lower_bound[hwref, i] <= d(z_{h,w},i) <= feature_upper_bound[hwref, i]
                    prototype_distances = self.prototype_distances[:, next_proto_idx]  # (P)
                    feature_lower_bound_distances[hwref, :] = torch.maximum(
                        feature_lower_bound_distances[hwref, :], torch.abs(prototype_distances - next_proto_dis)
                    )  # (P)
                    assert feature_lower_bound_distances.shape == (
                        self.H * self.W,
                        self.P,
                    ), f"Feature lower bound shape: {feature_lower_bound_distances.shape}"
                    feature_upper_bound_distances[hwref, :] = torch.minimum(
                        feature_upper_bound_distances[hwref, :], prototype_distances + next_proto_dis
                    )  # (P)
                    assert feature_upper_bound_distances.shape == (
                        self.H * self.W,
                        self.P,
                    ), f"Feature upper bound shape: {feature_upper_bound_distances.shape}"
                    assert [
                        (feature_upper_bound_distances[hwref, i] >= feature_lower_bound_distances[hwref, i])
                        for i in range(self.P)
                    ], "Upper bound is less than lower bound"
                elif triangle_inequality and not (strategy_2):
                    # print("TRIANGULAR INEQUALITY") if counter + iidx == 0 else None
                    # unrestricted access to all prototype-feature vector pairs
                    # 1. check that the explanation has at least one prototype-featurevector pair per feature vector (e.g. 1 prototype per feature vector)
                    # 1.a. if not, add the prototype-feature vector pair with the highest similarity score for a feature vector not in the explanation yet.
                    # 1.b. if yes, add the prototype-feature vector pair with the highest similarity score, regardless of the feature vector.
                    # This means we have to redo the 'next_proto_dis' and 'next_proto_idx' computation.
                    total_num_feature_vectors = self.H * self.W
                    num_explained_pairs = len(self.explanation)
                    if num_explained_pairs < total_num_feature_vectors:
                        # what shape is the explanation ?
                        explained_vectors = [el[0] for el in self.explanation]
                        # greedy approach
                        for hw in range(total_num_feature_vectors):
                            if hw not in explained_vectors:
                                # print(f"dists shape: {distances_with_proto[iidx, :, hw].shape}")
                                next_proto_dis, next_proto_idx = torch.min(distances_with_proto[iidx, :, hw], dim=0)
                                next_proto_dis = next_proto_dis.item()
                                next_proto_idx = next_proto_idx.item()
                                self.explanation.append((hw, next_proto_idx, next_proto_dis))
                                # print(self.explanation)
                                hwref = hw
                                if not torch.isfinite(self.prototype_distances).all():
                                    print("NaNs or Infs found in `self.prototype_distances`")
                                break
                    else:
                        (
                            print("Explained all feature vectors at least once")
                            if (num_explained_pairs == total_num_feature_vectors) and verbose
                            else None
                        )
                        # I want the next prototype-feature vector pair with the highest similarity score, with the distance, the prototype, and the feature vector.
                        # next_proto_dis, next_proto_idx = torch.min(min_dis_img, dim=0) # (2) does not work -> I only sample distances that are smallest for each prototype, and I want to sample everything
                        next_proto_dis = float("inf")
                        next_proto_idx = -1
                        for hw in range(total_num_feature_vectors):
                            dis_img_hw, dis_img_idx_hw = (
                                distances_with_proto[iidx, :, hw],
                                min_dis_idx[iidx, hw],
                            )  # (P), (P)
                            next_proto_dis_hw, next_proto_idx_hw = torch.min(dis_img_hw, dim=0)  # (2)
                            next_proto_dis_hw = next_proto_dis_hw.item()
                            next_proto_idx_hw = next_proto_idx_hw.item()
                            if next_proto_dis_hw < next_proto_dis:
                                next_proto_dis = next_proto_dis_hw
                                next_proto_idx = next_proto_idx_hw
                                hwref = hw
                        #
                        #
                        self.explanation.append((hwref, next_proto_idx, next_proto_dis))  # {hw, p, v}
                        postfix_str = f"predicted class: {self.c}, true label: {true_label}, conf: {prediction_conf:.2f}, hwref: {hwref}, next_proto_idx: {next_proto_idx}, next_proto_dis: {next_proto_dis:.2f}, n_unverif: {len(unverified)}, cex conf: {torch.max(unverified_conf):.3f}"
                        if verbose:
                            t.set_postfix_str(postfix_str)
                            t.refresh()
                elif hypersphere_approximation:
                    # print("HYPERSPHERE APPROXIMATION") if counter + iidx == 0 else None
                    total_num_feature_vectors = self.H * self.W
                    num_explained_pairs = len(self.explanation)
                    if num_explained_pairs < total_num_feature_vectors:
                        # what shape is the explanation ?
                        explained_vectors = [el[0] for el in self.explanation]
                        # greedy approach
                        if counter == 0:
                            estimated_centers = torch.zeros(
                                (total_num_feature_vectors, self.D), device=self.device
                            )  # (H*W, D)
                            estimated_radii = torch.zeros(
                                (total_num_feature_vectors, 1), device=self.device
                            )  # (H*W, 1)
                        for hw in range(total_num_feature_vectors):
                            if hw not in explained_vectors:
                                next_proto_dis, next_proto_idx = torch.min(distances_with_proto[iidx, :, hw], dim=0)
                                next_proto_dis, next_proto_idx = next_proto_dis.item(), next_proto_idx.item()
                                # estimated_centers[hw] = self.model.classifier.prototypes[next_proto_idx]  # (D) # center of the hypersphere
                                # estimated_radii[hw] = next_proto_dis  # (1) # radius of the hypersphere
                                self.explanation.append((hw, next_proto_idx, next_proto_dis))
                                hwref = hw
                                break
                    else:
                        (
                            print("Explained all feature vectors at least once")
                            if (num_explained_pairs == total_num_feature_vectors) and verbose
                            else None
                        )
                        # print("Hypersphere Approximation: Selecting next distance via Smallest Scalar Product")
                        # I want the next prototype-feature vector pair with the highest similarity score, with the distance, the prototype, and the feature vector.
                        # next_proto_dis, next_proto_idx = torch.min(min_dis_img, dim=0) # (2) does not work -> I only sample distances that are smallest for each prototype, and I want to sample everything
                        next_proto_dis = float("inf")
                        next_proto_idx = -1
                        choose_closest_point = False
                        choose_largest_angle = True
                        if choose_closest_point:
                            for hw in range(total_num_feature_vectors):
                                dis_img_hw, dis_img_idx_hw = (
                                    distances_with_proto[iidx, :, hw],
                                    min_dis_idx[iidx, hw],
                                )  # (P), (P)
                                next_proto_dis_hw, next_proto_idx_hw = torch.min(dis_img_hw, dim=0)  # (2)
                                next_proto_dis_hw = next_proto_dis_hw.item()
                                next_proto_idx_hw = next_proto_idx_hw.item()
                                if next_proto_dis_hw < next_proto_dis:
                                    next_proto_dis = next_proto_dis_hw
                                    next_proto_idx = next_proto_idx_hw
                                    hwref = hw
                                    
                        elif choose_largest_angle:
                            # print(f"Estimated centers shape: {estimated_centers.shape}")
                            # scalar_prods = estimated_centers @ self.model.classifier.prototypes.squeeze().t()  # (H*W, P) @ (P, D) -> (H*W, P)
                            # scalar_prods = estimated_centers @ self.prototypes.squeeze().t()  # (H*W, P) @ (P, D) -> (H*W, P)
                            
                            # P-C Shape (HW, P, D)
                            centers_to_protos = self.prototypes.squeeze((2,3)).squeeze(0) - estimated_centers.unsqueeze(1) 
                            # print(centers_to_protos.shape)
                            
                            # Z-C Shape # (H*W, 1, D)
                            centers_to_feats = z.flatten(start_dim=2).swapaxes(1,2).swapaxes(0,1) - estimated_centers.unsqueeze(1)
                            # print(centers_to_feats.shape)
                            
                            # Out: (H*W, P)
                            cos = nn.CosineSimilarity(dim=2, eps=1e-6)
                            cos_sim = cos(centers_to_protos, centers_to_feats)
                            # cos_sim = pairwise_cosine_similarity(estimated_centers)
                            # scalar_prods = scalar_prods / (
                            #     estimated_radii.view(total_num_feature_vectors, 1) * self.model.classifier.prototypes.norm(dim=1)
                            # ) # (H*W, P) / (H*W, 1) * (P,) -> (H*W, P)
                            # normalized_scalar_prods = scalar_prods / (torch.norm(estimated_centers, dim=1).view(-1,1) @ torch.norm(self.model.classifier.prototypes.squeeze(), dim=1).view(1,-1))
                            # scalar_prods = torch.abs(normalized_scalar_prods)  # (H*W, P)
                            scalar_prods = torch.abs(cos_sim)  # (H*W, P)
                            # get the hw and p with the smallest scalar product i.e. largest angle + largest distance
                            # print(f"scalar_prods shape: {scalar_prods.shape}")
                            assert scalar_prods.shape == (self.H * self.W, self.P), f"Scalar products shape: {scalar_prods.shape}"
                            for hw in range(total_num_feature_vectors):
                                dis_img_hw, dis_img_idx_hw = (
                                    distances_with_proto[iidx, :, hw],
                                    min_dis_idx[iidx, hw],
                                )  # (P), (P)>
                                # elements in the explanation are characterized by an "inf" distance
                                # we want to choose the next (feature vector, prototype) pair that is not in the explanation yet, thus disregarding the "inf" distances
                                # we want a boolean mask for the distances
                                proto_hw_available = torch.logical_not(dis_img_hw.isinf()) # (P)
                                angle_proto = torch.where(proto_hw_available, scalar_prods[hw,:], torch.zeros_like(scalar_prods[hw,:]))
                                # next_proto_dis_hw, next_proto_idx_hw = torch.min(dis_img_hw, dim=0)  # (2)
                                next_proto_angle_hw, next_proto_idx_hw = torch.max(angle_proto, dim=0)  # (2)
                                next_proto_dis_hw = dis_img_hw[next_proto_idx_hw]  # (1)
                                next_proto_dis_hw = next_proto_dis_hw.item()
                                next_proto_idx_hw = next_proto_idx_hw.item()
                                scalar_prod = scalar_prods[hw, next_proto_idx_hw].item()  #
                                if next_proto_dis_hw < next_proto_dis or (
                                    next_proto_dis_hw == next_proto_dis and scalar_prod < scalar_prods[hwref, next_proto_idx]
                                ):
                                    next_proto_dis = next_proto_dis_hw
                                    next_proto_idx = next_proto_idx_hw
                                    hwref = hw
                        else:
                            raise ValueError("Invalid choice for closest point or largest angle.")
                            
                        # append to explanation
                        self.explanation.append((hwref, next_proto_idx, next_proto_dis))  # {hw, p, v}
                # nuke the prototype-feature vector pair distance
                # set it to infinity
                distances_with_proto[iidx, next_proto_idx, hwref] = float("inf")
                if triangle_inequality:

                    # fix values for that prototype and that feature vector
                    feature_lower_bound_distances[hwref, next_proto_idx] = next_proto_dis  # (1) [d]
                    feature_upper_bound_distances[hwref, next_proto_idx] = next_proto_dis  # (1) [d]

                    # update bounds
                    # we use the triangle inequality to update the bounds, and nothing else
                    feature_lower_bound_distances[hwref, :] = torch.maximum(
                        feature_lower_bound_distances[hwref, :],
                        torch.abs(self.prototype_distances[next_proto_idx, :] - next_proto_dis),
                    )  # (P)
                    feature_lower_bound_distances[hwref, :] = torch.clamp(
                        feature_lower_bound_distances[hwref, :], min=0
                    )  # (P)
                    feature_upper_bound_distances[hwref, :] = torch.minimum(
                        feature_upper_bound_distances[hwref, :],
                        self.prototype_distances[next_proto_idx, :] + next_proto_dis,
                    )  # (P)
                    feature_upper_bound_distances[hwref, :] = torch.clamp(
                        feature_upper_bound_distances[hwref, :], max=float("inf")
                    )  # (P)
                    assert feature_upper_bound_distances.shape == (
                        self.H * self.W,
                        self.P,
                    ), f"Feature upper bound shape: {feature_upper_bound_distances.shape}"
                    assert feature_lower_bound_distances.shape == (
                        self.H * self.W,
                        self.P,
                    ), f"Feature lower bound shape: {feature_lower_bound_distances.shape}"
                    assert [
                        (feature_upper_bound_distances[hwref, i] >= feature_lower_bound_distances[hwref, i])
                        for i in range(self.P)
                    ], "Upper bound is less than lower bound"
                    
                    
                    
                elif hypersphere_approximation:
                    # h = |C_1, C_3| = \frac{1}{2 * |C_1, C_2|} * (r_1^2 - r_2^2 + |C_1, C_2|^2)
                    # r_3 = \sqrt{r_1^2 - h^2}
                    # d(p_i, z_{h,w}) \in [d(p_i, C_3) - r_3, d(p_i, C_3) + r_3] \forall i
                    # print(f"\n --- Debugging hwref={hwref} ---")
                    # # Store old bounds for comparison if needed
                    # old_lower = feature_lower_bound_distances[hwref, :].clone()
                    # old_upper = feature_upper_bound_distances[hwref, :].clone()
                    if len(self.explanation) > total_num_feature_vectors:
                        c1 = self.prototypes[next_proto_idx].squeeze()  # (D)
                        # print(f"c1 shape: {c1.shape}")
                        assert c1.shape == (self.D,), f"Prototype shape: {c1.shape}"
                        c2 = estimated_centers[hwref]  # (D)
                        # print(f"c2 shape: {c2.shape}")
                        assert c2.shape == (self.D,), f"Estimated center shape: {c2.shape}"
                        r1 = next_proto_dis  # (1)
                        # print(f"r1 : {type(r1)}")
                        r2 = estimated_radii[hwref]  # (1)
                        if DEBUG:
                            print(f"r2 shape: {r2.shape}")
                            print(f"r2 : {type(r2)}")
                            print(f"r2 : {r2}")
                        assert r2.shape == (1,), f"Estimated radius shape: {r2.shape}"
                        assert r2 > 0, "Negative radius or zero radius"
                        # print(f"r2 : {type(r2)}")
                        # c1_c2 = torch.norm(c1 - c2)  # (1)
                        # c1_c2 = torch.cdist(c1.unsqueeze(0), c2.unsqueeze(0)) ** 2  # (1)
                        # d_squared = torch.cdist(c1.unsqueeze(0), c2.unsqueeze(0)).pow(2).squeeze() # Ensure scalar d^2
                        d_squared = torch.norm(c1[None] - c2[None], dim=-1).pow(2).squeeze()  # (1) squared distance
                        d_alt_squared = torch.cdist(c1.unsqueeze(0), c2.unsqueeze(0)).pow(2).squeeze()  # (1) squared distance
                        assert torch.isclose(d_squared, d_alt_squared, atol=self.epsilon), f"Distance mismatch: {d_squared} vs {d_alt_squared}"
                        assert d_squared >= 0, f"Negative squared distance: {d_squared}"
                        
                        # Add small epsilon for stability if d can be zero, although assertion should prevent this
                        d = torch.sqrt(d_squared + self.epsilon)  # Calculate d
                        # h = 1 / ((2 * c1_c2) * (r1**2 - r2**2 + c1_c2**2))  # (1)
                        r1_val = r1.item() if torch.is_tensor(r1) else r1  # Ensure scalar
                        r2_val = r2.item() if torch.is_tensor(r2) else r2  # Ensure scalar

                        h = (r1_val**2 - r2_val**2 + d_squared) / (2 * d + self.epsilon) # Add epsilon to denominator for stability
                        h_inv = (r2_val**2 - r1_val**2 + d_squared) / (2 * d + self.epsilon)  # (1)
                        
                        sqrt_argument = r1**2 - h**2
                        # Clamp negative values resulting from float errors to 0 before sqrt
                        r3 = torch.sqrt(torch.relu(sqrt_argument))
                        r3_val = r3.item() if torch.is_tensor(r3) else r3
                        if torch.any(sqrt_argument < 0):
                            print(f"Warning: Clamped negative sqrt argument for r3 at hwref={hwref}")
                        # print(f"r3 : {type(r3)}")
                        unit_vector_c1_c2 = (c2 - c1) / (d + self.epsilon) # Add epsilon
                        unit_vector_c2_c1 = (c1 - c2) / (d + self.epsilon)  # (D)
                        c3 = c1 + h * unit_vector_c1_c2
                        c3_alt = c2 + h_inv * unit_vector_c2_c1  # (D)
                        if not (torch.isclose(c3, c3_alt, atol=self.epsilon).all(), f"Centers mismatch: distance {torch.norm(c3 - c3_alt)}"):
                            print(f"Warning: Centers mismatch at hwref={hwref}, c3={c3}, c3_alt={c3_alt}")
                            warning_counter += 1
                        assert c3.shape == (self.D,), f"Estimated center shape: {c3.shape}"

                        if estimated_radii[hwref] < r3:
                            if warning_counter < 1:
                                print(f"WARNING! previous r3:{estimated_radii[hwref].item()}, new r3:{r3.item()}")
                            warning_counter += 1
                            
                        
                        # estimated_centers[hwref] = c3  # (D)
                        estimated_centers[hwref] = c3_alt  # (D)
                        estimated_radii[hwref] = r3  # (1)
                        
                        postfix_str = (
                            f"predicted={selected_class:03d}, true={true_label:03d}, conf={confidence_scores[selected_class]:.2f}, hw={hwref:02d}, p={next_proto_idx:04d}, r3={r3:.4f}, {f'n_unverif={len(unverified)}'if (len(unverified) > 1 or len(unverified) == 0) else f'unverified class={unverified[0]}'}, cex conf={torch.max(unverified_conf).item():.3f}, warnings={warning_counter}"
                        )
                        
                        t.set_postfix_str(postfix_str)
                        t.refresh()
                        
                    else:
                        estimated_centers[hwref] = self.prototypes[next_proto_idx].squeeze()  # (D)
                        estimated_radii[hwref] = next_proto_dis
                        assert estimated_centers.shape == (
                            self.H * self.W,
                            self.D,
                        ), f"Estimated centers shape: {estimated_centers.shape}"
                        assert estimated_radii.shape == (
                            self.H * self.W,
                            1,
                        ), f"Estimated radii shape: {estimated_radii.shape}"
                        assert estimated_radii[hwref] > 0, "Negative radius or zero radius"

                    # assert that there is no memory issue and the process does not crash
                    assert estimated_centers.shape == (
                        self.H * self.W,
                        self.D,
                    ), f"Estimated centers shape: {estimated_centers.shape}"
                    assert estimated_radii.shape == (
                        self.H * self.W,
                        1,
                    ), f"Estimated radii shape: {estimated_radii.shape}"
                    assert [
                        (estimated_radii[hwref] >= 0) for hwref in range(total_num_feature_vectors)
                    ], "Negative radius"
                    assert [
                        ((torch.cdist(estimated_centers[hwref].unsqueeze(0), self.prototypes[next_proto_idx].squeeze().unsqueeze(0)) ** 2) <= next_proto_dis)
                        for hwref in range(total_num_feature_vectors)
                    ], "Estimated center is not in the hypersphere"
                    # update bounds
                    # we use the hypersphere approximation to update the bounds, and nothing else
                    # the estimated distance between a prototype and a feature vector is in the interval [d(p_i, C_3) - r_3, d(p_i, C_3) + r_3]
                    
                    previous_lower_bound_hwref = feature_lower_bound_distances[hwref, :].clone()
                    previous_upper_bound_hwref = feature_upper_bound_distances[hwref, :].clone()
                    new_lower_bounds_hwref = torch.abs(
                        torch.cdist(self.prototypes.squeeze(), estimated_centers[hwref].unsqueeze(0)).squeeze()
                        - estimated_radii[hwref]
                    )
                    if (new_lower_bounds_hwref == previous_lower_bound_hwref).all():
                        # print("No change in the lower bound")
                        pass
                    # feature_upper_bound_distances[hwref, :] = torch.abs(
                    #     torch.norm(self.prototypes - estimated_centers[hwref])
                    #     + estimated_radii[hwref]
                    # )  # (P)
                    new_upper_bounds_hwref = torch.abs(
                        torch.cdist(self.prototypes.squeeze(), estimated_centers[hwref].unsqueeze(0)).squeeze()
                        + estimated_radii[hwref]
                    )
                    if (new_upper_bounds_hwref == previous_upper_bound_hwref).all():
                        # print("No change in the upper bound")
                        pass
                        
                    # === DEBUGGING ===
                    if DEBUG and (len(self.explanation) > total_num_feature_vectors):
                        print(f"r1={r1}, r2={r2}, d={d}, h={h}")
                        print(f"sqrt_arg = {sqrt_argument}")  # Add this if using clamping fix
                        print(f"r_3 = {r3}")
                        print(f"C_3 = {estimated_centers[hwref]}")
                        new_lower = new_lower_bounds_hwref
                        new_upper = new_upper_bounds_hwref
                        print(f"Calculated new_lower sample: {new_lower[:5]}")
                        print(f"Calculated new_upper sample: {new_upper[:5]}")
                        # Check calculated bounds *before* combining with old ones
                        inconsistent_new = torch.where(new_lower > new_upper + self.epsilon)[0] # Use small tolerance
                        if len(inconsistent_new) > 0:
                            print(">>> Inconsistency found in NEWLY CALCULATED bounds <<<")
                            print("Indices:", inconsistent_new)
                            print("New Lower:", new_lower[inconsistent_new])
                            print("New Upper:", new_upper[inconsistent_new])
                            # Check for NaNs in inputs
                            if torch.isnan(r3): print("r3 is NaN!")
                            if torch.isnan(estimated_centers[hwref]).any(): print("C_3 has NaN!")
                    # === END DEBUGGING ===
                    
                    feature_lower_bound_distances[hwref, :] = torch.maximum(
                        feature_lower_bound_distances[hwref, :], new_lower_bounds_hwref
                    )
                    feature_upper_bound_distances[hwref, :] = torch.minimum(
                        feature_upper_bound_distances[hwref, :], new_upper_bounds_hwref
                    )
                    
                    # fix values for that prototype and that feature vector
                    feature_lower_bound_distances[hwref, next_proto_idx] = next_proto_dis  # (1) [d]
                    feature_upper_bound_distances[hwref, next_proto_idx] = next_proto_dis  # (1) [d]
                    
                    feature_lower_bound_distances = torch.clamp(feature_lower_bound_distances, min=0)  # (H*W, P)
                    feature_upper_bound_distances = torch.clamp(
                        feature_upper_bound_distances, max=float("inf")
                    )  # (H*W, P)
                    assert feature_upper_bound_distances.shape == (
                        self.H * self.W,
                        self.P,
                    ), f"Feature upper bound shape: {feature_upper_bound_distances.shape}"
                    assert feature_lower_bound_distances.shape == (
                        self.H * self.W,
                        self.P,
                    ), f"Feature lower bound shape: {feature_lower_bound_distances.shape}"
                    assert [
                        (feature_upper_bound_distances[hwref, i] >= feature_lower_bound_distances[hwref, i])
                        for i in range(self.P)
                    ], "Upper bound is less than lower bound"
                else:
                    pass
                
                # 1. Identify indices where lower bound exceeds upper bound + tolerance
                inconsistent_indices = torch.where(
                    feature_lower_bound_distances[hwref, :] > feature_upper_bound_distances[hwref, :] + self.epsilon
                )[0]

                # 2. Check if any inconsistencies were found
                if len(inconsistent_indices) > 0:
                    # Log a warning for debugging purposes
                    if verbose:
                        print("Inconsistency found in bounds:")
                        print(f"Lower: {feature_lower_bound_distances[hwref, inconsistent_indices]}")
                        print(f"Upper: {feature_upper_bound_distances[hwref, inconsistent_indices]}")
                        print(f"Indices: {inconsistent_indices.tolist()}")
                    # print(f"Warning: Correcting {len(inconsistent_indices)} inconsistent bounds for hwref={hwref} at indices {inconsistent_indices.tolist()}")
                    # print(f"Problematic Lower: {feature_lower_bound_distances[hwref, inconsistent_indices]}") # Optional: print values before correction
                    # print(f"Problematic Upper: {feature_upper_bound_distances[hwref, inconsistent_indices]}") # Optional: print values before correction

                    # 3. Apply Correction: Set the lower bound equal to the upper bound for these indices.
                    #    This enforces lower <= upper while respecting the upper bound constraint.
                    #    It assumes the upper bound is more likely "correct" or at least a safer limit
                    #    when inconsistency arises.
                    feature_lower_bound_distances[hwref, inconsistent_indices] = feature_upper_bound_distances[hwref, inconsistent_indices]

                    # --- Optional: Verify Correction Immediately ---
                    # If you want to be absolutely sure the correction worked:
                    corrected_lower = feature_lower_bound_distances[hwref, inconsistent_indices]
                    corrected_upper = feature_upper_bound_distances[hwref, inconsistent_indices]
                    if not torch.all(corrected_lower <= corrected_upper + self.epsilon):
                        print(">>> ERROR: Correction failed! Bounds still inconsistent.")
                    #      # Handle this more critical error if needed
                    # --- End Optional Verification ---

                # 4. Perform the final assertion check (which should now always pass if correction logic is sound)
                assert torch.all(feature_lower_bound_distances[hwref, :] <= feature_upper_bound_distances[hwref, :] + self.epsilon), \
                    f"CRITICAL ERROR: Feature bounds inconsistent for hwref={hwref}: lower > upper *even after correction*."

                # 5. Optional: Print or log the corrected bounds for debugging
                if DEBUG:
                    print(f"Corrected Lower: {feature_lower_bound_distances[hwref, :]}")
                    print(f"Corrected Upper: {feature_upper_bound_distances[hwref, :]}")
                
                # update global bounds
                lower_bound_distances = torch.min(feature_lower_bound_distances, dim=0).values  # (H*W, P) -> (P)
                upper_bound_distances = torch.min(feature_upper_bound_distances, dim=0).values  # (H*W, P) -> (P)
                # === Inside the loop, AFTER the global bounds are updated ===

                # === DEBUGGING ===
                if DEBUG:
                    final_lower = feature_lower_bound_distances[hwref, :]
                    final_upper = feature_upper_bound_distances[hwref, :]
                    inconsistent_final = torch.where(final_lower > final_upper + self.epsilon)[0] # Use small tolerance
                    if len(inconsistent_final) > 0:
                        print(">>> Inconsistency found AFTER update <<<")
                        print("Indices:", inconsistent_final)
                        print("Final Lower:", final_lower[inconsistent_final])
                        print("Final Upper:", final_upper[inconsistent_final])
                        print("Gap:", final_upper[inconsistent_final] - final_lower[inconsistent_final])
                # === END DEBUGGING ===
                
                # === Assertions for correctness and consistency ===
                # Note: These assertions should be removed or replaced with logging in production code.
                
                # 1. Check internal consistency of bounds for the specific feature vector updated (hwref)
                #    (This was already present for triangle_inequality, ensure it's there/correct for hypersphere too)
                # assert torch.all(feature_lower_bound_distances[hwref, :] <= feature_upper_bound_distances[hwref, :]), \
                #     f"Feature bounds inconsistent for hwref={hwref}: lower > upper."
                assert torch.all(feature_lower_bound_distances[hwref, :] <= feature_upper_bound_distances[hwref, :] + self.epsilon), \
                    f"Feature bounds inconsistent for hwref={hwref}: lower > upper (within tolerance)."
                # 2. Check internal consistency of the derived *global* bounds
                #    lower_bound_distances[p] = min_hw(lower_bound(hw, p))
                #    upper_bound_distances[p] = min_hw(upper_bound(hw, p))
                #    We expect the true minimum distance (real_distances[p]) to be between these.
                # if not torch.all(lower_bound_distances <= upper_bound_distances + 0.1*self.epsilon):
                #     print("Lower bounds: ", lower_bound_distances)
                #     print("Upper bounds: ", upper_bound_distances)
                #     print("Real distances: ", real_distances)
                assert torch.all(lower_bound_distances <= upper_bound_distances + self.epsilon), \
                    f"Global bounds inconsistent: min_hw(lower) > min_hw(upper) for some prototype p." \
                    # Note: While feature-wise lower<=upper holds, min(lower)<=min(upper) is also expected
                    # because real_distances[p] must lie between them (see next assertion).

                # 3. Check correctness: The known true minimum distance must be within the derived global bounds
                #    `real_distances` was calculated as `min_dis[iidx].detach().clone()`, which holds the
                #    true minimum distance between prototype p and *any* feature vector for image iidx.
                torch.set_printoptions(precision=10)
                assert torch.all(lower_bound_distances <= real_distances + self.epsilon), \
                    f"Correctness violation: True minimum distance is below calculated global lower bound.\n" \
                    f"Lower bounds: {lower_bound_distances[lower_bound_distances > real_distances + self.epsilon]}\n" \
                    f"Real distances: {real_distances[lower_bound_distances > real_distances + self.epsilon]}"
                    # Added tolerance self.epsilon for floating point comparisons

                assert torch.all(real_distances <= upper_bound_distances + self.epsilon), \
                    f"Correctness violation: True minimum distance is above calculated global upper bound.\n" \
                    f"Real distances: {real_distances[real_distances > upper_bound_distances + self.epsilon]}\n" \
                    f"Upper bounds: {upper_bound_distances[real_distances > upper_bound_distances + self.epsilon]}"
                    # Added tolerance self.epsilon

                # 4. Keep the check that the currently sampled distance is valid relative to the true minimum
                #    This ensures the sampling process itself isn't flawed.
                assert next_proto_dis >= real_distances[next_proto_idx].item() - self.epsilon, \
                    f"Sampled distance {next_proto_dis} is smaller than the true minimum distance " \
                    f"{real_distances[next_proto_idx].item()} for prototype {next_proto_idx}."
                    # Added tolerance
                
                # ---
                # sanity check : make sure that the actual distance is within the bounds
                sanity_check = False
                if (hypersphere_approximation or triangle_inequality) and sanity_check:
                    if len(self.explanation) > total_num_feature_vectors:
                        # print(f"{len(self.explanation)} prototype-feature vector pairs chosen")
                        for pidx in range(self.P):
                            # print(f"Prototype {pidx}")
                            # print(f"Lower bound: {lower_bound_distances[pidx]}")
                            # print(f"Upper bound: {upper_bound_distances[pidx]}")
                            # Actual distance : minimum distance between the prototype and any feature vector
                            kept_distance = real_distances[pidx].item()
                            # assert (
                            #     lower_bound_distances[pidx] <= kept_distance <= upper_bound_distances[pidx]
                            # ), f"Actual distance is not within the bounds for prototype {pidx}\n{lower_bound_distances[pidx]} <= {kept_distance} <= {upper_bound_distances[pidx]}"
                            if torch.abs(lower_bound_distances[pidx] - kept_distance) < self.epsilon:
                                print(f"Lower bound is close to the actual distance for prototype {pidx}")
                                lower_bound_distances[pidx] = kept_distance
                            if torch.abs(upper_bound_distances[pidx] - kept_distance) < self.epsilon:
                                print(f"Upper bound is close to the actual distance for prototype {pidx}")
                                upper_bound_distances[pidx] = kept_distance
                # ---
                # verify conditions
                weights = self.weights  # (P, K) # last layer of the ProtoPNet architecture
                lower_bound_sim = self.model.classifier.similarity_layer.distances_to_similarities(
                    upper_bound_distances
                )  # biggest distance -> smallest similarity
                upper_bound_sim = self.model.classifier.similarity_layer.distances_to_similarities(
                    lower_bound_distances
                )  # smallest distance -> biggest similarity
                # treat nans as zero
                lower_bound_sim = torch.nan_to_num(lower_bound_sim, 0.0)
                # lower_bound_sim *= torch.logical_not(lower_bound_sim.isnan) + 0*torch.isnan(lower_bound_sim)
                # lower_bound, upper_bound # (N, P)
                with torch.no_grad():
                    predicted_class_weights = weights[selected_class]  # (P)
                    batch_selector = weights > predicted_class_weights  # (P, K) # boolean matrix
                    similarities_to_check = upper_bound_sim * batch_selector + lower_bound_sim * (
                        torch.logical_not(batch_selector)
                    )  # (P, K)
                    # res(i) = upper_bound_sim(i) if w_{i,k} > w_{i,c} else lower_bound_sim(i)
                    # similarities_to_check = similarities_to_check[:, unverified] # only classes that were not verified beforehand # actually NO
                    # check = similarities_to_check.swapaxes(0,1) # (K, P)
                    # decision_output = self.decision_head(similarities_to_check)  # (K, K)
                    # Perform manual linear pass using the corrected (200, 202) effective weights
                    # similarities_to_check is (K, P) and self.weights is (K, P)
                    # We want output (K, K), so we multiply (K, P) @ (P, K)
                    decision_output = torch.matmul(similarities_to_check, self.weights.t())  # (K, K)
                
                # print(decision_output.shape)
                new_unverified = unverified.copy()
                unverified_conf = torch.zeros(self.num_classes)
                for uidx in unverified:
                    if decision_output[uidx, selected_class] > decision_output[uidx, uidx]:
                        new_unverified.remove(uidx)
                    else:
                        unverified_conf[uidx] = decision_output[uidx, uidx]
                unverified = new_unverified

                
                # deactivate prototype
                min_dis_img[next_proto_idx] = torch.Tensor([float("inf")])
                # print(f"Iteration {counter}, unverified length = {len(unverified)}")
                counter += 1
                # Free up memory
                if self.device == "cuda:0":
                    # print("Freeing up memory")
                    torch.cuda.empty_cache()
                if (counter % self.counter_step == 0):
                    if verbose:
                        print(f"Iteration {counter}, unverified length = {len(unverified)}")
                        if len(unverified) < 10:
                            print("Unverified classes: ", unverified)
                if (counter > self.max_explanations) and verbose:
                    print("Max explanations reached")
                    print("Unverified classes: ", unverified)
                    for hw in range(total_num_feature_vectors):
                        explanation_hw = [el for el in self.explanation if el[0] == hw]
                        print(f"Feature vector {hw} has {len(explanation_hw)} prototype-feature vector pairs")
                        # print(explanation_hw, "\n", "-" * 50)
                    # save as csv
                    break
            # np.save(f"top{counter}_blazingly_fast", upper_bound_sim.detach().cpu().numpy())
            # np.save(f"explainer_{counter}_blazingly_fast", np.asarray(self.explanation))
            
            # np.save(
            #     f"overflows/explanation_{self.batch_size*self.batch_idx + iidx}_blazingly_fast",
            #     np.asarray(self.explanation),
            # )
            exp_size = len(self.explanation)
            if verbose:
                t.close()
                # print(f"Explained with {exp_size}  distances")
            # self.explanations_size.append(exp_size)
            # if prediction_corr:
            #     self.correct_explanations.append(exp_size)
            # else:
            #     self.incorrect_explanations.append(exp_size)

    def _create_counterexample(self, k: int, output_type=torch.Tensor):
        """
        create a counterexample for class k vs class c
        """
        # N <- batch_size
        # verified <- previous class already checked (initially empty)
        # selected class # (N,)
        # weights # (P, K)
        # lower_bound, upper_bound # (N, P)
        # selected_weights = torch.index_select(weights, dim=1, index=selected_class).swapeaxes(0,-1).unsqueeze(-1) # (N, P, 1)
        # batch_selector = weights.unsqueeze(0) > selected_weights # (N, P, K)
        # res = upper_bound.unsqueeze(-1)*batch_selector + lower_bound.unsqueeze(-1)*(torch.logical_not(batch_selector)) # (N, P, K)
        #
        # for iidx in range(N):
        #       if iidx in verified:
        #           continue
        #       check = res[iidx].swapaxes(0,1)
        #       decision_output = torch.matmul(check, weights) ##% decision_head(check) ## in case of bias=True
        #       predicted_classes = set(torch.argmax(decision_output,1).detach().cpu().numpy())
        #       if predicted_classes == selected_class[iidx]:
        #           verified.append(iidx)
        #
        # --> verified, friends who shall not be named

        if k == self.c:
            return None
        # create a counterexample for class i vs class c
        s = np.zeros(self.num_prototypes)
        # for all p in P
        for i in range(self.num_prototypes):
            # if w_k,i - w_c,i >= 0, then set s_i = upper_bound[i]
            # if w_k,i - w_c,i < 0, then set s_i = lower_bound[i]
            # if self.weights[k][i] - self.weights[self.c][i] >= 0:
            if self.weights[k][i] >= self.weights[self.c][i]:
                # w_k,i >= w_c,i
                s[i] = self.upper_bound[i]  # s_i \in [lb_i, ub_i]
            else:
                s[i] = self.lower_bound[i]
        # convert to tensor or numpy array
        if output_type == torch.Tensor:
            s = torch.tensor(s, dtype=torch.float32)
        elif output_type == np.ndarray:
            s = np.array(s, dtype=np.float32)

        return s

    def verify(self, candidates=None) -> bool | list[int]:
        """
        construct (K-1) counterexamples (K = number of classes)
        for class k != c, construct a sample that maximizes the logit of class k vs c (positive counterexample)
        for each counterexample, check that class c has the highest logit
        """
        verification = True
        if candidates is None:
            candidates = list(range(self.num_classes))  # K classes
            output_candidates = False
        else:
            output_candidates = True
            new_candidates = candidates.copy()
        for k in candidates:
            if k == self.c:
                continue
            adv = self._create_counterexample(k)
            # print(adv)
            adv = adv.to(self.device)
            # check that class c has the highest logit
            logits = self.decision_head(adv)
            logit_k = logits[k]
            logit_c = logits[self.c]
            if logit_k > logit_c:
                verification = False
                if output_candidates is False:
                    break
            else:
                # a class that verifies the explanation will verify for next steps.
                if output_candidates is True:
                    new_candidates.remove(k)
        # print("verification: ", verification)
        if output_candidates is True:
            return new_candidates
        return verification

    def _find_best_top_k(self, start: int = 50, step: int = 10, max: int = 100):
        for k in range(start, max, step):
            # print("trying top-" + str(k) + " explanations")
            self._update_top_k_sim(k)
            if self.verify():
                # print("verified with top-" + str(k) + " explanations")
                return k
        return None

    def top_k_explanations(self, k: int = 10):
        """
        find the top k explanations for the prediction of the model
        """
        similarities = self.model.classifier.similarity_layer(self.z, self.model.classifier.prototypes)  # (1, P, H, W)
        # flatten the similarities tensor to (1, P, H*W)
        similarities = similarities.view(1, self.model.classifier.num_prototypes, -1)
        max_similarities = (torch.max(similarities, dim=2))[0]  # (1, P)
        # print("similarities shape (after flatten): ", similarities.shape)  # (1, P, H*W)
        # print("max_similarities shape: ", max_similarities.shape)  # (1, P)
        ordered_similarities = torch.argsort(max_similarities, descending=True)
        top_k_prototypes = ordered_similarities[:k]
        # print("top k prototypes: ")
        for i in range(k):
            el = top_k_prototypes[0][i].item()
            # print(el, max_similarities[0][el].item())
        return [(el.item(), max_similarities[0][el].item()) for el in top_k_prototypes[0][:k]]

    def _update_top_k_sim(self, k: int = 10):
        """
        update the similarities with the top_k_explanation.
        """
        top_k_explanations = self.top_k_explanations(k)
        top_k_dict = {el[0]: el[1] for el in top_k_explanations}
        # print("top k dict: ", top_k_dict)
        min_sim = top_k_explanations[-1][1]
        # print("min similarity: ", min_sim)
        # for all prototypes *not* in the top k explanations, set the upper bound to the minimum similarity
        # for all prototypes in the top k explanations, set the upper bound and lower bound to the similarity score
        # print("num prototypes: ", self.model.classifier.num_prototypes)
        for i in range(self.model.classifier.num_prototypes):
            if i not in [el[0] for el in top_k_explanations]:
                self.upper_bound[i] = min_sim
            else:
                self.upper_bound[i] = self.lower_bound[i] = top_k_dict[i]
        # print("upper bound: ", self.upper_bound)
        # print("lower bound: ", self.lower_bound)

    def _max_similarity(self):
        """
        For each prototype, get the maximum similarity score obtainable and the associated height and width of the feature vector.
        Computed once and stored in self.max_similarities.

        Returns
        for p in P, v = max {sim(p, z_ij)}
        [(p, v, (h, w)) ...]
        """
        # avoid recomputing the max similarities
        if self.max_similarities is not None:
            return self.max_similarities

        similarities = self.model.classifier.similarities(self.z).squeeze()  # (1, P, H, W) -> (P, H, W)
        # print("similarities shape: ", similarities.shape)
        similarities = similarities.view(self.num_prototypes, -1)  # (P, H, W) -> (P, H*W)

        max_sim, max_sim_idx = torch.max(similarities, dim=1)  # (P, H*W) -> (P)
        max_sim = max_sim.cpu().detach().numpy()
        max_sim_idx = max_sim_idx.cpu().detach().numpy()
        # max_sim_idx = torch.argmax(similarities, dim=1).cpu().detach().numpy()  # (P, H*W) -> (P)
        # print("max sim idx shape: ", max_sim_idx.shape)
        # change the indices to height and width
        max_sim_idx = [(idx // self.H, idx % self.W) for idx in max_sim_idx]  # DO NOT DO THAT

        obj = zip(range(self.num_prototypes), max_sim, max_sim_idx)
        max_similarities = list(obj)
        self.max_similarities = max_similarities
        return max_similarities
    
    def _find_next_distance(self):
        # has been tensorized (kind of)
        """
        Find the next distance to add to the explanation.
        The next distance is the one that represents the shortest distance (highest similarity) between a feature vector and a prototype among the prototypes
        not in the explanation.
        E = [{h_1,w_1, p_1, v_1}, {h_2,w_2, p_2, v_2}, ..., {h_n,w_n, p_n, v_n}] is the current explanation
        where h_i, w_i are the height and width of the feature vector, p_i is the prototype index, and v_i is the value of the similarity score.
        """
        possible_prototypes = [i for i in range(self.num_prototypes) if i not in [el[2] for el in self.E]]
        explainands = self._max_similarity()
        # tensorize max_sim # (N, P, 2)

        # max_index = torch.argmax(max_sim[...,0], dim=1, keepdims=true).unsqueeze(-1).tile((1,1,2)) # (N,1,2)
        # next_proto = torch.gather(max_sim, 1, max_index) # (N,1,2)
        # for iidx, v in enumerate(torch.argmax(max_sim[...,0], dim=1)):
        #   max_sim[iidx,v,0] = -1 # nuked prototypes
        # ...
        # ...
        # torch.cat(explanations)
        next_sim = 0
        next_explainand = None

        for j in possible_prototypes:
            _, sim, (h, w) = explainands[j]
            if sim > next_sim:
                next_sim = sim
                next_explainand = (h, w, j, sim)

        # self.explanation.append(next_explainand)
        return next_explainand

    def _sim_to_distance(self, sim):
        """
        sim = log_e ((dist + 1) / (dist + eps)) ## not log_2
        from sim we want dist
        """
        # exp_sim = 2 ** sim
        exp_sim = torch.exp(torch.Tensor([sim]))  # e^s and not 2^s
        assert torch.equal(
            torch.Tensor([sim]), torch.log(exp_sim)
        ), f"Similarity is {sim} instead of {torch.log(exp_sim)}"
        try:
            epsilon = self.model.classifier.similarity_layer.stability_factor
        except AttributeError:
            epsilon = 1e-4
        dist = (1 - exp_sim * epsilon) / (exp_sim - 1)  # recall: s = log_e((d+1)/(d+eps))
        if dist < 0:
            dist = 0
        return np.array(dist)

    def _triangle_inequality(self, p: int, i: int, v: float):
        """
        Triangle inequality: ||AB - AC|| <= BC <= AB + AC
        In the context of the explanation, we have:
        * d(p, z_{h,w}) = v
        *||d(p, i) - v|| <= d(z_{h,w}, i) <= d(p, i) + v
        """
        # v : similarity
        # d(p, i) : distance
        # d_v : associated distance
        d_v = self._sim_to_distance(v)
        assert d_v >= 0, f"Distance is negative: {d_v}"
        sim_v = self.model.classifier.similarity_layer.distances_to_similarities(torch.Tensor([d_v]))[0].item()
        assert sim_v == v, f"Similarity is {sim_v} instead of {v}"
        # intermediate output : d_lb, d_ub
        # final output : s_ub, s_lb
        d_pi = self.prototype_distances[p, i]
        assert d_pi >= 0, f"Distance between prototype {p} and prototype {i} is negative: {d_pi}"
        d_lb = np.abs(d_pi - d_v)
        d_ub = d_pi + d_v
        assert (
            d_ub >= d_lb
        ), f"Upper bound {d_ub} is less than lower bound {d_lb}. \nUsing d_v = {d_v}, d_pi = {d_pi}, v = {v} with p = {p}, i = {i} and c = {self.c}. \nThe similarity layer is {self.model.classifier.similarity_layer.__class__} with eps = {self.model.classifier.similarity_layer.stability_factor}"
        s_ub, s_lb = self.model.classifier.similarity_layer.distances_to_similarities(torch.Tensor([d_lb, d_ub]))
        assert (
            s_ub >= s_lb
        ), f"Upper bound {s_ub} is less than lower bound {s_lb}. \nUsing d_v = {d_v}, d_pi = {d_pi}, v = {v} with p = {p}, i = {i} and c = {self.c}. \nThe similarity layer is {self.model.classifier.similarity_layer.__class__} with eps = {self.model.classifier.similarity_layer.stability_factor}"
        assert (
            s_ub <= self.max_sim
        ), f"Upper bound {s_ub} is greater than the maximum similarity score {self.max_sim}. \nUsing d_v = {d_v}, d_pi = {d_pi}, v = {v} with p = {p}, i = {i} and c = {self.c}. \nThe similarity layer is {self.model.classifier.similarity_layer.__class__} with eps = {self.model.classifier.similarity_layer.stability_factor}"
        assert (
            s_lb >= 0
        ), f"Lower bound {s_lb} is negative. \nUsing d_v = {d_v}, d_pi = {d_pi}, v = {v} with p = {p}, i = {i} and c = {self.c}. \nThe similarity layer is {self.model.classifier.similarity_layer.__class__} with eps = {self.model.classifier.similarity_layer.stability_factor}"
        # I WANT TO DIE
        # d_pi < 0 <- WHY IS IT NEGATIVE ?
        # HOW ?
        return s_lb, s_ub

    def _update_bounds(self):
        """
        Update the similarity scores bounds for each prototype.
        """
        # NOTE: can be (and should be) tensorized
        for i in range(self.num_prototypes):
            lb = np.max(self.feature_lower_bound[:, :, i], axis=(0, 1))  # (H, W) -> ()
            ub = np.max(self.feature_upper_bound[:, :, i], axis=(0, 1))  # (H, W) -> ()
            self.lower_bound[i] = lb
            self.upper_bound[i] = ub

    def _update_formal_explanation(self):
        """
        At each step, update the formal explanation with the next distance.
        With the next distance, update the lower bounds and upper bounds for each feature vector - prototype similarity score.
        Update the similarity scores bounds for each prototype.
        """
        # Tensorize
        # next_proto  # (N,1,2)

        next_distance = self._find_next_distance()  # (h, w, p, v)
        self.explanation.append(next_distance)  # add the next distance to the explanation
        h, w, p, v = next_distance
        # for prototype p, for each feature vector, update the upper bound to v if v is less than the current upper bound.
        self.feature_lower_bound[h, w, p] = v
        self.feature_upper_bound[h, w, p] = v

        # trying to tensorize
        # for i, j in range(self.H, self.W):
        #       self.feature_upper_bound[i,j,p] = np.min([self.feature_upper_bound[i,j,p], v])
        # how do we do that?
        # with homogeneous tensors
        for i in range(self.H):
            for j in range(self.W):
                self.feature_upper_bound[i, j, p] = np.min([self.feature_upper_bound[i, j, p], v])
        # self.feature_upper_bound[...,p] = np.min(self.feature_upper_bound[...,p], v)
        assert np.max(self.feature_upper_bound[..., p]) == v

        # update the lower bounds and upper bounds for each feature vector - prototype similarity score for h, w.
        for i in range(self.num_prototypes):
            if i == p:
                continue
            lb, ub = self._triangle_inequality(p, i, v)  # (lb, ub) = (||d_{p, i} - v||, d_{p, i} + v)
            # known : d(p, i) ; d(p, z_{h,w}) = v
            # unknown : d(i, z_{h,w})
            # lower bound : ||d(p, i) - v||
            # upper bound : d(p, i) + v
            self.feature_lower_bound[h, w, i] = np.maximum(self.feature_lower_bound[h, w, i], lb)
            self.feature_upper_bound[h, w, i] = np.minimum(self.feature_upper_bound[h, w, i], ub)
            # if lb > self.feature_lower_bound[h, w, i]:
            #     self.feature_lower_bound[h, w, i] = lb
            # if ub < self.feature_lower_bound[h, w, i]:
            #     self.feature_lower_bound[h, w, i] = ub
        # update the similarity scores bounds for each prototype.
        self._update_bounds()  # TBD - update the bounds for each prototype # (max(feature_lower_bound[h,w,p]), max(feature_upper_bound[h,w,p])) for each prototype p
        assert self.lower_bound[p] == v, f"Lower bound for prototype {p} is {self.lower_bound[p]} instead of {v}"
        assert self.upper_bound[p] == v, f"Upper bound for prototype {p} is {self.upper_bound[p]} instead of {v}"

    def _generate_explanation(self, verbose: bool = False):
        """
        Generate the formal explanation for the prediction of the model.
        """
        candidates = list(range(self.num_classes))  # K = set of classes
        candidates.remove(self.c)  # K \ {c}
        while len(candidates) > 0:
            self._update_formal_explanation()
            candidates = self.verify(candidates)
            # if self.verify():
            #     break
        return self.explanation

    def _print_max_similarity(self):
        """
        print the maximum similarity score obtainable (when the input is a prototype) and the associated height and width of the feature vector.
        """
        max_sim = self._max_similarity()
        for idx, sim, (h, w) in max_sim:
            print(f"Prototype {idx}: {sim} at ({h}, {w})")

    def _print_explanation(self):
        """
        print the formal explanation for the prediction of the model.
        """
        print("Formal explanation for class ", self.c)
        for h, w, p, v in self.explanation:
            print(f"Feature vector at ({h}, {w}) has a similarity score of {v} with prototype {p}")
        # print("Explanation size: ", len(self.explanation))

    def __repr__(self) -> str:
        return f"Formal explanation for class {self.c}:\n{self.explanation}"

    def __str__(self) -> str:
        return f"Formal explanation for class {self.c}:\n{self.explanation}"

    
    ### Fallback methods for distance finding ###
    # === Strategy 1 ===
    def _find_minimum_distance_not_in_E(self, indices_in_E_mask, distances_with_proto):
        """
        Fallback: Find the (hw, p) pair not in E with the minimum true distance.
        Args:
            indices_in_E_mask: Boolean tensor (H*W, P) where True means already in E.
        Returns:
            tuple: (best_hw, best_p, min_dist) or (None, None, None) if all are masked.
        """
        print("Fallback: Searching for minimum distance not in E.")
        # Get true distances for the current image, shape (P, H*W)
        true_distances_img = distances_with_proto[self.iidx, :, :]
        # Transpose to (H*W, P) to match the mask
        true_distances_img_hw_p = true_distances_img.T

        # Apply the mask: set distances for pairs already in E to infinity
        masked_distances = torch.where(
            indices_in_E_mask,
            torch.tensor(float('inf'), device=self.device),
            true_distances_img_hw_p
        )

        if torch.all(torch.isinf(masked_distances)):
            print("Fallback Warning: All distances are masked or infinite.")
            return None, None, None  # Should not happen if P*H*W > 0

        # Find the minimum value in the masked distances
        min_dist = torch.min(masked_distances)
        # Find the indices of this minimum value
        min_indices = torch.where(masked_distances == min_dist)
        # Take the first occurrence if there are multiple minimums
        best_hw = min_indices[0][0].item()
        best_p = min_indices[1][0].item()

        print(f"Fallback found: hw={best_hw}, p={best_p}, dist={min_dist.item()}")
        return best_hw, best_p, min_dist.item()

    def _find_next_distance_max_uncertainty(self, E, feature_lower_bound_distances, feature_upper_bound_distances, distances_with_proto):
        """
        Find the next distance (hw, p, dist) to add to the explanation E
        by selecting the pair (hw, p) not currently in E that has the
        maximum finite uncertainty in its distance bounds.
        Uses _find_minimum_distance_not_in_E as fallback.

        Returns:
            tuple: (best_hw, best_p, true_dist) or (None, None, None) if fallback fails.
        """
        num_hw = self.H * self.W
        num_p = self.num_prototypes

        # 1. Create a mask of pairs already in the explanation E
        # Initialize mask with False
        indices_in_E_mask = torch.zeros((num_hw, num_p), dtype=torch.bool, device=self.device)
        if E:  # Check if E is not empty
            # Assuming E stores tuples (hw, p, dist)
            # Convert E to tensor indices for efficient masking
            hw_indices = torch.tensor([el[0] for el in E], device=self.device, dtype=torch.long)
            p_indices = torch.tensor([el[1] for el in E], device=self.device, dtype=torch.long)
            indices_in_E_mask[hw_indices, p_indices] = True

        # 2. Calculate uncertainty (upper_bound - lower_bound)
        uncertainty = feature_upper_bound_distances - feature_lower_bound_distances

        # 3. Create mask for invalid uncertainty values (NaN, Inf) or those already in E
        valid_uncertainty_mask = torch.isfinite(uncertainty) & (~indices_in_E_mask)

        # 4. Check if any valid uncertain pairs remain
        if not torch.any(valid_uncertainty_mask):
            print("MaxUncertainty: No valid finite uncertain pairs remaining.")
            # Call fallback using the mask of pairs already in E
            return self._find_minimum_distance_not_in_E(indices_in_E_mask, distances_with_proto)

        # 5. Find the maximum uncertainty among valid pairs
        # Apply mask: set invalid/masked uncertainties to a very low value
        masked_uncertainty = torch.where(
            valid_uncertainty_mask,
            uncertainty,
            torch.tensor(float('-inf'), device=self.device)
        )

        # Find the maximum value and its flattened index
        max_unc_value = torch.max(masked_uncertainty)

        # Check if the max value found is still -inf (means only invalid pairs were left somehow)
        if torch.isinf(max_unc_value) and max_unc_value < 0:
            print("MaxUncertainty: Max uncertainty value is -inf after masking.")
            return self._find_minimum_distance_not_in_E(indices_in_E_mask)

        # Find the index of the maximum uncertainty
        # Using argmax can be tricky if multiple elements share the max value.
        # Find all indices matching the max value and pick the first one.
        max_indices = torch.where(masked_uncertainty == max_unc_value)
        best_hw = max_indices[0][0].item()
        best_p = max_indices[1][0].item()

        # 6. Retrieve the true distance for the selected pair (best_hw, best_p)
        # Ensure indexing matches the shape of self.distances_with_proto (N, P, H*W)
        true_dist = distances_with_proto[self.iidx, best_p, best_hw].item()

        # print(f"MaxUncertainty found: hw={best_hw}, p={best_p}, uncertainty={max_unc_value.item()}, true_dist={true_dist}") # Optional debug print
        return best_hw, best_p, true_dist
     # === Strategy 2: Verification-Driven Selection ===

    def _find_next_distance_verification_driven(self, unverified, feature_lower_bound_distances, feature_upper_bound_distances, distances_with_proto, verbose=False):
        """
        Find the next distance (hw, p, dist) using Strategy 2: Verification-Driven Selection.
        Prioritizes pairs that most likely help eliminate the 'closest' unverified class.
        Uses _find_next_distance_max_uncertainty as a fallback.

        Returns:
            tuple: (best_hw, best_p, true_dist) or (None, None, None) if fallback fails.
        """
        if not unverified:
            print("VerificationDriven: No unverified classes left.")
            # Or maybe just return None, None, None ? Depends on how the loop handles empty unverified
            return self._find_next_distance_max_uncertainty(self.explanation, feature_lower_bound_distances, feature_upper_bound_distances, distances_with_proto) # Fallback just in case

        # --- Calculate Global Bounds ---
        # Note: These are calculated later in the main loop anyway, recalculating
        # here adds overhead but ensures this function uses the latest state *before*
        # the new distance is chosen. Consider passing them if already computed.
        lower_bound_distances = torch.min(feature_lower_bound_distances, dim=0).values
        upper_bound_distances = torch.min(feature_upper_bound_distances, dim=0).values

        # --- Convert to Global Similarity Bounds ---
        # Remember: low distance -> high similarity
        s_lower_global = self.model.classifier.similarity_layer.distances_to_similarities(upper_bound_distances)
        s_upper_global = self.model.classifier.similarity_layer.distances_to_similarities(lower_bound_distances)

        # Handle potential NaNs/Infs from conversion (e.g., dist=inf -> sim=0, dist=0 -> sim=1 or max)
        # Ensure sim bounds are reasonable, e.g., clamp between 0 and 1 or expected range.
        # Note: nan_to_num happens later in your main loop, but maybe needed here too.
        s_lower_global = torch.nan_to_num(s_lower_global, nan=0.0, posinf=0.0, neginf=0.0)
        s_upper_global = torch.nan_to_num(s_upper_global, nan=0.0, posinf=0.0, neginf=0.0)
        # Ensure lower <= upper after potential NaN handling
        s_lower_global = torch.minimum(s_lower_global, s_upper_global)

        # --- Strategy Step 1: Find Most Critical Unverified Class ---
        critical_k = self._find_most_critical_unverified_class(unverified, s_lower_global, s_upper_global)

        if critical_k is None:
            if verbose:
                print("VerificationDriven: No critical class found, using Fallback.")
            return self._find_next_distance_max_uncertainty(self.explanation, feature_lower_bound_distances, feature_upper_bound_distances, distances_with_proto)

        # --- Strategy Step 2: Find Most Uncertain Prototype for (c, k*) ---
        critical_p = self._find_most_uncertain_prototype(s_lower_global, s_upper_global, critical_k)

        if critical_p is None:
            if verbose:
                print(f"VerificationDriven: No uncertain prototype for c={self.selected_class}, k={critical_k}, using Fallback.")
            return self._find_next_distance_max_uncertainty()

        # --- Strategy Step 3: Find Most Uncertain Feature for p* ---
        best_hw = self._find_most_uncertain_feature(critical_p, feature_lower_bound_distances, feature_upper_bound_distances)

        if best_hw is None:
            if verbose:
                print(f"VerificationDriven: No uncertain feature for p={critical_p}, using Fallback.")
            return self._find_next_distance_max_uncertainty()

        # --- Strategy Step 4: Retrieve True Distance ---
        try:
            true_dist = self.distances_with_proto[self.iidx, critical_p, best_hw].item()
            if verbose:
                print(f"VerificationDriven selected: hw={best_hw}, p={critical_p}, k={critical_k}, dist={true_dist:.4f}")
            return best_hw, critical_p, true_dist
        except IndexError:
            print(f"ERROR: Indexing error retrieving true distance for p={critical_p}, hw={best_hw}")
            print(f"distances_with_proto shape: {self.distances_with_proto.shape}")
            # Fallback if indexing fails
            return self._find_next_distance_max_uncertainty()


    # === Helper Functions for Strategy 2 ===

    def _find_most_critical_unverified_class(self, unverified, s_lower_global, s_upper_global):
        """Finds unverified class k where upper_bound(score(k) - score(c)) is minimal non-negative."""
        min_score_diff_upper = float('inf')
        critical_k = None
        W = self.weights
        c = self.selected_class

        for k in unverified:
            if k == c: 
                continue # Should not happen if unverified list is correct

            Wk = W[k, :]
            Wc = W[c, :]

            # Determine which similarity bound to use for the upper bound of score(k) - score(c)
            # We want to maximize score(k) and minimize score(c) simultaneously
            selector_k_vs_c = Wk > Wc # Where Wk contributes more positively (or less negatively) than Wc
                                      # If Wk > Wc, use s_upper for k and s_lower for c to maximize diff
                                      # If Wk <= Wc, use s_lower for k and s_upper for c to maximize diff
            # print(f"Selector for k={k}: {selector_k_vs_c.shape}")
            # print(f"s_lower_global: {s_lower_global.shape}, s_upper_global: {s_upper_global.shape}")
            sim_for_k = torch.where(selector_k_vs_c, s_upper_global, s_lower_global)
            sim_for_c = torch.where(selector_k_vs_c, s_lower_global, s_upper_global)

            margin_upper_bound = torch.sum(Wk * sim_for_k - Wc * sim_for_c)
            margin_upper_bound_val = margin_upper_bound.item()

            # Check if k could potentially beat c (margin >= 0 within tolerance)
            # and if it's the closest call so far
            if margin_upper_bound_val >= -self.epsilon and margin_upper_bound_val < min_score_diff_upper:
                min_score_diff_upper = margin_upper_bound_val
                critical_k = k

        return critical_k


    def _find_most_uncertain_prototype(self, s_lower_global, s_upper_global, k):
        """Finds prototype p contributing most uncertainty to distinguishing c and k."""
        masked_p_uncertainty = -1.0
        critical_p = None
        W = self.weights
        c = self.selected_class

        # Calculate uncertainty contribution for each prototype
        sim_range = s_upper_global - s_lower_global
        weight_diff = torch.abs(W[c, :] - W[k, :])
        p_uncertainty = weight_diff * sim_range

        # Ensure we only consider valid, finite uncertainties
        valid_mask = torch.isfinite(p_uncertainty) & (sim_range >= 0) # Sim range should be non-negative

        if not torch.any(valid_mask):
            return None # No valid prototypes found

        # Find prototype with max uncertainty among valid ones
        masked_p_uncertainty = torch.where(
            valid_mask,
            p_uncertainty,
            torch.tensor(-1.0, device=self.device) # Use -1 to ignore invalid ones in max
        )

        max_val = torch.max(masked_p_uncertainty)

        # Check if any positive uncertainty was found
        if max_val < 0: # If max is -1 or slightly negative due to float issues
            return None

        # Find the index (prototype) of the maximum value
        # Note: argmax returns the first occurrence in case of ties
        critical_p = torch.argmax(masked_p_uncertainty).item()

        return critical_p


    def _find_most_uncertain_feature(self, p_star, feature_lower_bound_distances, feature_upper_bound_distances):
        """Finds feature hw with the largest finite distance uncertainty for prototype p_star."""
        num_hw = self.H * self.W
        # E is a list of tuples (hw, p, dist) where hw is the index of the feature vector
        E = self.explanation
        # 1. Create mask of pairs (hw, p_star) already in the explanation E
        indices_in_E_mask_p = torch.zeros(num_hw, dtype=torch.bool, device=self.device)
        if E:
            # Optimize: Pre-calculate a set or dict for faster lookups if H*W is large
            indices_in_E_for_p = {el[0] for el in E if el[1] == p_star}
            if indices_in_E_for_p: # Check if set is not empty
                hw_indices_p = torch.tensor(list(indices_in_E_for_p), device=self.device, dtype=torch.long)
                # Ensure indices are within bounds (should always be if H, W correct)
                if hw_indices_p.numel() > 0:
                    hw_indices_p = hw_indices_p[hw_indices_p < num_hw]
                    if hw_indices_p.numel() > 0:
                        indices_in_E_mask_p[hw_indices_p] = True

        # 2. Calculate uncertainty for the specific prototype p_star
        uncertainty_p = feature_upper_bound_distances[:, p_star] - \
                        feature_lower_bound_distances[:, p_star]

        # 3. Create mask for valid uncertainty values (Finite and not in E)
        valid_mask = torch.isfinite(uncertainty_p) & (~indices_in_E_mask_p) & (uncertainty_p >= 0)

        # 4. Check if any valid uncertain features remain for this prototype
        if not torch.any(valid_mask):
            return None

        # 5. Find hw with max uncertainty among valid ones
        masked_unc_p = torch.where(
            valid_mask,
            uncertainty_p,
            torch.tensor(-1.0, device=self.device)
        )

        max_val = torch.max(masked_unc_p)

        if max_val < 0: # No positive finite uncertainty found
            return None

        best_hw = torch.argmax(masked_unc_p).item()
        return best_hw


class TopKFormalExplanation(FormalExplanationBase):
    """
    A class to generate formal explanations for the prediction of a model using the top-k prototypes.
    Inherits from FormalExplanationBase.
    """

    def __init__(self, model, device="cuda:0", **kwargs):
        super().__init__(model, device=device, **kwargs)
        self.explanation = []  # List to store the explanation
        
    def explain_one(self, x, y, verbose=True):
        """
        Generate a formal explanation for a single input x with label y.
        Args:
            x: Input tensor.
            y: True label of the input.
            verbose: If True, print additional information.
        """
        verbose = True
        # self.forward(x, y, verbose=verbose, top_k=True, max_only=False, triangle_inequality=False, hypersphere_approximation=False)
        # exp = self.explanation.copy()  # Copy the explanation to return
        # self.explanation = []  # Clear the explanation for the next call
        # return exp
        
        ### PSEUDO CODE ###.
        # 0. Initialize the explanation E = {} and the unverified classes unverified = {0, ..., C} \ {c}
        # 1. Forward pass
        # 1._. While there are unverified classes:
        # 1.a. Get j = NextPrototype(E) # where j is the most activated prototype not in E
        # 1.c. Update the explanation $E$ with the new prototype j and the associated activation and distance
        # 1.d. Verify the explanation with the activation bounds i.e. check how many unverified classes remain
        # 1.e. If there are unverified classes, go back to step 1
        ### END PSEUDO CODE ###
        
        check_memory_usage(threshold_mb=5000)
        
        x = x.to(self.device)  # (N, C, H, W) # images
        y = y.to(self.device)  # (N) # labels
        self.batch_size = x.size(0)  # N
        
        # # Get the model's output
        with torch.no_grad():
            similarities = self.model.similarities(x)  # (N, P, H, W)
            logits = self.model(x)[0]  # (N, K)
            self.batch_size = x.shape[0]  # (N)
        
        prediction_conf, predicted_class = torch.max(logits.squeeze(), dim=0)  # (N, K) -> (N)
        self.c = int(predicted_class.item())  # (1) # predicted class
        
        
        # 1. Calculate dense activations per prototype
        if isinstance(self.model.classifier, ProtoPoolClassifier):
            dists_with_proto = self.model.distances(x).view(1, self.num_prototypes, -1)
            min_dists = torch.min(dists_with_proto, dim=2)[0].squeeze(0)
            avg_dists = torch.mean(dists_with_proto, dim=2).squeeze(0)
            
            sim_layer = self.model.classifier.similarity_layer
            s_min = sim_layer.distances_to_similarities(min_dists)
            s_avg = sim_layer.distances_to_similarities(avg_dists)
            
            A_dense = s_min - s_avg
        else:
            A_dense = torch.max(similarities[0].view(self.num_prototypes, -1), dim=1).values
        # 2. Re-apply the exact sparsity mask used by the classifier
        if hasattr(self.model.classifier, "_apply_sparsity"):
            with torch.no_grad():
                # _apply_sparsity expects a batch dimension: (N, P)
                A_sparse = self.model.classifier._apply_sparsity(A_dense.unsqueeze(0))
                A = A_sparse.squeeze(0)  # Unwrap back to (P)
                
                # # Count how many prototypes survived the sparsity mask
                # num_active = (A > 0).sum().item()
                # print(f"Number of actually active prototypes for this image: {num_active}")
        else:
            A = A_dense
        
        E: dict = {}
        
        # verification init
        unverified = list(range(self.num_classes))
        unverified.remove(self.c)
        
        if verbose:
            t = tqdm(
                total=len(unverified),
                desc="Explaining",
                unit="prototype",
                leave=False,
            )
            t.n = 0
            t.refresh()

        while unverified:
            # Get the next prototype
            j, act_j = self._next_prototype(E, A)
            
            # Add the prototype to the explanation
            E.update({j: act_j})
            # print(f"Added prototype {j} with activation {act_j}. Size of E: {len(E)}")
            if len(E) > len(A):
                print("Error: Explanation size exceeds the number of prototypes")
                exit(1)
            
            # update the bounds
            lower_bound, upper_bound = self._update_bounds(E)
            
            unverified, unverified_conf = self._verify_explanation(
                lower_bound, upper_bound, unverified
            )
            
            if verbose:
                t.n += 1
                postfix_str = f"pred cls: {self.c}, true cls: {y.item()}, conf: {prediction_conf:.2f}, next_p: {j:04d}, next_act: {act_j:.3f}, n_unverif: {len(unverified)}, cex conf: {torch.max(unverified_conf):.3f}"
                # postfix_str = f"most-activated proto: {torch}"
                t.set_postfix_str(postfix_str)
                t.refresh()
        
        if verbose:
            t.close()
        self.explanation = []  # empty explanation
        return E  # Return the explanation as a dictionary {prototype_index: activation_value}

    def _next_prototype(self, E, A) -> tuple[int, float]:
        """
        Get the next prototype to add to the explanation.
        Args:
            E: Current explanation.
            A: Activation values of the prototypes.
        Returns:
            j: Index of the next prototype.
            act_j: Activation value of the next prototype.
        """
        # Get the most activated prototype not in E
        existing_p = list(E.keys())
        activations = torch.tensor(A, device=self.device)  # (P)
        activations[existing_p] = float("-inf")  # Set the activations of existing prototypes to -inf
        next_proto_idx: int = int(torch.argmax(activations).item())  # (P)
        next_proto_act = activations[next_proto_idx].item()  # (1)
        return next_proto_idx, next_proto_act

    def _update_bounds(self, E) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update the lower and upper bounds of the explanation based on the current explanation E.
        Args:
            E: Current explanation.
        Returns:
            lower_bound: Lower bound of the explanation.
            upper_bound: Upper bound of the explanation.
        """
        lower_bound = torch.zeros(self.num_prototypes, device=self.device)  # (P)
        upper_bound = torch.zeros(self.num_prototypes, device=self.device)  # (P)
        
        lower_bound[list(E.keys())] = torch.tensor(list(E.values()), device=self.device)  # (P)
        upper_bound[list(E.keys())] = torch.tensor(list(E.values()), device=self.device)  # (P)
        
        # Set the upper bound for the prototypes not in E to the minimum activation value in the explanation
        min_activation = torch.min(torch.tensor(list(E.values()), device=self.device))  # (1)
        upper_bound[upper_bound == 0] = min_activation  # (P)
        
        return lower_bound, upper_bound  # (P), (P)

    def _verify_explanation(self, lower_bound, upper_bound, unverified_classes) -> tuple[list, torch.Tensor]:
        # verify conditions
        weights = self.weights  # (P, K) # last layer of the ProtoPNet architecture
        
        with torch.no_grad():
            selected_class = self.c  # (1) # predicted class
            predicted_class_weights = weights[selected_class]  # (P)
            batch_selector = weights > predicted_class_weights  # (P, K) # boolean matrix
            # similarities_to_check = upper_bound * batch_selector + lower_bound * (
            #     torch.logical_not(batch_selector)
            # )  # (P, K)
            similarities_to_check = torch.where(batch_selector, upper_bound, lower_bound)
            # decision_output = self.decision_head(similarities_to_check)  # (K, K)
            # ProtoPool's linear layer expects (C*S) inputs, not (P). We must use the compressed weights.
            decision_output = torch.matmul(similarities_to_check, self.weights.t())  # (K, K)
            
        
        # print(decision_output.shape)
        new_unverified = unverified_classes.copy()  # Copy the list of unverified classes
        # unverified_conf = torch.zeros(self.num_classes)
        unverified_conf = torch.ones(self.num_classes) * -1 * float("inf")  # Initialize with negative infinity for unverified classes
        
        for uidx in unverified_classes:
            # print(f"Best counterfactual for class {uidx}: {decision_output[uidx, uidx]:.3f} vs {decision_output[uidx, selected_class]:.3f}")
            if decision_output[uidx, selected_class] > decision_output[uidx, uidx]:
                new_unverified.remove(uidx)
            else:
                unverified_conf[uidx] = decision_output[uidx, uidx]
        unverified = new_unverified
        
        return unverified, torch.tensor(unverified_conf, device=self.device).max(dim=0)[0]  # Return the maximum confidence of the unverified classes
    

# class TopKFormalExplanation(FormalExplanationBase):
#     """
#     A class to generate formal explanations for the prediction of a model using the top-k prototypes.
#     Inherits from FormalExplanationBase.
#     """

#     def __init__(self, model, device="cuda:0", **kwargs):
#         super().__init__(model, device=device, **kwargs)
#         self.explanation = []  # List to store the explanation
        
#         self.estimated_centers = None
#         self.estimated_radii = None
    
#     def explain_one(self, x, y, verbose=False):
#         """
#         Generate a formal explanation for a single input x with label y.
#         Args:
#             x: Input tensor.
#             y: True label of the input.
#             verbose: If True, print additional information.
#         """
#         self.forward(x, y, verbose=verbose, top_k=True, max_only=False, triangle_inequality=False, hypersphere_approximation=False)

#         # --- Filter explanation based on Sparsity Mask ---
#         # If the classifier applied a mask, we must remove 'dead' prototypes that 
#         # the base explainer picked up from the dense distances.
#         if hasattr(self.model.classifier, "_apply_sparsity"):
#             with torch.no_grad():
#                 # Re-compute sparsity mask for this input
#                 _, distances = self.model(x)
#                 sims = self.model.classifier.similarity_layer.distances_to_similarities(distances)
#                 sparse_sims = self.model.classifier._apply_sparsity(sims)
                
#                 # Active if similarity > 0 (Handling BatchTopK / TopK logic)
#                 # Move to CPU to match explanation list format
#                 active_mask = (sparse_sims > 0).view(-1).cpu()

#             # Filter: Keep pair only if prototype_idx is active
#             # self.explanation structure is expected to be [(proto_idx, score), ...]
#             self.explanation = [
#                 pair for pair in self.explanation 
#                 if active_mask[pair[0]]
#             ]
#         # --- 

#         exp = self.explanation.copy()  # Copy the explanation to return
#         self.explanation = []  # Clear the explanation for the next call
#         return exp


class SpatialFormalExplanation(FormalExplanationBase):
    """
    A class to generate spatial formal explanations for the prediction of a model.
    Inherits from FormalExplanationBase.
    """

    def __init__(self, model, device="cuda:0", paradigm="triangle", **kwargs):
        super().__init__(model, device=device, **kwargs)
        
        self.verbose = True
        
        self.paradigm = paradigm  # Paradigm for the explanation, e.g., "triangle", "hypersphere"
        self.explanation = []  # List to store the explanation
        
        
        # # --- PROTOPOOL: Force Weight Compression Locally ---
        # if isinstance(self.model.classifier, ProtoPoolClassifier):
        #     # 1. Get mapping (C, S_per_class) -> Active Prototypes Indices
        #     slot_to_proto_map = self.model.classifier.class_mapping
        #     if isinstance(slot_to_proto_map, torch.Tensor):
        #         slot_to_proto_map = slot_to_proto_map.cpu().numpy()
            
        #     num_classes = self.model.classifier.num_classes
        #     num_slots_per_class = self.model.classifier.num_slots_per_class
        #     raw_weights = self.model.classifier.last_layer.weight.data # (C, 2000)

        #     # 2. Compress (C, 2000) -> (C, 202)
        #     # Use self.P (which is 202)
        #     effective_weights = torch.zeros((num_classes, self.P), device=self.device)
            
        #     current_slot_linear_idx = 0
        #     for c in range(num_classes):
        #         for s in range(num_slots_per_class):
        #             p_idx = slot_to_proto_map[c, s]
        #             effective_weights[:, p_idx] += raw_weights[:, current_slot_linear_idx]
        #             current_slot_linear_idx += 1
            
        #     self.weights = effective_weights
        #     print(f"SpatialFormalExplanation: Compressed weights to {self.weights.shape}")
        # # -----------------------------------------------------
        
        # Pre-compute and cache frequently used tensors
        self._hw_indices_cache = None
        self._mask_cache = None
        
        # --- PERSISTENT STATE (Hypersphere Paradigm) ---
        # Current geometric state
        self.h_centers = None     # Shape (HW, D)
        self.h_radii = None       # Shape (HW, 1)
        
        # Initial geometric state (cached for fast resets)
        self._h_centers_init = None
        self._h_radii_init = None
        
        # The "Index Map": Tracks which pairs are active O(1)
        self.active_mask = None   # Shape (HW, P), dtype=bool
        
        # Exact distances cache to avoid O(|E|) conversions in backward pass
        self.known_distances = None # Shape (HW, P), initialized to NaN
        
        # Maps hw_idx -> List of prototype indices in order of addition
        # Needed to ensure deterministic reconstruction of spheres
        self.add_order = {}
        
    def explain_one(self, x, y, verbose=False, batch_update: bool = True, fast_backward: bool = False) -> dict:
        """
        Generate a spatial formal explanation for a single input x with label y.
        Args:
            x: Input tensor.
            y: True label of the input.
            verbose: If True, print additional information.
        """
        if self.verbose is not None:
            verbose = self.verbose
        else:
            verbose = False
        ### PSEUDO CODE ###
        # -1. Initialize the explainer with the distances with prototypes and the feature vectors.
        # 0. initialize the explanation $E$ by giving, for each feature vector, the closest prototype and the associated similarity score / distance.
        # 1. Forward pass
        # 1._. While there are unverified classes:
        # 1.a. Get (l,j) = NextPair(E) # where l is the feature vector and j is the prototype
        # 1.b. Compute the distance between the feature vector l and the prototype j
        # 1.c. Update the explanation $E$ with the new pair (l,j) and the associated distance
        # 1.d. lb, ub = GenerateBounds(E, paradigm))
        # 1.e. Verify the explanation with the bounds i.e. check how many unverified classes remain
        # 1.f. If there are unverified classes, go back to step 1
        # 2. Backward pass
        # 2._. While the explanation is valid:
        # 2.a. Remove the last pair (l,j) from the explanation *that was not marked as verified* and update the explanation
        # 2.b. lb, ub = GenerateBounds(E, paradigm))
        # 2.c. Verify the explanation with the bounds
        # 2.c.1. If the explanation is still valid, go back to step 2 with the new explanation
        # 2.c.2. If the explanation is not valid, go back to step 2 with the same explanation but mark the last pair as verified
        # 2.c.3. If the explanation is not valid and there are no more pairs to remove, stop the process and return the explanation
        # 3. Return the explanation $E$ with the associated distances and the verified pairs
        ### END PSEUDO CODE ###
        
        # --- Initialization ---
        # Verify triangle inequality before proceeding (only if L2 distances)
        if self.model.classifier.similarity_layer.distances.__class__ == SquaredEuclideanDistance and self.epsilon is not None:
            if not self.verify_triangle_inequality(x, tolerance=self.epsilon):
                raise ValueError("Triangle inequality does not hold for input distances. Check distance computation.")
            
        check_memory_usage(threshold_mb=5000)
        
        x = x.to(self.device)  # (N, C, H, W) # images
        y_true = y.to(self.device)  # (N) # labels
        self.model.eval()
        self.batch_size = x.size(0)  # N
        
        with torch.no_grad():
            # 1. First, check if a custom distance function was provided by a subclass
            if hasattr(self, 'feature_distance_fn') and self.feature_distance_fn is not None:
                distances_with_proto = self.feature_distance_fn(x)  # (N, P, H, W)
                _, self.num_prototypes, self.H, self.W = distances_with_proto.shape
                
            # 2. Fallback to specific handling for Cosine models
            elif isinstance(self.model.classifier.similarity_layer, CosineSimilarity):
                sims_with_proto = self.model.similarities(x)  # (N, P, H, W)
                distances_with_proto = 1 - sims_with_proto  # Convert similarities to distances for uniformity
                _, self.num_prototypes, self.H, self.W = sims_with_proto.shape
                
            # 3. Fallback to default L2 models if the attribute wasn't explicitly overridden
            elif self.model.classifier.similarity_layer.distances.__class__ == SquaredEuclideanDistance:
                distances_with_proto = self.feature_distance_fn(x)  # (N, P, H, W)
                _, self.num_prototypes, self.H, self.W = distances_with_proto.shape
                
            else:
                raise ValueError(
                    f"Unsupported similarity layer: {self.model.classifier.similarity_layer.__class__.__name__}. "
                    "Please assign `self.feature_distance_fn` in your custom explainer."
                )
            
            logits = self.model(x)[0]  # (N, K)
            y_pred = torch.argmax(logits, dim=1)
            y = y_pred.clone()  # Use predicted class for explanation

        distances_with_proto = distances_with_proto.view(self.batch_size, self.num_prototypes, -1)  # (N, P, H, W) -> (N, P, H*W)
        self.distances_with_proto = distances_with_proto
        
        # --- CRITICAL PROTOPOOL FIX: S_avg must be computed on original d^2 distances ---
        if isinstance(self.model.classifier, ProtoPoolClassifier):
            # Fetch true native distances, bypassing the explainer's SQRT wrapper
            true_model_dists = self.model.distances(x).view(self.batch_size, self.num_prototypes, -1)
            self.true_avg_distances = torch.mean(true_model_dists, dim=-1)[0] # Shape (P)
        # -------------------------------------------------------------------------------- # (N, P)
        
        self._distances_transposed = self.distances_with_proto[0].T  # (H*W, P)
        with torch.no_grad():
            self.z = self.model.features(x)
            self._z_features_flat = self.z.flatten(start_dim=2).permute(2, 0, 1).squeeze(1).detach()
            self.similarity_values = self.model.similarities(x).detach()
        
        prediction_conf, predicted_class = torch.max(logits.squeeze(), dim=0)  # (N, K) -> (N)
        self.c: int = int(predicted_class.item())  # (1)
        with torch.no_grad():
            selected_class = self.c  # (1) # predicted class
            predicted_class_weights = self.weights[selected_class]  # (P)
            # Cache this mask
            self._cached_batch_selector = self.weights > predicted_class_weights  # (P, K) # boolean matrix
                
        # Pre-compute indices and masks for reuse
        self.total_hw = self.H * self.W
        if self._hw_indices_cache is None or len(self._hw_indices_cache) != self.total_hw:
            self._hw_indices_cache = torch.arange(self.total_hw, device=self.device)

        # 1. INITIALIZATION & STATE SETUP
        E: dict = {}  # Explanation dictionary to store pairs of (feature vector, prototype) and their distances
        E = self._initialize_explanation(x, y)  # Initialize the explanation with the closest prototype for each feature vector
        
        # Sanity checks
        assert self.H is not None and self.W is not None, "Height and width of the feature vectors must be initialized."
        assert self.H > 0 and self.W > 0, "Height and width of the feature vectors must be greater than 0."

        # Forward pass
        unverified_classes = list(range(self.num_classes))  # List of unverified classes
        unverified_classes.remove(self.c)  # Remove the true class from the list of unverified classes
        
        # Mask management for batch updates
        available_pairs_mask = torch.ones((self.total_hw, self.P), dtype=torch.bool, device=self.device)
        
        if E:  # Mark pairs from the initialization step as unavailable
            initial_hw, initial_p = zip(*E.keys())
            available_pairs_mask[list(initial_hw), list(initial_p)] = False

        # progress tracking
        if verbose:
            # Set total to the maximum possible number of pairs
            t = tqdm(total=(self.H * self.W * self.num_prototypes), desc="Forward pass", unit="pair")
            t.n = len(E)  # Initialize with pairs from initialization
            t.refresh()

        # 2. FORWARD PASS
        while unverified_classes:
            pairs_to_add = {}  # dict of {(hw, p): dist}
            
            if batch_update:
                # --- BATCH UPDATE LOGIC ---
                hw_indices, p_indices, dists = self._next_batch_of_pairs(available_pairs_mask)
                
                if hw_indices.numel() == 0:
                    break

                # Update the persistent mask for the next iteration
                available_pairs_mask[hw_indices, p_indices] = False
                
                # Update E and queue for state update
                for hw, p, d in zip(hw_indices, p_indices, dists):
                    pair = (hw.item(), p.item())
                    E[pair] = d.item()
                    pairs_to_add[pair] = d.item()
                
                if verbose:
                    t.update(hw_indices.numel())
                    
            else:
                # --- SINGLE PAIR LOGIC ---
                l, j = self._next_pair_round_robin_vectorized(E)
                distance = self._compute_distance(l, j)
                E = self._update_explanation(E, l, j, distance)
                
                # Update masks/progress
                available_pairs_mask[l, j] = False
                pairs_to_add[(l, j)] = distance
                
                if verbose:
                    t.update(1)

            # --- OPTIMIZATION: Incremental State Update ---
            if self.paradigm == "hypersphere" and pairs_to_add:
                self._update_hypersphere_state_incremental(pairs_to_add)
            
            # --- COMMON LOGIC FOR BOTH BRANCHES ---
            # Generate bounds based on the explanation and the chosen paradigm
            # Note: pairs_to_add is a dict {(hw, p): dist}, convert to list of tuples
            new_pairs_list = [(k[0], k[1], v) for k, v in pairs_to_add.items()]
            (lb, ub) = self._generate_bounds(E, paradigm=self.paradigm, new_pairs_only=new_pairs_list)
            
            # Verify the explanation with the bounds
            unverified_classes, unverified_conf = self._verify_explanation(lb, ub, unverified_classes)
            # If there are unverified classes, continue to the next iteration
            if verbose:
                postfix_str = f"pred_cls: {self.c}, n_unverif: {len(unverified_classes)}"
                if unverified_classes:
                    postfix_str += f", max_cex_conf: {torch.max(unverified_conf):.3f}"
                t.set_postfix_str(postfix_str)
                t.refresh()
            
            if len(E) >= (self.total_hw * self.num_prototypes):
                break
        if verbose:
            t.close()
            print(f"Forward pass completed. Length of explanation E: {len(E)}")

        # 3. BACKWARD PASS
        if fast_backward:
            # --- FAST, BINARY-SEARCH BACKWARD PASS (NEW) ---
            if verbose:
                print("Starting fast backward pass (binary search)...")
                start_time = time.time()

            E_minimal = self._find_minimal_explanation_binary_search(E)
            
            if verbose:
                end_time = time.time()
                print(f"Fast backward pass completed in {end_time - start_time:.2f}s. "
                    f"Reduced explanation from {len(E)} to {len(E_minimal)} pairs.")
            return E_minimal

        else:
            marked_as_verified: set[tuple[int, int]] = set()  # Set to keep track of pairs that were marked as verified
            is_valid = True  # Flag to check if the explanation is still valid
            
            if verbose:
                t = tqdm(total=len(E), desc="Backward pass", unit="pair")

            while (len(E) > 1):
                # We check if there are more pairs to remove
                if len(E) == len(marked_as_verified):
                    break

                l, j = self._remove_last_pair(E, marked_as_verified)
                removed_distance = E[(l, j)]
                # E = self._update_explanation(E, l, j, None)  # Remove the pair from the explanation
                del E[(l, j)] # Temporarily remove from E dict
                
                # --- OPTIMIZATION: SNAPSHOT STATE ---
                if self.paradigm == "hypersphere":
                    # 1. Snapshot the current state of feature 'l'
                    saved_center = self.h_centers[l].clone()
                    saved_radius = self.h_radii[l].clone()
                    
                    # 2. Update Mask
                    self.active_mask[l, j] = False
                    self.known_distances[l, j] = float('nan')
                    
                    # 3. Rebuild (Unavoidable for removal, but we save on the Revert)
                    self._rebuild_hypersphere_state_single(l)
                
                # Generate bounds based on the updated explanation
                lb, ub = self._generate_bounds(E, paradigm=self.paradigm)
                unverified_classes, unverified_conf = self._verify_explanation(lb, ub, [k for k in range(self.num_classes) if k != self.c])
                is_valid = len(unverified_classes) == 0  # Check if the explanation is still valid
                
                if is_valid:
                    # --- COMMIT REMOVAL ---
                    # The trial was successful. Now we permanently update the order list.
                    if self.paradigm == "hypersphere":
                        # We only remove from list here. Mask and Distances already updated in Trial.
                        if j in self.add_order[l]:
                            self.add_order[l].remove(j)
                    
                    if verbose:
                        t.update(1)
                    continue
                
                # If the explanation is not valid, we mark the last pair as verified
                # --- OPTIMIZATION: FAST REVERT ---
                E[(l, j)] = removed_distance
                marked_as_verified.add((l, j))
                
                if self.paradigm == "hypersphere":
                    self.active_mask[l, j] = True
                    self.known_distances[l, j] = removed_distance
                    
                    # RESTORE SNAPSHOT instead of rebuilding
                    self.h_centers[l] = saved_center
                    self.h_radii[l] = saved_radius
                    
                    # MARK DIRTY (Important: Manual update requires manual dirty flag)
                    if not hasattr(self, '_dirty_hw_indices'):
                        self._dirty_hw_indices = set()
                    self._dirty_hw_indices.add(l)
                
                if verbose:
                    postfix_str = f"pred cls: {self.c}, true cls: {y_true.item()}, conf: {prediction_conf:.2f}, mark_l: {l:02d}, mark_p: {j:04d}, rem_dis: {removed_distance:.2f}, n_unverif: {len(unverified_classes)}"
                    if len(unverified_classes) > 0:
                        postfix_str += f", max_cex_conf: {torch.max(unverified_conf):.3f}"
                    t.set_postfix_str(postfix_str)
                    t.n += 1
                    t.refresh()
                    
            # Return the explanation E with the associated distances and the verified pairs
            if verbose:
                t.close()
                print(f"\nBackward pass completed. Length of explanation E: {len(E)}")
                # print(f"Explanation for input x with label {y.item()}:")
        # print(f"Final explanation E: {E}")
        return E
    
    def _initialize_explanation(self, x, y):
        """
        Initialize the explanation with the closest prototype for each feature vector.
        """
        self.c = y.item()
        
        # 1. Compute closest prototypes (Initialization)
        min_dists, min_proto_indices = torch.min(self._distances_transposed, dim=1) # (HW,)
        
        # 2. Build Explanation Dictionary
        E_init = {
            (hw.item(), p.item()): d.item() 
            for hw, p, d in zip(self._hw_indices_cache, min_proto_indices, min_dists)
        }
        
        # 3. Initialize Persistent State (Hypersphere)
        if self.paradigm == "hypersphere":
            self.h_centers = self.prototypes.squeeze()[(min_proto_indices,)] # (HW, D)
            self.h_radii = min_dists.unsqueeze(1) # (HW, 1)
            
            # Initialize Mask
            self.active_mask = torch.zeros((self.total_hw, self.num_prototypes), dtype=torch.bool, device=self.device)
            self.active_mask[self._hw_indices_cache, min_proto_indices] = True
            
            # Initialize Known Distances Cache (NaNs everywhere)
            self.known_distances = torch.full((self.total_hw, self.num_prototypes), float('nan'), device=self.device)
            # Fill knowns
            self.known_distances[self._hw_indices_cache, min_proto_indices] = min_dists
            
            # Initialize Add Order
            self.add_order = {}
            for hw, p in zip(self._hw_indices_cache.tolist(), min_proto_indices.tolist()):
                self.add_order[hw] = [p]
            
            # Aliases
            self.estimated_centers = self.h_centers
            self.estimated_radii = self.h_radii
        
        return E_init

    def _next_pair(self, E):
        """
        Get the next pair (feature vector, prototype) to explain.
        Args:
            E: Explanation dictionary with pairs of (feature vector, prototype) and their distances.
        Returns:
            tuple: (feature vector index, prototype index)
        """
        
        assert self.H is not None and self.W is not None, "Height and width of the feature vectors must be initialized."
        assert self.H > 0 and self.W > 0, "Height and width of the feature vectors must be greater than 0."
        
        # Create mask more efficiently using advanced indexing
        mask = torch.ones((self.total_hw, self.num_prototypes), dtype=torch.bool, device=self.device)
        if E:
            existing_hw, existing_p = zip(*E.keys())
            existing_hw_tensor = torch.tensor(existing_hw, device=self.device)
            existing_p_tensor = torch.tensor(existing_p, device=self.device)
            mask[existing_hw_tensor, existing_p_tensor] = False
        
        # Find the pair with the minimum distance that is not already in the explanation
        best_pair = None
        if self.paradigm == "triangle":
            # Vectorized distance computation
            distances_masked = self._distances_transposed.clone()  # (H*W, P)
            distances_masked[~mask] = float('inf')  # Mask out existing pairs
            min_idx = torch.argmin(distances_masked)
            best_pair = (min_idx // self.num_prototypes, min_idx % self.num_prototypes)
            
        elif self.paradigm == "hypersphere":
            # The next pair is the one with the maximum absolute cosine value that is not already in the explanation
            # P-C Shape (HW, P, D)
            centers_to_protos = self.prototypes.squeeze((2,3)).squeeze(0) - self.estimated_centers.unsqueeze(1) 
            # print(centers_to_protos.shape)
            
            # Z-C Shape # (H*W, 1, D)
            centers_to_feats = self.z.flatten(start_dim=2).swapaxes(1,2).swapaxes(0,1) - self.estimated_centers.unsqueeze(1)
            # print(centers_to_feats.shape)
            
            # Out: (H*W, P)
            cos = nn.CosineSimilarity(dim=2, eps=1e-8)
            cos_sim = cos(centers_to_protos, centers_to_feats)
            # cos_sim = pairwise_cosine_similarity(estimated_centers)
            # scalar_prods = scalar_prods / (
            #     estimated_radii.view(total_num_feature_vectors, 1) * self.model.classifier.prototypes.norm(dim=1)
            # ) # (H*W, P) / (H*W, 1) * (P,) -> (H*W, P)
            # normalized_scalar_prods = scalar_prods / (torch.norm(estimated_centers, dim=1).view(-1,1) @ torch.norm(self.model.classifier.prototypes.squeeze(), dim=1).view(1,-1))
            # scalar_prods = torch.abs(normalized_scalar_prods)  # (H*W, P)
            scalar_prods = torch.abs(cos_sim)  # (H*W, P)
            # get the hw and p with the smallest scalar product i.e. largest angle + largest distance
            # print(f"scalar_prods shape: {scalar_prods.shape}")
            assert scalar_prods.shape == (self.H * self.W, self.P), f"Scalar products shape: {scalar_prods.shape}"
            
            # --- OPTIMIZATION ---
            # Modify the score to factor in the uncertainty (radius) of each feature vector.
            # This balances refining uncertain regions (large radius) with picking
            # geometrically optimal pairs (high cosine similarity).
            scores = self.estimated_radii * scalar_prods  # Shape: (H*W, 1) * (H*W, P) -> (H*W, P)
            # ---

            # Vectorized maximum finding using the new, balanced scores
            scores_masked = scores.clone()
            scores_masked[~mask] = 0.0  # Mask out existing pairs by setting their score to 0
            max_idx = torch.argmax(scores_masked)
            best_pair = (max_idx // self.num_prototypes, max_idx % self.num_prototypes)

        else:
            raise ValueError(f"Unknown paradigm: {self.paradigm}. Supported paradigms are 'triangle' and 'hypersphere'.")
        if best_pair is None:
            raise ValueError("No valid pair found for the next explanation step. Check the explanation dictionary and the distances.")
        return best_pair

    def _compute_distance(self, l, j):
        """
        Compute the distance between the feature vector l and the prototype j.
        Args:
            l: Feature vector index.
            j: Prototype index.
        Returns:
            float: Distance between the feature vector and the prototype.
        """
        # l is the feature vector index, j is the prototype index
        distance = self._distances_transposed[l, j].item()
        if distance < 0:
            raise ValueError(f"Computed distance is negative: {distance}. Check the distance computation and the input data.")
        return distance
    
    def _update_explanation(self, E, l, j, distance):
        """
        Update the explanation E with the new pair (l,j) and the associated distance.
        Args:
            E: Explanation dictionary with pairs of (feature vector, prototype) and their distances.
            l: Feature vector index.
            j: Prototype index.
            distance: Distance between the feature vector and the prototype.
        Returns:
            dict: Updated explanation dictionary.
        """
        if distance is None:
            # If distance is None, we are removing the pair from the explanation
            if (l, j) in E:
                del E[(l, j)]
        else:
            # Otherwise, we add or update the pair in the explanation
            E[(l, j)] = distance
        return E
    
    def _remove_last_pair(self, E, marked_as_verified):
        """
        Remove the last pair (l,j) from the explanation that was not marked as verified.
        Args:
            E: Explanation dictionary with pairs of (feature vector, prototype) and their distances.
            marked_as_verified: Set of pairs that were marked as verified.
        Returns:
            tuple: (feature vector index, prototype index) of the removed pair.
        """
        if not E:
            raise ValueError("Explanation is empty. Cannot remove last pair.")
        
        # Get the last pair in the explanation
        E_reversed = reversed(E)
        last_pair = next(E_reversed)  # Get the last pair in the explanation
        
        # If the last pair is already marked as verified, we need to find the next unverified pair
        while last_pair in marked_as_verified:
            # print(f"Skipping verified pair {last_pair}.")
            last_pair = next(E_reversed)  # Skip to the next iteration to find an unverified pair
        # Return the last unverified pair
        l, j = last_pair
        return l, j  # Return the feature vector index and prototype index of the removed pair
    
    def _update_hypersphere_state_incremental(self, new_pairs: dict):
        """
        Updates hyperspheres, mask, known_distances, and order incrementally.
        """
        for (hw, p), dist in new_pairs.items():
            self.active_mask[hw, p] = True
            self.known_distances[hw, p] = dist
            
            # Update Order safely
            if hw not in self.add_order:
                self.add_order[hw] = []
            
            # Only append if not already in list (idempotency fix)
            # This handles restoration logic or duplicate calls safely
            if p not in self.add_order[hw]:
                self.add_order[hw].append(p)
            
            # Perform Intersection
            self._intersect_single_feature(hw, p, dist)

    def _rebuild_hypersphere_state_single(self, hw_idx: int):
        """
        Rebuilds the hypersphere for 'hw_idx' from scratch using self.add_order
        filtering by self.active_mask.
        
        This logic is used for BOTH Trial (with masked pair) and Revert (with unmasked pair).
        """
        current_order = self.add_order[hw_idx]
        if not current_order: return

        # 1. Identify active prototypes in order
        # We assume self.active_mask is already updated for the current state (Trial or Revert)
        active_protos_in_order = [p for p in current_order if self.active_mask[hw_idx, p]]
        
        if not active_protos_in_order:
            # Fallback for empty state (should not happen in valid flow)
            self.h_radii[hw_idx] = float('inf')
            return

        # 2. Initialization (First Active Prototype)
        p_first = active_protos_in_order[0]
        # We can fetch distance from known_distances directly
        d_first = self.known_distances[hw_idx, p_first].item()
        
        self.h_centers[hw_idx] = self.prototypes[p_first].squeeze()
        self.h_radii[hw_idx] = d_first
        
        # 3. Sequential Intersection
        for p in active_protos_in_order[1:]:
            d = self.known_distances[hw_idx, p].item()
            self._intersect_single_feature(hw_idx, p, d)

    def _intersect_single_feature(self, hw_idx, p_idx, new_radius):
        c1 = self.prototypes[p_idx].squeeze()
        c2 = self.h_centers[hw_idx]
        r1 = new_radius
        r2 = self.h_radii[hw_idx].item()
        
        d_squared = torch.norm(c1 - c2).pow(2)
        d = torch.sqrt(d_squared)
        
        # --- REVERT TO HERON'S FORMULA (Theorem 1 / Def 4) ---
        
        # 1. Calculate semi-perimeter 'p'
        p = 0.5 * (d + r1 + r2)
        
        # 2. Calculate the Heron argument: p(p-d)(p-r1)(p-r2)
        # We use relu() to protect against tiny negative values due to floating point errors
        heron_arg = p * (p - d) * (p - r1) * (p - r2)
        
        # 3. Calculate r3: (2/d) * sqrt(Heron argument)
        r3 = (2.0 / (d + self.epsilon)) * torch.sqrt(torch.relu(heron_arg))
        
        # -----------------------------------------------------

        # We still compute 'h' (distance from c1 to the intersection plane) 
        # to update the center C3. The projection formula is robust for this.
        h = (r1**2 - r2**2 + d_squared) / (2 * d + self.epsilon)
        
        unit_c1_c2 = (c2 - c1) / (d + self.epsilon)
        c3 = c1 + h * unit_c1_c2
        
        self.h_centers[hw_idx] = c3
        self.h_radii[hw_idx] = r3
        
        # MARK DIRTY
        if not hasattr(self, '_dirty_hw_indices'):
            self._dirty_hw_indices = set()
        self._dirty_hw_indices.add(hw_idx)

    def _generate_bounds(self, E, paradigm="triangle", new_pairs_only=None) -> tuple[torch.Tensor, torch.Tensor]:
        if not hasattr(self, '_proto_pos_cache'):
            self._proto_pos_cache = self.prototypes.squeeze().unsqueeze(0)

        if not hasattr(self, '_cached_lb_features'):
            self._cached_lb_features = torch.zeros((self.total_hw, self.P), device=self.device)
            self._cached_ub_features = torch.full((self.total_hw, self.P), float('inf'), device=self.device)
            self._is_exact = torch.zeros((self.total_hw, self.P), dtype=torch.bool, device=self.device)

        if not E and not new_pairs_only:
            return self._cached_lb_features.min(dim=0).values, self._cached_ub_features.min(dim=0).values

        # ====================================================================
        # 1. ULTRA-FAST DICTIONARY PARSING (Numpy C-Backend)
        # ====================================================================
        parsed_E = False
        all_hw_t, all_p_t, all_d_t = None, None, None

        if new_pairs_only:
            hw_t = torch.tensor([x[0] for x in new_pairs_only], device=self.device, dtype=torch.long)
            p_t = torch.tensor([x[1] for x in new_pairs_only], device=self.device, dtype=torch.long)
            d_t = torch.tensor([x[2] for x in new_pairs_only], device=self.device, dtype=torch.float32)
        elif E:
            keys = np.array(list(E.keys()), dtype=np.int64)
            vals = np.array(list(E.values()), dtype=np.float32)
            hw_t = all_hw_t = torch.from_numpy(keys[:, 0]).to(self.device)
            p_t = all_p_t = torch.from_numpy(keys[:, 1]).to(self.device)
            d_t = all_d_t = torch.from_numpy(vals).to(self.device)
            parsed_E = True
            
            self._cached_lb_features.fill_(0.0)
            self._cached_ub_features.fill_(float('inf'))
            self._is_exact.fill_(False)

        self._is_exact[hw_t, p_t] = True
        
        # ====================================================================
        # TRIANGLE PARADIGM: Vectorized Spatial Loops (No scatter_reduce)
        # ====================================================================
        if paradigm == "triangle":
            C_current = torch.full((self.total_hw, self.P), float('inf'), device=self.device)
            C_current[hw_t, p_t] = d_t
            
            for hw in torch.unique(hw_t):
                valid_mask = ~torch.isinf(C_current[hw])
                obs_ps = torch.where(valid_mask)[0]
                if len(obs_ps) == 0: continue
                
                C_hw = C_current[hw, obs_ps].unsqueeze(1) # (K, 1)
                P_dists = self.prototype_distances[obs_ps, :] # (K, P)
                
                lb_batch = torch.abs(P_dists - C_hw)
                ub_batch = P_dists + C_hw
                
                best_ub = ub_batch.min(dim=0).values
                best_lb = lb_batch.max(dim=0).values
                
                exact_hw_mask = self._is_exact[hw]
                
                if new_pairs_only:
                    new_ub = torch.minimum(self._cached_ub_features[hw], best_ub)
                    new_lb = torch.maximum(self._cached_lb_features[hw], best_lb)
                    self._cached_ub_features[hw] = torch.where(exact_hw_mask, self._cached_ub_features[hw], new_ub)
                    self._cached_lb_features[hw] = torch.where(exact_hw_mask, self._cached_lb_features[hw], new_lb)
                else:
                    self._cached_ub_features[hw] = torch.where(exact_hw_mask, C_current[hw], best_ub)
                    self._cached_lb_features[hw] = torch.where(exact_hw_mask, C_current[hw], best_lb)

        # ====================================================================
        # HYPERSPHERE PARADIGM: Exact Masking & Cache Clears
        # ====================================================================
        elif paradigm == "hypersphere":
            if parsed_E and hasattr(self, '_dirty_hw_indices'):
                self._dirty_hw_indices.clear()
            
            if not hasattr(self, '_cache_h_dists') or (parsed_E and len(self._dirty_hw_indices) == 0):
                self._cache_h_dists = torch.zeros((self.total_hw, self.P), device=self.device)
                self._cache_h_dists = torch.norm(self.h_centers.unsqueeze(1) - self._proto_pos_cache, dim=2)
            elif hasattr(self, '_dirty_hw_indices') and self._dirty_hw_indices:
                dirty_indices = torch.tensor(list(self._dirty_hw_indices), device=self.device, dtype=torch.long)
                new_dists = torch.norm(self.h_centers[dirty_indices].unsqueeze(1) - self._proto_pos_cache, dim=2)
                self._cache_h_dists[dirty_indices] = new_dists
                self._dirty_hw_indices.clear()

            new_lb = self._cache_h_dists - self.h_radii
            new_ub = self._cache_h_dists + self.h_radii
            
            # Write bounds only to non-exact pairs
            self._cached_lb_features = torch.where(self._is_exact, self._cached_lb_features, torch.maximum(self._cached_lb_features, new_lb))
            self._cached_ub_features = torch.where(self._is_exact, self._cached_ub_features, torch.minimum(self._cached_ub_features, new_ub))

        # ====================================================================
        # GLOBAL EXACT OVERWRITE (Fixes the Float Corruption)
        # ====================================================================
        if parsed_E:
            self._cached_lb_features[all_hw_t, all_p_t] = all_d_t
            self._cached_ub_features[all_hw_t, all_p_t] = all_d_t
        elif new_pairs_only:
            self._cached_lb_features[hw_t, p_t] = d_t
            self._cached_ub_features[hw_t, p_t] = d_t

        lower_bound_distances = self._cached_lb_features.min(dim=0).values
        upper_bound_distances = self._cached_ub_features.min(dim=0).values

        # --- CRITICAL PROTOPOOL CONVERSION ---
        if isinstance(self.model.classifier, ProtoPoolClassifier):
            sim_layer = self.model.classifier.similarity_layer
            
            # 1. Exact average distance (using the un-monkey-patched d^2)
            exact_avg_sims = sim_layer.distances_to_similarities(self.true_avg_distances)
            
            # 2. Convert spatial bounds (d) back to Network-native distances (d^2)
            if hasattr(self, 'use_wrapper') and getattr(self, 'use_wrapper', False):
                lb_net = torch.pow(lower_bound_distances, 2)
                ub_net = torch.pow(upper_bound_distances, 2)
            else:
                lb_net = lower_bound_distances
                ub_net = upper_bound_distances
            
            # 3. Bound S(d_min) using squared bounds. (Note: smaller distance = LARGER similarity)
            s_min_lower = sim_layer.distances_to_similarities(ub_net)
            s_min_upper = sim_layer.distances_to_similarities(lb_net)
            
            # 4. Final Similarity Bounds = S_min - S_avg
            final_sim_lower = s_min_lower - exact_avg_sims
            final_sim_upper = s_min_upper - exact_avg_sims
            
            return (final_sim_lower, final_sim_upper) # Return Similarities!
        # -------------------------------------

        return lower_bound_distances, upper_bound_distances
    

    def _verify_explanation(self, lower_bound, upper_bound, unverified_classes):
        # verify conditions
        weights = self.weights  # (P, K) # last layer of the ProtoPNet architecture
        
        # --- PROTOPOOL CHECK ---
        if isinstance(self.model.classifier, ProtoPoolClassifier):
            # Input bounds are already similarities (handled safely in _generate_bounds)
            lower_bound_sim = lower_bound
            upper_bound_sim = upper_bound
        else:
            # Input bounds are distances. If they are SQRT wrapped, square them back to d^2!
            if hasattr(self, 'use_wrapper') and getattr(self, 'use_wrapper', False):
                lb_net = torch.pow(lower_bound, 2)
                ub_net = torch.pow(upper_bound, 2)
            else:
                lb_net = lower_bound
                ub_net = upper_bound
                
            lower_bound_sim = self.model.classifier.similarity_layer.distances_to_similarities(ub_net)
            upper_bound_sim = self.model.classifier.similarity_layer.distances_to_similarities(lb_net)
        # -----------------------
        
        
        # treat nans as zero
        lower_bound_sim = torch.nan_to_num(lower_bound_sim, 0.0)
        # lower_bound_sim *= torch.logical_not(lower_bound_sim.isnan) + 0*torch.isnan(lower_bound_sim)
        # lower_bound, upper_bound # (N, P)
        with torch.no_grad():
            batch_selector = self._cached_batch_selector # use cached version
            # similarities_to_check = upper_bound_sim * batch_selector + lower_bound_sim * (
            #     torch.logical_not(batch_selector)
            # )  # (P, K)
            # # res(i) = upper_bound_sim(i) if w_{i,k} > w_{i,c} else lower_bound_sim(i)
            # # similarities_to_check = similarities_to_check[:, unverified] # only classes that were not verified beforehand # actually NO
            # # check = similarities_to_check.swapaxes(0,1) # (K, P)
            # # decision_output = self.decision_head(similarities_to_check)  # (K, K)
            # # Perform manual linear pass using the corrected (200, 202) effective weights
            # # similarities_to_check is (K, P) and self.weights is (K, P)
            # # We want output (K, K), so we multiply (K, P) @ (P, K)
            # decision_output = torch.matmul(similarities_to_check, self.weights.t())  # (K, K)
            
            similarities_to_check = torch.where(batch_selector, upper_bound_sim, lower_bound_sim)
            
            decision_output = torch.matmul(similarities_to_check, self.weights.t())  # (K, K)
        
        # print(decision_output.shape)
        new_unverified = unverified_classes.copy()  # Copy the list of unverified classes
        # unverified_conf = torch.zeros(self.num_classes)
        unverified_conf = torch.ones(self.num_classes) * -1 * float("inf")  # Initialize with negative infinity for unverified classes
        
        for uidx in unverified_classes:
            # print(f"Best counterfactual for class {uidx}: {decision_output[uidx, uidx]:.3f} vs {decision_output[uidx, selected_class]:.3f}")
            if decision_output[uidx, self.c] > decision_output[uidx, uidx]:
                new_unverified.remove(uidx)
            else:
                unverified_conf[uidx] = decision_output[uidx, uidx]
        unverified = new_unverified
        
        return unverified, torch.tensor(unverified_conf, device=self.device).max(dim=0)[0]  # Return the maximum confidence of the unverified classes
    
    ###############################
    # Next pair selection methods #
    ###############################

    def _next_pair_round_robin(self, E):
        """
        Simplest approach: Round-robin through feature vectors, but order them by potential impact.
        """
        if self.paradigm == "hypersphere":
            # Count selections per feature vector
            feature_counts = torch.zeros(self.total_hw, device=self.device)
            for (hw, p) in E.keys():
                feature_counts[hw] += 1
            
            # Find feature vectors with minimum selections
            min_selections = torch.min(feature_counts).item()
            candidate_features = torch.where(feature_counts == min_selections)[0]
            
            # Among candidates, pick the one with highest potential cosine similarity
            best_hw = None
            best_p = None
            best_score = -1
            
            for hw in candidate_features:
                hw_val = hw.item()
                for p in range(self.num_prototypes):
                    if (hw_val, p) not in E:  # Not already selected
                        # Compute cosine similarity for this pair
                        center = self.estimated_centers[hw_val]
                        proto_pos = self.prototypes[p].squeeze()
                        feat_pos = self.z.flatten(start_dim=2).swapaxes(1,2).swapaxes(0,1)[hw_val, 0]
                        
                        center_to_proto = proto_pos - center
                        center_to_feat = feat_pos - center
                        
                        cos_sim = torch.abs(torch.nn.functional.cosine_similarity(
                            center_to_proto.unsqueeze(0), 
                            center_to_feat.unsqueeze(0), 
                            dim=1
                        )).item()
                        
                        if cos_sim > best_score:
                            best_score = cos_sim
                            best_hw = hw_val
                            best_p = p
            
            return (best_hw, best_p) if best_hw is not None else (0, 0)
        
    def _next_pair_round_robin_vectorized(self, E):
        """
        A vectorized and optimized version of the round-robin selection strategy.
        """
        if self.paradigm == "hypersphere":
            # 1. VECTORIZED COUNTING
            # If the explanation is empty, all features are candidates with 0 selections.
            if not E:
                feature_counts = torch.zeros(self.total_hw, dtype=torch.long, device=self.device)
            else:
                # Get all feature vector indices from the explanation keys.
                hw_indices = torch.tensor([hw for hw, p in E.keys()], dtype=torch.long, device=self.device)
                # Count occurrences of each feature vector index in a single operation.
                feature_counts = torch.bincount(hw_indices, minlength=self.total_hw)

            # Find the set of feature vectors that have been selected the minimum number of times.
            min_selections = torch.min(feature_counts)
            candidate_features = torch.where(feature_counts == min_selections)[0]
            
            if candidate_features.numel() == 0:
                # Edge case: if no candidates are found, fallback to a default.
                return 0, 0

            # 2. BATCH COMPUTATION OF SCORES
            # Pre-fetch all data needed for the candidate features.
            candidate_centers = self.estimated_centers[candidate_features]      # Shape: (num_candidates, D)
            all_protos = self.prototypes.squeeze()                              # Shape: (P, D)
            
            # Pre-calculate feature positions ONCE.
            feat_pos_all = self._z_features_flat # Shape: (H*W, D)
            candidate_feat_pos = feat_pos_all[candidate_features]               # Shape: (num_candidates, D)

            # Compute all vectors needed for cosine similarity using broadcasting.
            # `centers_to_protos` shape: (num_candidates, P, D)
            centers_to_protos = all_protos.unsqueeze(0) - candidate_centers.unsqueeze(1)
            # `centers_to_feats` shape: (num_candidates, 1, D)
            centers_to_feats = candidate_feat_pos.unsqueeze(1) - candidate_centers.unsqueeze(1)
            
            # Compute all cosine similarities at once.
            # Resulting shape: (num_candidates, P)
            scores = torch.abs(torch.nn.functional.cosine_similarity(centers_to_protos, centers_to_feats, dim=2))

            # 3. VECTORIZED MASKING
            # Create a boolean mask to filter out pairs already present in the explanation E.
            mask = torch.ones_like(scores, dtype=torch.bool)
            if E:
                # Build a sparse representation of E for efficient lookup
                e_keys_tensor = torch.tensor(list(E.keys()), dtype=torch.long, device=self.device)
                # For each candidate, find which prototypes are already in E
                for i, hw in enumerate(candidate_features):
                    # Find all pairs in E that match the current candidate feature vector
                    p_indices_in_e = e_keys_tensor[e_keys_tensor[:, 0] == hw, 1]
                    if p_indices_in_e.numel() > 0:
                        mask[i, p_indices_in_e] = False

            # Apply the mask. Invalid pairs get a score of -1 so they won't be picked.
            scores[~mask] = -1.0

            # 4. VECTORIZED SEARCH
            # Find the index of the highest score in the flattened score tensor.
            if scores.numel() == 0 or torch.all(scores == -1.0):
                # Fallback if all possible pairs for candidates are already selected
                return 0, 0 
                
            flat_idx = torch.argmax(scores.flatten())
            
            # Convert the flat index back to 2D indices: (candidate_idx, prototype_idx).
            candidate_idx = flat_idx // self.num_prototypes
            best_p = (flat_idx % self.num_prototypes).item()
            
            # Retrieve the original feature vector index from the candidates tensor.
            best_hw = candidate_features[candidate_idx].item()
            
            return best_hw, best_p
        
        elif self.paradigm == "triangle":
            if not E:
                feature_counts = torch.zeros(self.total_hw, dtype=torch.long, device=self.device)
            else:
                hw_indices = torch.tensor([hw for hw, p in E.keys()], dtype=torch.long, device=self.device)
                feature_counts = torch.bincount(hw_indices, minlength=self.total_hw)

            min_selections = torch.min(feature_counts)
            candidate_features = torch.where(feature_counts == min_selections)[0]

            if candidate_features.numel() == 0:
                return 0, 0

            # Get the distances for candidate features
            candidate_distances = self._distances_transposed[candidate_features, :]

            mask = torch.ones_like(candidate_distances, dtype=torch.bool)
            if E:
                e_keys_tensor = torch.tensor(list(E.keys()), dtype=torch.long, device=self.device)
                for i, hw in enumerate(candidate_features):
                    p_indices_in_e = e_keys_tensor[e_keys_tensor[:, 0] == hw, 1]
                    if p_indices_in_e.numel() > 0:
                        mask[i, p_indices_in_e] = False

            candidate_distances[~mask] = float('inf')

            flat_idx = torch.argmin(candidate_distances.flatten())
            candidate_idx = flat_idx // self.num_prototypes
            best_p = (flat_idx % self.num_prototypes).item()
            best_hw = candidate_features[candidate_idx].item()

            return best_hw, best_p

        else:
            raise ValueError(f"Unknown paradigm: {self.paradigm}. Supported paradigms are 'triangle' and 'hypersphere'.")

    def _next_pair_progressive(self, E):
        """
        Alternative approach: Progressive selection strategy that naturally avoids repetitive selections.
        """
        if self.paradigm == "hypersphere":
            # Group existing pairs by feature vector
            feature_to_prototypes = {}
            for (hw, p) in E.keys():
                if hw not in feature_to_prototypes:
                    feature_to_prototypes[hw] = []
                feature_to_prototypes[hw].append(p)
            
            # Create mask for existing pairs
            mask = torch.ones((self.total_hw, self.num_prototypes), dtype=torch.bool, device=self.device)
            if E:
                existing_hw, existing_p = zip(*E.keys())
                existing_hw_tensor = torch.tensor(existing_hw, device=self.device)
                existing_p_tensor = torch.tensor(existing_p, device=self.device)
                mask[existing_hw_tensor, existing_p_tensor] = False
            
            # Compute base scores
            centers_to_protos = self.prototypes.squeeze((2,3)).squeeze(0) - self.estimated_centers.unsqueeze(1) 
            centers_to_feats = self.z.flatten(start_dim=2).swapaxes(1,2).swapaxes(0,1) - self.estimated_centers.unsqueeze(1)
            cos = nn.CosineSimilarity(div=2, eps=1e-8)
            cos_sim = cos(centers_to_protos, centers_to_feats)
            base_scores = torch.abs(cos_sim)
            
            # PROGRESSIVE PENALTY: Apply exponentially increasing penalty for repeated feature selections
            penalty_factor = torch.ones_like(base_scores)
            for hw in range(self.total_hw):
                if hw in feature_to_prototypes:
                    num_selections = len(feature_to_prototypes[hw])
                    # Exponential decay: each additional selection gets exponentially less likely
                    penalty_factor[hw, :] *= (0.5 ** num_selections)
            
            # Apply progressive penalty
            adjusted_scores = base_scores * penalty_factor
            adjusted_scores[~mask] = 0.0
            
            max_idx = torch.argmax(adjusted_scores)
            best_pair = (max_idx // self.num_prototypes, max_idx % self.num_prototypes)
            
            return best_pair
    
    def _next_batch_of_pairs(self, available_pairs_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Finds the next closest prototype for each patch using a persistent mask.

        Args:
            available_pairs_mask: A boolean tensor of shape (H*W, P) where True
                                indicates a pair is available to be selected.

        Returns:
            (hw_indices, proto_indices, distances): a tuple of tensors for the new pairs.
        """
        if self.paradigm == "triangle":
            # Apply the mask by setting unavailable distances to infinity
            # This avoids creating a full copy with .clone() if memory is a concern.
            distances_masked = torch.where(
                available_pairs_mask,
                self._distances_transposed,
                torch.tensor(float('inf'), device=self.device)
            )

            # Find the minimum distance (i.e., the best next prototype) for each patch
            min_dists, min_proto_indices = torch.min(distances_masked, dim=1)

            # Filter out patches where all prototypes have already been selected
            valid_mask = min_dists != float('inf')
            if not torch.any(valid_mask):
                return torch.empty(0), torch.empty(0), torch.empty(0)

            # Return the new pairs as tensors, letting the caller handle the dictionary
            hw_indices = self._hw_indices_cache[valid_mask]
            proto_indices = min_proto_indices[valid_mask]
            dists = min_dists[valid_mask]

            return hw_indices, proto_indices, dists
        
        elif self.paradigm == "hypersphere":
            # Batch implementation for hypersphere paradigm
        
            # P-C Shape (HW, P, D)
            centers_to_protos = self.prototypes.squeeze((2,3)).squeeze(0) - self.estimated_centers.unsqueeze(1) 
            
            # Z-C Shape # (H*W, 1, D)
            centers_to_feats = self._z_features_flat.unsqueeze(1) - self.estimated_centers.unsqueeze(1)
            
            # Compute cosine similarities for all pairs: (H*W, P)
            cos = nn.CosineSimilarity(dim=2, eps=1e-8)
            cos_sim = cos(centers_to_protos, centers_to_feats)
            base_scores = torch.abs(cos_sim)  # (H*W, P)
            
            # Modify the score to factor in the uncertainty (radius) of each feature vector.
            # This balances refining uncertain regions (large radius) with picking
            # geometrically optimal pairs (high cosine similarity).
            scores = self.estimated_radii * base_scores  # Shape: (H*W, 1) * (H*W, P) -> (H*W, P)
            
            # Set unavailable pairs to 0.0 so they won't be selected
            scores_masked = torch.where(
                available_pairs_mask,
                scores,
                torch.tensor(0.0, device=self.device)
            )
            
            # Find the maximum score (best prototype) for each patch
            max_scores, max_proto_indices = torch.max(scores_masked, dim=1)
            
            # Filter out patches where all prototypes have already been selected (score = 0)
            valid_mask = max_scores > 0.0
            if not torch.any(valid_mask):
                return torch.empty(0), torch.empty(0), torch.empty(0)
            
            hw_indices = self._hw_indices_cache[valid_mask]
            proto_indices = max_proto_indices[valid_mask]
            
            # Get the actual distances for the selected pairs
            selected_distances = self._distances_transposed[hw_indices, proto_indices]
            
            return hw_indices, proto_indices, selected_distances
        
        else:
            raise ValueError(f"Unknown paradigm: {self.paradigm}. Supported paradigms are 'triangle' and 'hypersphere'.")
        
    #########################
    # Backward Pass Methods #
    #########################
    
    def _is_explanation_valid(self, explanation_pairs: set[tuple[int, int]]) -> bool:
        """
        Helper function to check if a given set of (patch, prototype) pairs
        forms a valid explanation for the predicted class.
        """
        # Quickly build an explanation dictionary from the pairs to test
        temp_E = {pair: self._distances_transposed[pair].item() for pair in explanation_pairs}

        # Generate and verify bounds based on this temporary explanation
        lb, ub = self._generate_bounds(temp_E, paradigm=self.paradigm)
        unverified, _ = self._verify_explanation(lb, ub, [k for k in range(self.num_classes) if k != self.c])

        return len(unverified) == 0

    def _find_minimal_explanation_binary_search(self, E: dict) -> dict:
        """
        Performs a fast, binary-search-based backward pass to find a subset-minimal
        explanation. This is a high-performance alternative to the iterative removal loop.

        Args:
            E: The full explanation dictionary from the forward pass.

        Returns:
            E*: A new, subset-minimal explanation dictionary.
        """
        candidate_pairs = list(E.keys())
        
        # This recursive function finds the essential pairs within a list of candidates
        def find_essential_subset(pairs_to_check, known_essential_pairs):
            # Base case: If removing this entire chunk of pairs is still valid,
            # then none of them are essential. We can discard them all.
            if self._is_explanation_valid(known_essential_pairs):
                return set()

            # Base case: If we're down to a single pair, and we know some pair in this
            # "chunk" is essential, it must be this one.
            if len(pairs_to_check) == 1:
                return set(pairs_to_check)

            # Recursive step: Split the candidates and search each half
            mid = len(pairs_to_check) // 2
            first_half, second_half = pairs_to_check[:mid], pairs_to_check[mid:]
            
            # Find essential pairs in the first half, assuming the second half is present
            essential_in_first = find_essential_subset(
                first_half, known_essential_pairs.union(second_half)
            )
            
            # Now, find essential pairs in the second half, adding the essential
            # ones we just found from the first half to our set of knowns.
            essential_in_second = find_essential_subset(
                second_half, known_essential_pairs.union(essential_in_first)
            )

            return essential_in_first.union(essential_in_second)

        # Start the search with an empty set of known essentials
        minimal_pairs_set = find_essential_subset(candidate_pairs, set())

        # Reconstruct the minimal explanation dictionary from the essential pairs
        minimal_E = {pair: E[pair] for pair in minimal_pairs_set}
        return minimal_E

###############################################################################
