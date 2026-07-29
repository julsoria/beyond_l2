
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm, trange
from typing import Optional
from pathlib import Path
from subset_minimal_axp import FormalExplanationBase, check_memory_usage, SpatialFormalExplanation, TopKFormalExplanation


class CosineSimFormalExplanation(SpatialFormalExplanation):
    """
    Formal Explanation for models using Cosine Similarity (e.g., TesNet).
    
    Uses the reverse of the Triangle Inequality to define bounds (on cosine similarities)

    Metric: Cosine Similarity
    Domain: [-1, 1]
    Paradigm: Triangle Inequality (strictly).
    """
    def __init__(self, model, device="cuda:0", **kwargs):
        super().__init__(model, device=device, **kwargs)
        assert self.paradigm in ['triangle', 'hypersphere'], "CosineSimFormalExplanation only supports 'triangle' or 'hypersphere' paradigms"
        # self.prototypes shape: (P, D) or (P, D, 1, 1)
        
        self.verbose = True
        
        # similarities output: (P, P, H, W) or (P, P)
        # We squeeze to ensure shape is exactly (P, P)
        with torch.no_grad():
            self.prototype_similarities = self.model.classifier.similarities(self.prototypes).squeeze().detach()

        # # debug
        # self.prototype_similarities = self.prototype_similarities.cpu()
        # # self.similarity_values = self.similarity_values.cpu()
        # np.savetxt("prototype_similarities.txt", self.prototype_similarities)
        # print("prototype_similarities shape:", self.prototype_similarities.shape)
        # # print("similarity_values shape:", self.similarity_values.shape)
        # print("prototype_similarities:", self.prototype_similarities)
        # # print("similarity_values:", self.similarity_values)
        # self.prototype_similarities = self.prototype_similarities.to(self.device)
        
        # --- OPTIMIZATION: Pre-compute Prototype Sines ---
        # Since prototype similarities are static, we pre-compute the clamped cosine
        # and the sine values: sin(theta) = sqrt(1 - cos^2(theta))
        self._protos_c = torch.clamp(self.prototype_similarities, -1.0, 1.0)
        self._protos_sin = torch.sqrt(1.0 - self._protos_c**2)
        
    def _initialize_explanation(self, x, y):
        """
        Natively initialize the explanation for CosineSimFormalExplanation,
        avoiding the base class Euclidean setup and explicitly seeding the
        spherical geometric trackers and bounds caches.
        """
        self.c = y.item()
        
        # 1. Compute closest prototypes (Initialization)
        min_dists, min_proto_indices = torch.min(self._distances_transposed, dim=1) # (HW,)
        
        # 2. Build Explanation Dictionary
        E_init = {
            (hw.item(), p.item()): d.item() 
            for hw, p, d in zip(self._hw_indices_cache, min_proto_indices, min_dists)
        }
        
        # 3. Setup Persistent Trackers (Required by backward pass pruning logic)
        self.active_mask = torch.zeros((self.total_hw, self.num_prototypes), dtype=torch.bool, device=self.device)
        self.active_mask[self._hw_indices_cache, min_proto_indices] = True
        
        self.known_distances = torch.full((self.total_hw, self.num_prototypes), float('nan'), device=self.device)
        self.known_distances[self._hw_indices_cache, min_proto_indices] = min_dists
        
        self.add_order = {}
        for hw, p in zip(self._hw_indices_cache.tolist(), min_proto_indices.tolist()):
            self.add_order[hw] = [p]

        # 4. Initialize Spherical Geometry States Natively
        if self.paradigm == "hypersphere":
            # Fetch similarities directly to convert to exact spherical angular radii
            sims_flat = self.similarity_values[0].flatten(start_dim=1).t()
            S_init = sims_flat[self._hw_indices_cache, min_proto_indices]
            r_init = torch.acos(torch.clamp(S_init, -1.0, 1.0))
            
            # Init native spherical trackers
            self._spherical_C_int = self.prototypes.squeeze()[min_proto_indices]  # (HW, D)
            self._spherical_r_int = r_init                                        # (HW,)
            
            # Link to base class variables so backward pass snapshots/restores work flawlessly
            self.h_centers = self._spherical_C_int
            self.h_radii = self._spherical_r_int.unsqueeze(1)
            
            # Satisfy base class _next_batch_of_pairs heuristic
            self.estimated_centers = self.h_centers
            self.estimated_radii = self.h_radii

        # 5. Force a full bounds generation to seed caches (lb/ub bounds)!
        # This prevents the forward pass from running blind with ub=1.0
        self._generate_bounds(E_init, paradigm=self.paradigm)
        
        return E_init

    def _generate_bounds(self, E, paradigm="triangle", new_pairs_only=None):
        if not hasattr(self, '_proto_pos_cache'):
            self._proto_pos_cache = self.prototypes.squeeze().unsqueeze(0).detach()
        
        if not hasattr(self, '_cached_lb_features'):
            self._cached_lb_features = torch.full((self.total_hw, self.P), -1.0, device=self.device)
            self._cached_ub_features = torch.full((self.total_hw, self.P),  1.0, device=self.device)
            self._is_exact = torch.zeros((self.total_hw, self.P), dtype=torch.bool, device=self.device)

        if not E and not new_pairs_only:
            return self._cached_lb_features.max(dim=0).values, self._cached_ub_features.max(dim=0).values

        # ====================================================================
        # 1. ULTRA-FAST DICTIONARY PARSING (Numpy C-Backend)
        # ====================================================================
        parsed_E = False
        all_hw_t, all_p_t, all_d_t = None, None, None

        if new_pairs_only:
            hw_t = torch.tensor([x[0] for x in new_pairs_only], device=self.device, dtype=torch.long)
            p_t = torch.tensor([x[1] for x in new_pairs_only], device=self.device, dtype=torch.long)
        else:
            keys = np.array(list(E.keys()), dtype=np.int64)
            hw_t = all_hw_t = torch.from_numpy(keys[:, 0]).to(self.device)
            p_t = all_p_t = torch.from_numpy(keys[:, 1]).to(self.device)
            parsed_E = True
            
            self._cached_lb_features.fill_(-1.0)
            self._cached_ub_features.fill_(1.0)
            self._is_exact.fill_(False)

        # Get exact similarities directly from the network output (BIT-PERFECT FIX)
        sims_flat = self.similarity_values[0].flatten(start_dim=1).t()
        d_t = sims_flat[hw_t, p_t]
        
        # Link the global variable if doing a full recompute
        if parsed_E:
            all_d_t = d_t
            
        self._is_exact[hw_t, p_t] = True

        # ====================================================================
        # TRIANGLE PARADIGM (No Scatter Reduce)
        # ====================================================================
        if paradigm == "triangle":
            C_current = torch.full((self.total_hw, self.P), -float('inf'), device=self.device)
            C_current[hw_t, p_t] = d_t
            
            for hw in torch.unique(hw_t):
                valid_mask = ~torch.isinf(C_current[hw])
                obs_ps = torch.where(valid_mask)[0]
                if len(obs_ps) == 0: continue
                
                sims_c = torch.clamp(C_current[hw, obs_ps].unsqueeze(1), -1.0, 1.0)
                sims_sin = torch.sqrt(1.0 - sims_c**2)
                
                protos_c = self._protos_c[obs_ps]
                protos_sin = self._protos_sin[obs_ps]
                
                term1 = sims_c * protos_c
                term2 = sims_sin * protos_sin
                
                lb_batch = term1 - term2
                ub_batch = term1 + term2
                
                wrap_around_mask = (sims_c + protos_c) < 0
                lb_batch = torch.where(wrap_around_mask, torch.tensor(-1.0, device=self.device), lb_batch)
                
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
        # HYPERSPHERE PARADIGM (Vectorized Fast Filtering)
        # ====================================================================
        elif paradigm == "hypersphere":
            if not hasattr(self, '_spherical_C_int'):
                self._spherical_C_int = torch.zeros((self.total_hw, self.prototypes.shape[1]), device=self.device)
                self._spherical_r_int = torch.full((self.total_hw,), -1.0, device=self.device)
                
            if parsed_E:
                self._spherical_r_int.fill_(-1.0)
                
                # Bypasses the triple python loop. Creates a Boolean masking matrix of E.
                # E_mask shape (HW, P) evaluates in C-backend instantly.
                E_mask = torch.zeros((self.total_hw, self.P), dtype=torch.bool, device=self.device)
                E_mask[all_hw_t, all_p_t] = True
                
                valid_hws = []
                valid_ps = []
                # Only loops over the physical patches (max 196) and exactly known chronological additions
                for hw in range(self.total_hw):
                    if hw in self.add_order:
                        for p in self.add_order[hw]:
                            if E_mask[hw, p]:
                                valid_hws.append(hw)
                                valid_ps.append(p)
                                
                hw_t = torch.tensor(valid_hws, device=self.device)
                p_t = torch.tensor(valid_ps, device=self.device)
                d_t = sims_flat[hw_t, p_t]

            r_new = torch.acos(torch.clamp(d_t, -1.0, 1.0))
            p_new = self.prototypes.squeeze()[p_t]
            
            for i, hw in enumerate(hw_t):
                if self._spherical_r_int[hw] < 0:
                    self._spherical_C_int[hw] = p_new[i]
                    self._spherical_r_int[hw] = r_new[i]
                else:
                    C_1, r_1 = self._spherical_C_int[hw], self._spherical_r_int[hw]
                    C_2, r_2 = p_new[i], r_new[i]
                    
                    cos_d = torch.clamp(torch.dot(C_1, C_2), -1.0, 1.0)
                    d_angle = torch.acos(cos_d)
                    
                    if d_angle > 1e-5:
                        num = torch.cos(r_2) - torch.cos(r_1) * cos_d
                        den = torch.cos(r_1) * torch.sin(d_angle)
                        alpha = torch.atan2(num, den)
                        
                        r_int = torch.acos(torch.clamp(torch.cos(r_1) / torch.cos(alpha), -1.0, 1.0))
                        C_int = (torch.sin(d_angle - alpha) * C_1 + torch.sin(alpha) * C_2) / torch.sin(d_angle)
                        
                        self._spherical_C_int[hw] = torch.nn.functional.normalize(C_int, dim=0)
                        self._spherical_r_int[hw] = r_int
                        
            self.estimated_centers = self._spherical_C_int
            self.estimated_radii = torch.where(self._spherical_r_int < 0, torch.tensor(torch.pi, device=self.device), self._spherical_r_int).unsqueeze(1)
            
            if len(hw_t) > 0:
                active_hws = torch.unique(hw_t)
                C_int_batch = self._spherical_C_int[active_hws]
                r_int_batch = self._spherical_r_int[active_hws].unsqueeze(1)
                
                cos_D = torch.matmul(C_int_batch, self.prototypes.squeeze().t())
                D_angle = torch.acos(torch.clamp(cos_D, -1.0, 1.0))
                
                lb_sim_batch = torch.cos(torch.clamp(D_angle + r_int_batch, max=torch.pi))
                ub_sim_batch = torch.cos(torch.clamp(D_angle - r_int_batch, min=0.0))
                
                exact_hw_mask = self._is_exact[active_hws]
                self._cached_ub_features[active_hws] = torch.where(exact_hw_mask, self._cached_ub_features[active_hws], ub_sim_batch)
                self._cached_lb_features[active_hws] = torch.where(exact_hw_mask, self._cached_lb_features[active_hws], lb_sim_batch)

        # ====================================================================
        # GLOBAL EXACT OVERWRITE
        # ====================================================================
        if parsed_E:
            self._cached_lb_features[all_hw_t, all_p_t] = all_d_t
            self._cached_ub_features[all_hw_t, all_p_t] = all_d_t
        elif new_pairs_only:
            self._cached_lb_features[hw_t, p_t] = d_t
            self._cached_ub_features[hw_t, p_t] = d_t

        return self._cached_lb_features.max(dim=0).values, self._cached_ub_features.max(dim=0).values
    

    def _verify_explanation(self, lower_bound, upper_bound, unverified_classes):
        lower_bound_sim = lower_bound
        upper_bound_sim = upper_bound
        
        weights = self.weights
        with torch.no_grad():
            selected_class = self.c
            predicted_class_weights = weights[selected_class]
            batch_selector = weights > predicted_class_weights
            
            # CRITICAL FIX 1: Use torch.where for unified codebase safety
            similarities_to_check = torch.where(batch_selector, upper_bound_sim, lower_bound_sim)
            decision_output = torch.matmul(similarities_to_check, weights.t())

        new_unverified = unverified_classes.copy()
        
        # CRITICAL FIX 2: Initialize directly on the GPU
        unverified_conf = torch.full((self.num_classes,), -float("inf"), device=self.device)
        
        for uidx in unverified_classes:
            if decision_output[uidx, selected_class] > decision_output[uidx, uidx]:
                new_unverified.remove(uidx)
            else:
                unverified_conf[uidx] = decision_output[uidx, uidx]
                
        return new_unverified, unverified_conf.max(dim=0)[0]

    def _update_hypersphere_state_incremental(self, new_pairs: dict):
        """
        Track the chronological order of added pairs so the backward pass 
        can perfectly reconstruct the non-commutative spherical intersections.
        """
        for (hw, p), dist in new_pairs.items():
            self.active_mask[hw, p] = True
            self.known_distances[hw, p] = dist
            
            if hw not in self.add_order:
                self.add_order[hw] = []
            # Only append if not already in list (idempotency fix for backward pass)
            if p not in self.add_order[hw]:
                self.add_order[hw].append(p)

    def _rebuild_hypersphere_state_single(self, hw_idx: int):
        # Disable base class Euclidean state tracking.
        pass


class IsotropicGaussianFormalExplanation(SpatialFormalExplanation):
    """
    Formal Explanation for Isotropic Gaussian models using the 
    Log-Density Transform: S = -0.5 * d_mahalanobis^2 - log(Z).
    
    Supports both paradigm="triangle" and paradigm="hypersphere" (HIA) by mapping 
    similarities into a universal Euclidean space, using the base class's native 
    Euclidean geometric operations, and mapping the resulting bounds back to similarities.
    """
    def __init__(self, model, device="cuda:0", **kwargs):
        super().__init__(model, device=device, **kwargs)
        
        # 1. Extract covariances (assume isotropic, shape (P,))
        self.covs = self.model.classifier.covariances.detach().to(self.device)
        assert self.covs.dim() == 1, "Covariances must be 1D (isotropic) for this explainer."
        
        protos = self.prototypes.squeeze().detach() # (P, D)
        self.D = protos.shape[1]
        
        # 2. Precompute normalization constants: log(Z_j) = (D/2) * log(2 * pi * sigma_j^2)
        self.log_Z = 0.5 * self.D * torch.log(2 * np.pi * self.covs)
        
        # 3. Force the internal prototype distances to be exact Euclidean distances
        # This guarantees the base class HIA and TI math operates in true Euclidean space.
        diff = protos.unsqueeze(1) - protos.unsqueeze(0)
        self.prototype_distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
        
    def _compute_euclidean_distances(self, x):
        """ 
        Step 1: Inverse Mapping 
        Similarities -> Universal Euclidean Distances 
        """
        sims = self.model.similarities(x) # (N, P, H, W)
        
        sigma_sq = self.covs.view(1, -1, 1, 1)
        log_Z = self.log_Z.view(1, -1, 1, 1)
        
        # d_euclid = sqrt( -2 * sigma^2 * (S + log Z) )
        d_sq = -2.0 * sigma_sq * (sims + log_Z)
        return torch.sqrt(torch.clamp(d_sq, min=0.0))
        
    def explain_one(self, x, y, verbose=False, batch_update=True, fast_backward=False):
        # 1. Point the expected distance function to our custom Inverse Mapping method
        self.feature_distance_fn = self._compute_euclidean_distances
        
        # 2. The base class will now natively detect the function and route 
        # straight through the Euclidean geometry path!
        try:
            return super().explain_one(x, y, verbose, batch_update, fast_backward)
        finally:
            # Clean up the pointer just in case the explainer is reused
            self.feature_distance_fn = None
            
    def _generate_bounds(self, E, paradigm="triangle", new_pairs_only=None):
        """
        Step 3 & 4: Target-Metric Projection and Forward Mapping
        """
        # 1. Obtain strict Euclidean distance bounds from the base class
        lb_euclid, ub_euclid = super()._generate_bounds(E, paradigm=paradigm, new_pairs_only=new_pairs_only)
        
        # 2. Forward Mapping: Euclidean Distances -> Similarities
        # S_k(d) = -0.5 * (d_euclid^2 / sigma_k^2) - log(Z_k)
        
        # Maximum distance yields the Minimum similarity (Lower Bound)
        lb_sim = -0.5 * (ub_euclid ** 2) / self.covs - self.log_Z
        
        # Minimum distance yields the Maximum similarity (Upper Bound)
        ub_sim = -0.5 * (lb_euclid ** 2) / self.covs - self.log_Z
        
        return lb_sim, ub_sim

    def _verify_explanation(self, lower_bound, upper_bound, unverified_classes):
        # Override to avoid the base class trying to convert distances to similarities,
        # because our _generate_bounds elegantly returns exact similarity bounds.
        lower_bound_sim = lower_bound
        upper_bound_sim = upper_bound
        
        weights = self.weights
        with torch.no_grad():
            selected_class = self.c
            predicted_class_weights = weights[selected_class]
            batch_selector = weights > predicted_class_weights
            
            similarities_to_check = upper_bound_sim * batch_selector + lower_bound_sim * (~batch_selector)
            decision_output = torch.matmul(similarities_to_check, weights.t())
            
        new_unverified = unverified_classes.copy()
        unverified_conf = torch.ones(self.num_classes) * -float("inf")
        
        for uidx in unverified_classes:
            if decision_output[uidx, selected_class] > decision_output[uidx, uidx]:
                new_unverified.remove(uidx)
            else:
                unverified_conf[uidx] = decision_output[uidx, uidx]
                
        return new_unverified, torch.tensor(unverified_conf, device=self.device).max(dim=0)[0]


class IsotropicLogDistanceFormalExplanation(SpatialFormalExplanation):
    """
    Formal Explanation for Isotropic models using the Legacy ProtoPNet 
    Log Transform: S = log((d_mahalanobis^2 + 1) / (d_mahalanobis^2 + epsilon)).
    
    Supports both paradigm="triangle" and paradigm="hypersphere" (HIA) by mapping 
    similarities into a universal Euclidean space, using the base class's native 
    Euclidean geometric operations, and mapping the resulting bounds back to similarities.
    """
    def __init__(self, model, device="cuda:0", **kwargs):
        super().__init__(model, device=device, **kwargs)
        
        # 1. Extract covariances (assume isotropic, shape (P,))
        self.covs = self.model.classifier.covariances.detach().to(self.device)
        assert self.covs.dim() == 1, "Covariances must be 1D (isotropic) for this explainer."
        
        # 2. Extract the stability factor epsilon
        self.epsilon = self.model.classifier.similarity_layer.stability_factor
        
        protos = self.prototypes.squeeze().detach() # (P, D)
        
        # 3. Force the internal prototype distances to be exact Euclidean distances
        # This guarantees the base class HIA and TI math operates in true Euclidean space.
        diff = protos.unsqueeze(1) - protos.unsqueeze(0)
        self.prototype_distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
        
    def _compute_euclidean_distances(self, x):
        """ 
        Step 1: Inverse Mapping 
        Similarities -> Universal Euclidean Distances 
        """
        sims = self.model.similarities(x) # (N, P, H, W)
        
        sigma_sq = self.covs.view(1, -1, 1, 1)
        
        # Inverse log transform: d_sigma^2 = (1 - epsilon * exp(S)) / (exp(S) - 1)
        exp_S = torch.exp(sims)
        denominator = torch.clamp(exp_S - 1.0, min=1e-7) # Protect against div-zero
        
        d_sigma_sq = (1.0 - self.epsilon * exp_S) / denominator
        d_sigma_sq = torch.clamp(d_sigma_sq, min=0.0)
        
        # d_euclid = sigma * d_sigma
        return torch.sqrt(sigma_sq * d_sigma_sq)
        
    def explain_one(self, x, y, verbose=False, batch_update=True, fast_backward=False):
        # 1. Point the expected distance function to our custom Inverse Mapping method
        self.feature_distance_fn = self._compute_euclidean_distances
        
        # 2. The base class will now natively detect the function and route 
        # straight through the Euclidean geometry path!
        try:
            return super().explain_one(x, y, verbose, batch_update, fast_backward)
        finally:
            # Clean up the pointer just in case the explainer is reused
            self.feature_distance_fn = None
            
    def _generate_bounds(self, E, paradigm="triangle", new_pairs_only=None):
        """
        Step 3 & 4: Target-Metric Projection and Forward Mapping
        """
        # 1. Obtain strict Euclidean distance bounds from the base class
        lb_euclid, ub_euclid = super()._generate_bounds(E, paradigm=paradigm, new_pairs_only=new_pairs_only)
        
        # 2. Forward Mapping: Euclidean Distances -> Similarities
        def dist_to_sim(d_euclid):
            d_sigma_sq = (d_euclid ** 2) / self.covs
            sim = torch.log((d_sigma_sq + 1.0) / (d_sigma_sq + self.epsilon))
            
            # CRITICAL FIX: PyTorch evaluates inf/inf as NaN. 
            # Mathematically, lim_{d->inf} log((d^2+1)/(d^2+eps)) = log(1) = 0.0
            sim = torch.where(torch.isinf(d_euclid), torch.tensor(0.0, device=self.device), sim)
            return sim
        
        # Maximum Euclidean distance yields the Minimum similarity (Lower Bound)
        lb_sim = dist_to_sim(ub_euclid)
        
        # Minimum Euclidean distance yields the Maximum similarity (Upper Bound)
        ub_sim = dist_to_sim(lb_euclid)
        
        return lb_sim, ub_sim
    
    def _verify_explanation(self, lower_bound, upper_bound, unverified_classes):
        lower_bound_sim = lower_bound
        upper_bound_sim = upper_bound
        
        weights = self.weights
        with torch.no_grad():
            selected_class = self.c
            predicted_class_weights = weights[selected_class]
            batch_selector = weights > predicted_class_weights
            
            # CRITICAL FIX: Use torch.where to avoid NaN from (0.0 * -inf)
            similarities_to_check = torch.where(batch_selector, upper_bound_sim, lower_bound_sim)
            decision_output = torch.matmul(similarities_to_check, weights.t())
            
        new_unverified = unverified_classes.copy()
        
        # CRITICAL FIX: Added device=self.device to prevent cross-device crashes
        unverified_conf = torch.ones(self.num_classes, device=self.device) * -float("inf")
        
        for uidx in unverified_classes:
            if decision_output[uidx, selected_class] > decision_output[uidx, uidx]:
                new_unverified.remove(uidx)
            else:
                unverified_conf[uidx] = decision_output[uidx, uidx]
                
        return new_unverified, unverified_conf.max(dim=0)[0]

class PIPSparseFormalExplanation(FormalExplanationBase):
    """
    Formal Explanation for PIP-Net using Sparse Evidence reasoning.
    
    This explanation leverages the strict non-negativity (w >= 0) and sparsity 
    of PIP-Net's final classification layer. It optimizes subset-minimality by 
    greedily selecting unobserved prototypes that yield the maximum drop in the 
    competitor's upper bound (Delta = w * (1 - A)).
    """
    def __init__(self, model, device="cuda:0", **kwargs):
        super().__init__(model, device=device, **kwargs)
        # Ensure weights are accessed and strictly non-negative
        self.weights = torch.clamp(self.weights, min=0.0)

    def explain_one(self, x, y, verbose=False, **kwargs):
        check_memory_usage(threshold_mb=5000)
        x = x.to(self.device)
        self.batch_size = x.size(0)

        with torch.no_grad():
            similarities = self.model.similarities(x)
            logits = self.model(x)[0]  # (N, C)

        # 2. Global Max Pooling: a(p) = max_{h,w} \hat{z}_{h,w,j}
        A = similarities[0].flatten(start_dim=1).max(dim=1).values  # Shape: (P,)
        
        prediction_conf, predicted_class = torch.max(logits.squeeze(), dim=0)
        self.c = int(predicted_class.item())

        # W shape: (K, P)
        W = self.weights

        # 3. Initialization
        # Initialize with all positive evidence for the predicted class
        mask = W[self.c] > 0  
        
        # Exact lower bound for predicted class 
        o_c = torch.sum(W[self.c, mask] * A[mask])

        if verbose:
            t = tqdm(desc="Explaining (PIP Sparse)", unit="step", leave=False)
            step = 0

        # 4. Expansion Loop
        while True:
            # Calculate upper bounds for all classes simultaneously
            observed_scores = torch.sum(W[:, mask] * A[mask].unsqueeze(0), dim=1)  # (K,)
            unobserved_max_scores = torch.sum(W[:, ~mask], dim=1)  # (K,)
            upper_bounds = observed_scores + unobserved_max_scores  # (K,)

            # Identify unverified competitor classes
            unverified_mask = upper_bounds >= o_c
            unverified_mask[self.c] = False 

            if not unverified_mask.any():
                break # Formally verified
            
            # Find the worst-case competitor c*
            worst_k = torch.argmax(
                torch.where(unverified_mask, upper_bounds, torch.tensor(-float('inf'), device=self.device))
            )
            
            # --- THE OPTIMIZATION FIX ---
            # We want to maximize the bound reduction (Delta) for the worst competitor.
            # Delta = Weight * (WorstCase_A - True_A) = W * (1.0 - A)
            candidate_deltas = W[worst_k] * (1.0 - A)
            
            # Mask out already observed prototypes by setting their delta to -infinity
            candidate_deltas[mask] = -float('inf')  
            
            # Select the prototype that shrinks the bound the fastest
            j_star = torch.argmax(candidate_deltas)
            
            # Add to explanation E
            mask[j_star] = True

            if verbose:
                step += 1
                t.update(1)
                t.set_postfix_str(f"worst_comp: {worst_k.item()}, gap: {(upper_bounds[worst_k] - o_c).item():.3f}, added: {j_star.item()}")

        if verbose:
            t.close()

        # 5. Reconstruct the formal Explanation dictionary E
        E = {}
        active_indices = torch.where(mask)[0]
        for idx in active_indices:
            E[idx.item()] = A[idx].item()

        return E


class PIPSimplexFormalExplanation(FormalExplanationBase):
    """
    Formal Explanation for PIP-Net leveraging the Spatial Softmax (Simplex) Constraint.
    
    Constraint: At every spatial location (h, w), the sum of prototype probabilities is 1.0.
    Aggregation: Global Max Pooling (GMP).
    """
    
    def __init__(self, model, device="cuda:0", **kwargs):
        super().__init__(model, device=device, **kwargs)
        self.weights = torch.clamp(self.weights, min=0.0)
        
    def explain_one(self, x, y, verbose=False, **kwargs):
        check_memory_usage(threshold_mb=5000)
        x = x.to(self.device)
        self.batch_size = x.size(0)
        
        with torch.no_grad():
            probs = self.model.similarities(x) 
            logits = self.model(x)[0]
            
        P, H, W = probs.shape[1], probs.shape[2], probs.shape[3]
        total_hw = H * W
        
        z_hat = probs[0].flatten(start_dim=1).t() 
        
        prediction_conf, predicted_class = torch.max(logits.squeeze(), dim=0)
        self.c = int(predicted_class.item())
        
        # --- Initialization ---
        E = {} 
        unverified = [c_prime for c_prime in range(self.num_classes) if c_prime != self.c]
        
        M_obs = torch.zeros(total_hw, device=self.device)   
        lb_A = torch.zeros(P, device=self.device) # Lower bound is the known Max
        mask = torch.ones((total_hw, P), dtype=torch.bool, device=self.device) 
        
        # Pre-compute the batch selector to know when to use UB vs LB
        # True if the prototype helps the competitor more than the predicted class
        batch_selector = self.weights > self.weights[self.c] # Shape: (K, P)
        
        if verbose:
            t = tqdm(desc="Explaining (PIP Simplex)", unit="step", leave=False)
            step = 0

        # --- Expansion Loop ---
        while unverified:
            # 1. Compute strict bounds using the Simplex constraint
            M_rem = 1.0 - M_obs
            global_max_rem = torch.max(M_rem)
            ub_A = torch.maximum(lb_A, global_max_rem)
            
            # 2. Construct the absolute worst-case activation profile for EACH competitor
            # Shape: (K, P). Row k contains the worst-case activations to help class k beat class c
            A_worst = ub_A.unsqueeze(0) * batch_selector + lb_A.unsqueeze(0) * (~batch_selector)
            
            # 3. Compute the scores under these worst-case profiles
            # For each competitor k, we calculate both scores using A_worst[k]
            score_c = torch.sum(self.weights[self.c] * A_worst, dim=1) # (K,)
            score_k = torch.sum(self.weights * A_worst, dim=1)         # (K,)
            
            # 4. Verification
            still_unverified = []
            for idx in unverified:
                # If competitor k still beats or ties c in its absolute best-case scenario
                if score_k[idx] >= score_c[idx]:
                    still_unverified.append(idx)
            
            unverified = still_unverified
            
            if not unverified:
                break
                
            # 5. Greedy Selection Strategy
            # Target the pair with the highest mass to drop global_max_rem as fast as possible
            masked_z = torch.where(mask, z_hat, torch.tensor(-1.0, device=self.device))
            max_idx = torch.argmax(masked_z)
            
            hw_star = max_idx // P
            j_star = max_idx % P
            val = z_hat[hw_star, j_star]
            
            if val <= 0:
                raise ValueError("Ran out of positive probability mass before verifying.")
                
            # 6. Commit to Explanation
            mask[hw_star, j_star] = False
            E[(hw_star.item(), j_star.item())] = val.item()
            
            # Update States
            M_obs[hw_star] += val
            lb_A[j_star] = torch.max(lb_A[j_star], val)
            
            if verbose:
                step += 1
                t.update(1)
                t.set_postfix_str(f"n_unverif: {len(unverified)}, M_rem_max: {global_max_rem:.3f}")

        if verbose:
            t.close()

        return E
