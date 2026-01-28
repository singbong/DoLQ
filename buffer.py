"""
Experience Buffer with Pair-based Storage and Boltzmann Sampling.

This module implements a system-wide experience buffer that stores ODE system pairs.
It clusters pairs by their Combined MSE and performs Boltzmann sampling on pairs
to preserve the coupling between dimensions (x0, x1, ...).
"""
from __future__ import annotations

from typing import List, Dict, Any, Optional, NamedTuple
import numpy as np
import scipy.special

from config import (
    BUFFER_SAMPLING_TEMPERATURE,
    BUFFER_FUNCTIONS_PER_PROMPT,
    BUFFER_MAX_SIZE,
)


class SystemProgramEntry(NamedTuple):
    """A complete ODE system pair entry in the experience buffer.
    
    Attributes:
        code_str_list: List of function code strings [x0_code, x1_code, ...]
        params_list: List of parameter arrays [x0_params, x1_params, ...]
        combined_mse: System-wide combined MSE (used for clustering)
        scores_per_dim: List of individual dimension MSEs
    """
    code_str_list: List[str]
    params_list: List[np.ndarray]
    scores_per_dim: List[float]


class DimProgramEntry(NamedTuple):
    """A single-dimension view of a program (for API compatibility).
    
    Attributes:
        code_str: Function code string for this dimension
        params: Parameter array for this function
        score: Train MSE score for this dimension
    """
    code_str: str
    params: np.ndarray
    score: float


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Returns the tempered softmax of 1D finite `logits`."""
    if not np.all(np.isfinite(logits)):
        # Replace non-finite values with min finite value or generic small value
        logits = np.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    
    if not np.issubdtype(logits.dtype, np.floating):
        logits = np.array(logits, dtype=np.float32)
    
    temperature = max(temperature, 1e-8)
    result = scipy.special.softmax(logits / temperature, axis=-1)
    
    # Ensure probabilities sum to 1
    if np.sum(result) == 0:
        result = np.ones_like(result) / len(result)
    else:
        result /= np.sum(result)
        
    return result


    def __init__(self, score: float, program: SystemProgramEntry):
        self._score = score  # internal score for sampling
        self._programs: List[SystemProgramEntry] = [program]
        # Use total length of all codes as a complexity metric
        self._lengths: List[int] = [sum(len(c) for c in program.code_str_list)]
    
    @property
    def score(self) -> float:
        """Negative score for Boltzmann sampling (lower MSE = higher prob)."""
        return -self._score
    
    @property
    def raw_score(self) -> float:
        return self._score
    
    @property
    def num_programs(self) -> int:
        return len(self._programs)
    
    def register_program(self, program: SystemProgramEntry) -> None:
        self._programs.append(program)
        self._lengths.append(sum(len(c) for c in program.code_str_list))
    
    def sample_program(self) -> SystemProgramEntry:
        """Sample a program, preferring shorter/simpler code."""
        if len(self._programs) == 1:
            return self._programs[0]
        
        min_len, max_len = min(self._lengths), max(self._lengths)
        if max_len == min_len:
            return self._programs[np.random.randint(len(self._programs))]
        
        # Softmax over negative lengths (shorter is better)
        normalized = (np.array(self._lengths) - min_len) / (max_len - min_len + 1e-6)
        probs = _softmax(-normalized, temperature=1.0)
        return self._programs[np.random.choice(len(self._programs), p=probs)]
    
    def get_best_program(self) -> SystemProgramEntry:
        """Return the simplest (shortest) program in this cluster."""
        return self._programs[np.argmin(self._lengths)]

    def get_all_programs(self) -> List[SystemProgramEntry]:
        """Return all programs in this cluster."""
        return self._programs


class MultiDimExperienceBuffer:
    """Experience Buffer storing Coupled ODE System Pairs.
    
    This replaces the old dimension-independent buffer logic.
    It now clusters by 'Combined MSE' and samples 'Pairs' together
    to preserve system dynamics coupling.
    """
    
    def __init__(
        self,
        dim: int,
        temperature_init: float = BUFFER_SAMPLING_TEMPERATURE,
        functions_per_prompt: int = BUFFER_FUNCTIONS_PER_PROMPT,
        max_size: int = BUFFER_MAX_SIZE,
    ):
        self.dim = dim
        self._temperature_init = temperature_init
        self._functions_per_prompt = functions_per_prompt
        self._max_size = max_size
        
        # Clusters key: Internal score -> SystemCluster
        self._clusters: Dict[float, SystemCluster] = {}
        self._num_programs: int = 0
        self._best_pair: Optional[SystemProgramEntry] = None
        self._best_total_mse: float = float('inf')
    
    @property
    def num_programs(self) -> int:
        return self._num_programs
    
    @property
    def num_clusters(self) -> int:
        return len(self._clusters)
    
    def register_program(
        self,
        code_str_list: List[str],
        params_list: List[np.ndarray],
        score_dict: Dict[str, List[float]],
    ) -> None:
        """Register a system pair.
        
        Args:
            code_str_list: List of function codes [x0, x1, ...]
            params_list: List of parameters [x0_p, x1_p, ...]
            score_dict: Dictionary containing 'train' scores list.
        """
        train_scores = score_dict.get('train', [])
        
        # Safety check
        if len(train_scores) < self.dim or len(code_str_list) < self.dim:
            return

        # Internal score for clustering/sampling (sum of MSEs)
        # This is strictly internal and not exposed as 'Combined MSE'
        total_mse = float(np.sum(train_scores[:self.dim]))
        
        # Create Entry
        entry = SystemProgramEntry(
            code_str_list=code_str_list[:self.dim],
            params_list=[np.array(p) if not isinstance(p, np.ndarray) else p for p in params_list[:self.dim]],
            scores_per_dim=train_scores[:self.dim]
        )
        
        # Add to Cluster
        if total_mse in self._clusters:
            self._clusters[total_mse].register_program(entry)
        else:
            self._clusters[total_mse] = SystemCluster(total_mse, entry)
        
        self._num_programs += 1
        
        # Track Global Best
        if total_mse < self._best_total_mse:
            self._best_total_mse = total_mse
            self._best_pair = entry
        
        # Prune if full
        if self._num_programs > self._max_size:
            self._prune_worst_cluster()
            
    def _prune_worst_cluster(self) -> None:
        if not self._clusters:
            return
        # Worst = Highest total MSE
        worst_key = max(self._clusters.keys())
        self._num_programs -= self._clusters[worst_key].num_programs
        del self._clusters[worst_key]
    
    def boltzmann_sample(
        self,
        n_samples: Optional[int] = None,
        temperature: Optional[float] = None,
        iteration: Optional[int] = None,
    ) -> List[List[DimProgramEntry]]:
        """Sample pairs using Boltzmann distribution on total MSE.
        
        Returns:
            List[List[DimProgramEntry]]: 
            Shape is [Dim][Sample].
            The k-th sample in dim 0 and k-th sample in dim 1 are from the SAME pair.
        """
        if not self._clusters:
            return []
        
        n_samples = n_samples or self._functions_per_prompt
        temperature = temperature or self._temperature_init
        
        # 1. Sample Clusters based on total MSE
        mse_keys = list(self._clusters.keys())
        cluster_scores = np.array([self._clusters[k].score for k in mse_keys]) # -MSE
        probs = _softmax(cluster_scores, temperature)
        
        # Limit n_samples
        n_nonzero = np.count_nonzero(probs)
        n_samples = min(n_samples, len(mse_keys), n_nonzero)
        if n_samples == 0:
            return []
            
        # Select indices of clusters
        cluster_indices = np.random.choice(len(mse_keys), size=n_samples, p=probs, replace=False)
        selected_clusters = [self._clusters[mse_keys[i]] for i in cluster_indices]
        
        # 2. Get 1 Representative Program from each selected cluster
        # (Using get_best_program inside cluster to prefer simpler code)
        selected_pairs: List[SystemProgramEntry] = [c.get_best_program() for c in selected_clusters]
        
        # 3. Transpose to [Dim][Sample] for API compatibility
        # result[d][s] = DimProgramEntry
        result: List[List[DimProgramEntry]] = [[] for _ in range(self.dim)]
        
        for pair in selected_pairs:
            for d in range(self.dim):
                dim_entry = DimProgramEntry(
                    code_str=pair.code_str_list[d],
                    params=pair.params_list[d],
                    score=pair.scores_per_dim[d]
                )
                result[d].append(dim_entry)
                
        return result

    def get_best_programs(self, n: int = 5) -> List[List[DimProgramEntry]]:
        """Get best pairs (Compatibility API)."""
        if not self._clusters:
            return [[] for _ in range(self.dim)]
            
        # Sort keys (total MSE) ascending
        sorted_keys = sorted(self._clusters.keys())
        
        best_pairs: List[SystemProgramEntry] = []
        for k in sorted_keys:
            # In each cluster, get the best (shortest) program
            best_pairs.append(self._clusters[k].get_best_program())
            if len(best_pairs) >= n:
                break
        
        # Transpose to Dim-wise list
        result: List[List[DimProgramEntry]] = [[] for _ in range(self.dim)]
        for pair in best_pairs:
            for d in range(self.dim):
                dim_entry = DimProgramEntry(
                    code_str=pair.code_str_list[d],
                    params=pair.params_list[d],
                    score=pair.scores_per_dim[d]
                )
                result[d].append(dim_entry)
        
        return result

    def get_statistics(self, iteration: Optional[int] = None) -> Dict[str, Any]:
        """Get buffer statistics."""
        # Sort clusters by score
        sorted_keys = sorted(self._clusters.keys())
        clusters_info = []
        for k in sorted_keys:
            best_in_cluster = self._clusters[k].get_best_program()
            clusters_info.append({
                'internal_score': k,
                'num_programs': self._clusters[k].num_programs,
                'example_codes': best_in_cluster.code_str_list
            })
            
        return {
            'dim': self.dim,
            'total_programs': self._num_programs,
            'total_clusters': self.num_clusters,
            'best_internal_score': self._best_total_mse if self._best_total_mse != float('inf') else None,
            'clusters': clusters_info
        }

    def get_all_programs(self) -> List[Dict[str, Any]]:
        """Return all system pairs (Dump format)."""
        all_pairs_data = []
        for cluster in self._clusters.values():
            for prog in cluster.get_all_programs():
                all_pairs_data.append({
                    'code_str_list': prog.code_str_list,
                    'params_list': [p.tolist() if hasattr(p, 'tolist') else p for p in prog.params_list],
                    'scores_per_dim': prog.scores_per_dim
                })
                
        # Sort by first dim score for readability
        all_pairs_data.sort(key=lambda x: x['scores_per_dim'][0] if x['scores_per_dim'] else 0)
        return all_pairs_data

    def __repr__(self) -> str:
        return (
            f"MultiDimExperienceBuffer(dim={self.dim}, "
            f"style=PairBased, "
            f"programs={self._num_programs}, "
            f"clusters={self.num_clusters})"
        )
