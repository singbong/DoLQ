"""
Comparison logic for ODE candidates.

This module centralizes the logic for determining which ODE candidate is 'better'.
Now uses a priority-based comparison on dimension-wise MSEs.
"""

from typing import Dict, List, Optional

def get_dim_scores_values(candidate: dict) -> List[float]:
    """Extract list of dimension scores sorted by key to ensure consistent order.
    
    Args:
        candidate: Candidate dictionary
        
    Returns:
        List of MSE values. Returns [inf] if invalid.
    """
    if not candidate:
        return [float('inf')]
        
    dim_scores = candidate.get('dim_scores', {})
    if not dim_scores:
        return [float('inf')]
        
    # Sort keys to ensure consistent order (e.g. x0_t, x1_t)
    # pair1[i] must correspond to same dimension as pair2[i]
    return [dim_scores[k] for k in sorted(dim_scores.keys())]


def compare_priority(pair1: List[float], pair2: List[float], epsilon: float = 1e-12) -> List[float]:
    """
    Compares two lists by sorting dimensions based on their minimum values.
    
    Returns:
        The 'better' list (smaller values preferred).
    """
    # If element counts differ (e.g., different dimensions) -> fallback to sum comparison
    # Using fallback as requested instead of raising an error.
    if len(pair1) != len(pair2):
        # Fallback to sum comparison
        return pair1 if sum(pair1) < sum(pair2) else pair2
    
    n = len(pair1)
    
    # 1. Calculate 'priority score' for each index (dimension)
    # Score criterion: the smaller value between the two pairs at that index (min(v1, v2))
    # Smaller values indicate more important dimensions (to be compared first)
    priorities = []
    for i in range(n):
        min_val = min(pair1[i], pair2[i])
        priorities.append((min_val, i))  # Store (value, index) tuple
    
    # 2. Sort indices by value in ascending order (smallest first)
    # e.g., [(0.1, 0), (0.5, 3), (10, 1)...] -> index order: 0 -> 3 -> 1 ...
    priorities.sort(key=lambda x: x[0])
    
    # Extract sorted index list
    sorted_indices = [item[1] for item in priorities]
    
    # 3. Compare in priority-sorted order
    for idx in sorted_indices:
        val1 = pair1[idx]
        val2 = pair2[idx]
        
        # If difference is >= epsilon, this dimension decides the winner
        if abs(val1 - val2) >= epsilon:
            return pair1 if val1 < val2 else pair2
            
        # If difference is < epsilon, proceed to next priority index (continue)

    # 4. If all dimensions are similar within epsilon
    # (completely equal or very similar) -> return pair1 (prefer keeping existing)
    return pair1


def is_better_than(candidate_new: Optional[dict], candidate_old: Optional[dict]) -> bool:
    """Return True if candidate_new is strictly better than candidate_old.
    
    Uses compare_priority logic.
    """
    if not candidate_new:
        return False
    
    if not candidate_old:
        # New candidate exists, old does not -> New is better
        return True
        
    scores_new = get_dim_scores_values(candidate_new)
    scores_old = get_dim_scores_values(candidate_old)
    
    better_scores = compare_priority(scores_new, scores_old)
    
    # If returned scores is the new one, then new is better (or equal)
    # But strict 'better' usually implies <. 
    # Logic in compare_priority returns pair1 if val1 < val2.
    # If equal, returns pair1.
    # Here we want is_better_than(new, old). 
    # If compare_priority(new, old) returns new, and they are NOT equal, then new is better.
    # Or simply: did compare_priority prefer new?
    
    # We need to distinguish "strictly better".
    # compare_priority returns the winner.
    if better_scores is scores_new:
        # Check if they are actually equal to old (prevent churn on equal)
        # compare_priority returns pair1 (new) if equal.
        # But usually is_better_than implies we want to switch only if strictly better for stability?
        # Alternatively, follow select_best_candidate logic.
        
        # Let's verify strict inequality for at least one differentiating dimension?
        # Or just rely on identity.
        # If compare_priority returns pair1 when equal, then is_better_than(A, A) returns True?
        # If is_better_than returns True on equality, we might have unnecessary updates.
        # Usually 'is_better_than' means strictly better.
        
        if scores_new == scores_old:
            return False
            
        return True
        
    return False


def select_best_candidate(candidates: List[dict]) -> Optional[dict]:
    """Select the single best candidate from a list of candidates using priority logic."""
    if not candidates:
        return None
    
    best_cand = candidates[0]
    for cand in candidates[1:]:
        if is_better_than(cand, best_cand):
            best_cand = cand
            
    return best_cand
