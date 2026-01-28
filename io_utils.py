"""
I/O utility functions for LLM ODE evolution.

This module contains file I/O, JSON serialization, and logging functions.
"""

import json
import os

from typing import Any, Dict, List
from pathlib import Path

import numpy as np


def generate_logs_dir_name(
    use_var_desc: bool,
    use_differential_evolution: bool,
    start_time: str,
    use_buffer: bool = False,
    use_scientist: bool = False,
    sigma: float = 0.0,
    use_gt: bool = False,
    forget_prob: float = 0.1
) -> str:
    """Generate log directory name based on experiment settings.
    
    Args:
        use_var_desc: Whether to use variable description
        use_differential_evolution: Whether to use Differential Evolution
        start_time: Experiment start time (YYYYMMDD_HHMMSS format)
        use_buffer: Whether to use experience buffer
        use_scientist: Whether to use scientist agent
        sigma: Noise level (sigma)
        use_gt: Whether to use ground truth target
    
    Returns:
        Generated directory name string
    """
    flag_parts = []
    
    # Add gt prefix if using ground truth
    if use_gt:
        flag_parts.append("gt")
    
    # Add sigma part
    sigma_str = f"sigma{str(sigma).replace('.', '')}"
    flag_parts.append(sigma_str)
    
    if use_var_desc:
        flag_parts.append("desc")
    if use_differential_evolution:
        flag_parts.append("de")
    if use_buffer:
        flag_parts.append("buffer")
    if use_scientist:
        flag_parts.append("scientist")
        
    # Add forget probability if different from default or always?
    # User requested it to be explicit to avoid overlap.
    # Format: forget01, forget99
    # prob is float, e.g. 0.01 -> 01, 0.99 -> 99, 0.1 -> 1
    # Use simple replacement
    prob_str = str(forget_prob).replace('.', '')
    flag_parts.append(f"forget{prob_str}")
    
    if flag_parts:
        flag_str = "_".join(flag_parts)
        return f"{flag_str}_{start_time}"
    else:
        return start_time


def convert_to_serializable(value: Any) -> Any:
    """Convert Python objects to JSON-serializable format.
    
    Args:
        value: Any Python object
        
    Returns:
        JSON-serializable representation
    """
    if value is None:
        return None
    elif isinstance(value, (str, int, float, bool)):
        return value

    elif isinstance(value, np.ndarray):
        return value.tolist()
    elif isinstance(value, list):
        return [convert_to_serializable(item) for item in value]
    elif isinstance(value, dict):
        return {k: convert_to_serializable(v) for k, v in value.items()}
    elif callable(value):
        return str(value)
    else:
        return str(value)


def save_result(
    result: Dict[str, Any], 
    index: int, 
    logs_dir: Path,
    iteration_json_dir: Path,
    problem_name: str
) -> None:
    """Save evolution results as JSON.
    
    Args:
        result: Result dictionary from evolution
        index: Evolution iteration index
        logs_dir: Path to logs directory
        iteration_json_dir: Path to iteration_json directory
        problem_name: Problem name for file naming
    """
    try:
        # Prepare result for JSON (exclude non-serializable items)
        # Keys to exclude from JSON output (internal state or user requested removal)
        exclude_keys = {
            'func_list', 
            'initial_state',
            'prev_best_pair',
            'previous_generation_pair', 
            'global_improvement', 
            'local_improvement'
        }
        
        # Prepare result for JSON
        result_for_json = {}
        if result is None:
            print(f"Warning: result is None for iteration {index}, saving empty dict")
            result_for_json = {}
        else:
            for k, v in result.items():
                if k in exclude_keys:
                    continue
                    
                # exclude removed_terms_per_dim / sampled_programs if None (cleaner log)
                if k in ['removed_terms_per_dim', 'sampled_programs'] and v is None:
                    continue
                    
                result_for_json[k] = v
        
        # Warn if result_for_json is empty
        if not result_for_json:
            print(f"Warning: result_for_json is empty for iteration {index} of {problem_name}")
        
        json_result = convert_to_serializable(result_for_json)
        save_path = iteration_json_dir / f"{problem_name}_{index}.json"
        
        # Ensure directory exists
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, ensure_ascii=False, indent=2)
            f.flush()  # Ensure data is written to disk
            os.fsync(f.fileno())  # Force write to disk
            
    except Exception as e:
        print(f"Error saving result for iteration {index} of {problem_name}: {e}")
        import traceback
        traceback.print_exc()
        # Try to save error info
        try:
            save_path = iteration_json_dir / f"{problem_name}_{index}.json"
            with open(save_path, 'w', encoding='utf-8') as f:
                error_info = {
                    "error": str(e),
                    "iteration": index,
                    "problem_name": problem_name
                }
                json.dump(error_info, f, ensure_ascii=False, indent=2)
        except:
            pass


def update_generated_equations(
    logs_dir: Path,
    iteration: int,
    evaluated_candidates: List[Dict[str, Any]],
    best_pair: Dict[str, Any] = None
) -> None:
    """Update generated_equations.json with all candidates from this iteration.
    
    This function is called after each evolution iteration to provide real-time logging.
    
    Args:
        logs_dir: Path to logs directory
        iteration: Current iteration number
        evaluated_candidates: List of all evaluated ODE system pairs from this iteration
        best_pair: Current global best pair (optional, for reference)
    """
    from datetime import datetime
    
    report_dir = logs_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    equations_file = report_dir / "generated_equations.json"
    
    # Load existing data or create new structure
    if equations_file.exists():
        with open(equations_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        data = {
            "description": "All generated ODE equations and their MSE scores per iteration",
            "iterations": [],
            "global_best": None
        }
    
    # Build candidates list for this iteration
    candidates_data = []
    for idx, candidate in enumerate(evaluated_candidates or []):
        candidate_entry = {
            "candidate_index": idx,
            "equations": convert_to_serializable(candidate.get('codes', {})),
            "dim_scores": convert_to_serializable(candidate.get('dim_scores', {})),
            "params": convert_to_serializable(candidate.get('dim_params', {})),
            "optimization_methods": convert_to_serializable(candidate.get('dim_opt_methods', {})),
            "reasoning": candidate.get('pair_reasoning', '')
        }
        candidates_data.append(candidate_entry)
    
    # Keep candidates in the given order (or we could sort by first dim score if needed)
    pass
    
    # Create iteration entry
    iteration_entry = {
        "iteration": iteration,
        "timestamp": datetime.now().isoformat(),
        "num_candidates": len(candidates_data),
        "best_candidate_scores": candidates_data[0]['dim_scores'] if candidates_data else None,
        "candidates": candidates_data
    }
    
    # Update or append iteration
    existing_iterations = {entry['iteration']: i for i, entry in enumerate(data['iterations'])}
    if iteration in existing_iterations:
        data['iterations'][existing_iterations[iteration]] = iteration_entry
    else:
        data['iterations'].append(iteration_entry)
    
    if best_pair:
        data['global_best'] = {
            "equations": convert_to_serializable(best_pair.get('codes', {})),
            "dim_scores": convert_to_serializable(best_pair.get('dim_scores', {})),
            "params": convert_to_serializable(best_pair.get('dim_params', {}))
        }
    
    # Write updated data
    with open(equations_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def print_experiment_config(
    problem_name: str, 
    dim: int, 
    max_params: int, 
    evolution_num: int,
    recursion_limit: int, 
    timeout: int, 
    max_retries: int, 
    model_name: str,
    use_var_desc: bool,
    use_differential_evolution: bool, 
    logs_dir: Path, 
    start_time: str,
    use_buffer: bool = False,
    use_scientist: bool = False,
    use_gt: bool = False,
    forget_prob: float = 0.1,
    de_tolerance: float = 1e-20,
    bfgs_tolerance: float = 1e-25,
    num_equations: int = 3
) -> None:
    """Print experiment configuration to console."""
    print("\n" + "="*60)
    print("Evolution Experiment Configuration")
    print("="*60)
    print(f"Problem name: {problem_name}")
    print(f"Dimension: {dim}D")
    print(f"Max parameters: {max_params}")
    print(f"Evolution iterations: {evolution_num}")
    print(f"Recursion limit: {recursion_limit}")
    print(f"Timeout: {timeout}s")
    print(f"Max retries: {max_retries}")
    print(f"Model name: {model_name}")
    print(f"Use variable description: {use_var_desc}")
    print(f"Use Differential Evolution: {use_differential_evolution}")
    print(f"Use experience buffer: {use_buffer}")
    print(f"Use scientist agent: {use_scientist}")
    print(f"Use ground truth targets: {use_gt}")
    print(f"Forget probability: {forget_prob}")
    print(f"DE tolerance: {de_tolerance}")
    print(f"BFGS tolerance: {bfgs_tolerance}")
    print(f"Num equations: {num_equations}")
    print(f"Log directory: {logs_dir}")
    print(f"Start time: {start_time}")
    print("="*60 + "\n")


def save_final_report(
    logs_dir: Path,
    problem_name: str,
    evolution_num: int,
    duration_str: str,
    total_error_count: int,
    config_info: Dict[str, Any],
    buffer_stats: Dict[str, Any],
    best_scores_per_dim: Dict[str, float],
    best_iteration_per_dim: Dict[str, int],
    best_code_per_dim: Dict[str, str],
    best_params_per_dim: Dict[str, List[float]],
    buffer: Any = None,
    research_notebook: Dict[str, Any] = None,
    best_iteration: int = None
) -> None:
    """Save final experiment reports (JSON, Text, Buffer Dump).
    
    Args:
        logs_dir: Path to logs directory
        problem_name: Problem name
        evolution_num: Number of iterations
        duration_str: Formatted duration string
        total_error_count: Total accumulated errors
        config_info: Configuration dictionary
        buffer_stats: Buffer statistics dictionary
        best_scores_per_dim: Best scores per dimension
        best_iteration_per_dim: Best iteration per dimension
        best_code_per_dim: Best code per dimension
        best_params_per_dim: Best parameters per dimension
        use_buffer: Whether buffer was enabled
        buffer: Buffer object (optional, for dumping)
        research_notebook: Final state of Research Notebook
    """
    report_dir = logs_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save Buffer Dump
    if buffer is not None:
        buffer_dump = buffer.get_all_programs()
        with open(report_dir / "buffer_dump.json", "w") as f:
            json.dump(convert_to_serializable(buffer_dump), f, indent=4)
            
    # 2. Prepare Final Report Dictionary
    final_report = {
        "problem_name": problem_name,
        "evolution_num": evolution_num,
        "total_duration": duration_str,
        "total_error_count": total_error_count,
        "config": config_info or {},
        "buffer_stats": buffer_stats,
        "best_scores_per_dim": best_scores_per_dim,
        "best_iteration_per_dim": best_iteration_per_dim,
        "best_code_per_dim": best_code_per_dim,
        "best_params_per_dim": best_params_per_dim,
        # Pair-based architecture fields
        "best_iteration": best_iteration,
        # Scientist logging fields
        "research_notebook": research_notebook or {}
    }
    
    # 3. Save Final Report JSON
    with open(report_dir / "final_report.json", "w") as f:
        json.dump(convert_to_serializable(final_report), f, indent=4)
        
    # 4. Save Final Report Text
    with open(report_dir / "final_report.txt", "w") as f:
        f.write(f"Final Evolution Report: {problem_name}\n")
        f.write("=======================================\n")
        
        # Write Config Info
        f.write("\n[Experiment Configuration]\n")
        f.write(f"  Total Duration: {duration_str}\n")
        f.write(f"  Total Errors Occurred: {total_error_count}\n")
        if config_info:
            for k, v in config_info.items():
                f.write(f"  {k}: {v}\n")
        
        # Write Scientst Info
        if research_notebook:
            f.write("\n[Scientist Report]\n")
            if research_notebook:
                f.write("  Research Notebook Status:\n")
                f.write(f"    Structural Learnings: {len(research_notebook.get('structural_learnings', []))}\n")
                f.write(f"    Accumulated Insight Length: {len(research_notebook.get('accumulated_insight', ''))} chars\n")
                if research_notebook.get('next_experiment_suggestion'):
                    f.write(f"    Next Experiment Suggestion: {research_notebook.get('next_experiment_suggestion')[:100]}...\n")

        # Write Buffer Stats
        f.write("\n[Buffer Statistics]\n")
        if buffer_stats:
            f.write(f"  Total Programs: {buffer_stats.get('total_programs')}\n")
            f.write(f"  Total Clusters: {buffer_stats.get('total_clusters')}\n")
            if 'best_internal_score' in buffer_stats:
                f.write(f"  Best Internal Score: {buffer_stats['best_internal_score']}\n")

            if 'clusters' in buffer_stats:
                f.write("  Clusters (Grouped by Internal Score, Top 5):\n")
                # Clusters are already sorted by MSE in buffer.py
                clusters = buffer_stats['clusters']
                
                for i, c_info in enumerate(clusters[:5]): # Show top 5 clusters
                     f.write(f"    Cluster {i+1}:\n")
                     f.write(f"      Internal Score: {c_info['internal_score']:.6e}\n")
                     f.write(f"      Count: {c_info['num_programs']}\n")
                     if 'example_codes' in c_info:
                         for dim_idx, code in enumerate(c_info['example_codes']):
                             code_snippet = code.replace('\n', '\n        ')
                             # Truncate if too long (optional, but good for summary)
                             if len(code_snippet) > 300:
                                 code_snippet = code_snippet[:300] + "..."
                             f.write(f"      Example Code (x{dim_idx}): {code_snippet}\n")
        else:
            f.write("  BUFFER_NOT_USED\n")
        
        # Write Best ODE System Pair (Unified View)
        f.write("\n[Best ODE System Pair]\n")
        if best_iteration is not None:
            f.write(f"  Achieved at Iteration: {best_iteration}\n")
            f.write("  " + "="*40 + "\n")
            
            # Sort dimensions for consistent output (x0, x1, ...)
            sorted_dims = sorted(best_code_per_dim.keys())
            
            for dim in sorted_dims:
                score = best_scores_per_dim.get(dim, float('inf'))
                code = best_code_per_dim.get(dim, "N/A")
                params = best_params_per_dim.get(dim, [])
                
                f.write(f"  Dimension: {dim}\n")
                f.write(f"    Score (MSE): {score:.6e}\n")
                f.write(f"    Parameters: {params}\n")
                f.write(f"    Code:\n")
                # Indent code block
                code_indented = "\n".join(["      " + line for line in code.split('\n')])
                f.write(f"{code_indented}\n")
                f.write("  " + "-"*40 + "\n")
        else:
            f.write("  No best pair recorded.\n")


