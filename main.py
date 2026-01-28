"""
Main entry point for LLM-guided ODE evolution experiment.
"""
import os
import warnings
import argparse
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

import numpy as np

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig
from tqdm import trange

from evolution import evolution_chain, init_score
from buffer import MultiDimExperienceBuffer
from config import BUFFER_FUNCTIONS_PER_PROMPT, DEFAULT_NUM_EQUATIONS, DE_TOLERANCE, BFGS_TOLERANCE, REMOVED_TERMS_FORGET_PROBABILITY
from init_func_str import *
from data_loader import load_dataframes, create_describe, create_df_dict
from io_utils import (
    generate_logs_dir_name,
    convert_to_serializable,
    save_result,
    print_experiment_config,
    save_final_report,
    update_generated_equations,
)
from compare import is_better_than

warnings.filterwarnings("ignore")
load_dotenv('.env')


def get_configurable(config: RunnableConfig) -> dict:
    """Extract configurable dictionary from RunnableConfig."""
    configurable = {}
    if hasattr(config, "configurable"):
        configurable_attr = config.configurable
        if isinstance(configurable_attr, dict):
            configurable = configurable_attr
        elif configurable_attr is not None:
            try:
                configurable = dict(configurable_attr)
            except (TypeError, ValueError):
                configurable = {}
    elif hasattr(config, "get"):
        configurable = config.get("configurable", {})
    else:
        configurable = getattr(config, "configurable", {})
        if not isinstance(configurable, dict):
            try:
                configurable = dict(configurable) if configurable is not None else {}
            except (TypeError, ValueError):
                configurable = {}
    return configurable


def create_initial_state(
    evo: evolution_chain, 
    df_dict: Dict[str, Any],
    init_func_str_list: List[str], 
    max_params: int,
    config: RunnableConfig
) -> Dict[str, Any]:
    """Create initial state for evolution."""
    configurable = get_configurable(config)
    describe = configurable.get("describe", "")
    
    # Create initial params list
    initial_params_list = [np.ones(max_params) for _ in init_func_str_list]
    
    # Calculate initial scores
    score = init_score(evo, init_func_str_list, df_dict, initial_params_list)
    
    return {
        "generated_code": init_func_str_list,
        "params_list": initial_params_list,
        "score": score,
        "describe": describe,
        "use_var_desc": configurable.get("use_var_desc", False),
        "use_differential_evolution": configurable.get("use_differential_evolution", False),
        "total_error_count": 0,
    }


def _sample_from_buffer(
    buffer: 'MultiDimExperienceBuffer',
    iteration: int,
    n_samples: int = BUFFER_FUNCTIONS_PER_PROMPT
) -> Optional[List[List[Dict]]]:
    """Sample programs from buffer and convert to serializable format.
    
    Args:
        buffer: Experience buffer instance
        iteration: Current evolution iteration
        n_samples: Number of samples per dimension
        
    Returns:
        List of dimension samples, each containing program dictionaries
    """
    stats = buffer.get_statistics(iteration=iteration)
    if stats['total_programs'] == 0:
        return None
    
    sampled_raw = buffer.boltzmann_sample(n_samples=n_samples, iteration=iteration)
    
    sampled_programs = []
    for dim_samples in sampled_raw:
        dim_converted = []
        for prog in dim_samples:
            prog_dict = prog._asdict()
            if hasattr(prog_dict['params'], 'tolist'):
                prog_dict['params'] = prog_dict['params'].tolist()
            dim_converted.append(prog_dict)
        sampled_programs.append(dim_converted)
    
    return sampled_programs


def run_evolution(
    evo: evolution_chain, 
    initial_state: Dict[str, Any],
    config: RunnableConfig, 
    evolution_num: int,
    logs_dir: Path, 
    iteration_json_dir: Path,
    problem_name: str,
    buffer: 'MultiDimExperienceBuffer' = None,
    use_buffer: bool = False,
    use_scientist: bool = False,

    dt_cols: List[str] = None,
    config_info: Dict[str, Any] = None
) -> List[Dict[str, Any]]:
    """Execute evolution process with pair-based tracking."""
    start_time_dt = datetime.now()
    result_list = [initial_state]
    
    # Track best pair (new architecture)
    best_pair: Optional[Dict[str, Any]] = None
    best_iteration: int = 0
    
    # For backward compatibility with save_final_report
    best_scores_per_dim: Dict[str, float] = {}
    best_code_per_dim: Dict[str, str] = {}
    best_params_per_dim: Dict[str, List[float]] = {}
    best_iteration_per_dim: Dict[str, int] = {}
    
    
    # Register initial state to buffer
    if use_buffer and buffer is not None:
        buffer.register_program(
            code_str_list=initial_state.get('generated_code', []),
            params_list=initial_state.get('params_list', []),
            score_dict=initial_state.get('score', {}),
        )
    
    configurable = get_configurable(config)
    
    for index in trange(1, evolution_num + 1):
        prev_state = result_list[index - 1]
        
        # Sample from buffer if enabled
        sampled_programs = None
        if use_buffer and buffer is not None:
            sampled_programs = _sample_from_buffer(buffer, index)
        
        # Build input for evolution step (pass best_pair)
        input_data = {
            "image_list": prev_state.get('image_list'),
            "generated_code": prev_state.get('generated_code'),
            "score": prev_state.get('score'),
            "params_list": prev_state.get('params_list'),
            "describe": configurable.get("describe", ""),
            "use_var_desc": configurable.get("use_var_desc", False),
            "use_differential_evolution": configurable.get("use_differential_evolution", False),
            "retry_count": 0,
            "max_retries": configurable.get("max_retries", 3),
            "sampled_programs": sampled_programs,
            "use_buffer": use_buffer,
            "use_scientist": use_scientist,
            "total_error_count": prev_state.get('total_error_count', 0),
            "research_notebook": prev_state.get('research_notebook'),
            # Pair-based fields
            "best_pair": best_pair,
            "prev_best_pair": prev_state.get('prev_best_pair'),
            "current_pair": prev_state.get('current_pair'),
            "previous_generation_pair": prev_state.get('previous_generation_pair'),
            # Remove list (accumulated across iterations)
            "removed_terms_per_dim": prev_state.get('removed_terms_per_dim'),
            # Iteration tracking
            "current_iteration": index,
            "total_iterations": evolution_num,
            "scientist_analysis_metadata": prev_state.get('scientist_analysis_metadata'),
        }
        
        evo_node = evo.link_nodes()
        try:
            result = evo_node.invoke(input_data, config)
        except Exception as e:
            import traceback
            print(f"\n[Evolution Error] iteration {index}: {e}", flush=True)
            traceback.print_exc()
            raise
        result_list.append(result)
        
        # Update best pair from result
        # Check both explicit best_pair from result and current_pair (in case graph didn't update best)
        candidates_for_best = []
        if result.get('best_pair'):
            candidates_for_best.append(result['best_pair'])
        if result.get('current_pair'):
            candidates_for_best.append(result['current_pair'])
            
        for candidate in candidates_for_best:
            if is_better_than(candidate, best_pair):
                best_pair = candidate
                best_iteration = index
        
        # Register result to experience buffer
        if use_buffer and buffer is not None:
            buffer.register_program(
                code_str_list=result.get('generated_code', []),
                params_list=result.get('params_list', []),
                score_dict=result.get('score', {}),
            )
        
        # Save results
        save_result(result, index, logs_dir, iteration_json_dir, problem_name)
        
        # Update generated_equations.json in real-time
        update_generated_equations(
            logs_dir=logs_dir,
            iteration=index,
            evaluated_candidates=result.get('evaluated_candidates', []),
            best_pair=best_pair
        )
    
    # Extract per-dimension data from best_pair
    # best_iteration_per_dim is now tracked in state (GraphState)
    final_state = result_list[-1] if result_list else {}
    if best_pair:
        for dt_col in dt_cols:
            best_scores_per_dim[dt_col] = best_pair['dim_scores'].get(dt_col, float('inf'))
            best_code_per_dim[dt_col] = best_pair['codes'].get(dt_col, '')
            best_params_per_dim[dt_col] = best_pair['dim_params'].get(dt_col, [])
            
        # Get from state if available, otherwise default to last iteration
        state_best_iter = final_state.get('best_iteration_per_dim', {})
        if state_best_iter:
            best_iteration_per_dim.update(state_best_iter)
        else:
            # Fallback (should not happen with new evolution.py)
             for dt_col in dt_cols:
                 best_iteration_per_dim[dt_col] = best_iteration
    
    # Final buffer summary & stats
    buffer_stats = {}
    if use_buffer and buffer is not None:
        buffer_stats = buffer.get_statistics(iteration=evolution_num)
    
    # Calculate duration
    end_time = datetime.now()
    duration = end_time - start_time_dt
    total_seconds = duration.total_seconds()
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    duration_str = f"{hours}h {minutes}m {seconds}s"
    
    # Get total error count
    total_error_count = result_list[-1].get('total_error_count', 0) if result_list else 0

    # Save Final Report
    save_final_report(
        logs_dir=logs_dir,
        problem_name=problem_name,
        evolution_num=evolution_num,
        duration_str=duration_str,
        total_error_count=total_error_count,
        config_info=config_info,
        buffer_stats=buffer_stats,
        best_scores_per_dim=best_scores_per_dim,
        best_iteration_per_dim=best_iteration_per_dim,
        best_code_per_dim=best_code_per_dim,
        best_params_per_dim=best_params_per_dim,
        buffer=buffer,
        research_notebook=result_list[-1].get('research_notebook'),
        best_iteration=best_iteration
    )
            
    print(f"\nFinal report saved to {logs_dir}")

    return result_list, buffer


def main(
    problem_name: str,
    max_params: int,
    dim: int,
    evolution_num: int,
    use_var_desc: bool = False,
    use_differential_evolution: bool = False,
    use_buffer: bool = False,
    use_scientist: bool = False,
    recursion_limit: int = 12,
    timeout: int = 180,
    max_retries: int = 2,
    sampler_model_name: str = "google/gemini-2.5-flash-lite",
    scientist_model_name: str = "google/gemini-2.5-flash-lite",
    num_equations: int = DEFAULT_NUM_EQUATIONS,
    de_tolerance: float = DE_TOLERANCE,
    bfgs_tolerance: float = BFGS_TOLERANCE,
    sigma: float = 0.0,
    use_gt: bool = False,
    forget_prob: float = REMOVED_TERMS_FORGET_PROBABILITY,
) -> None:
    """Main execution function."""
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    df_train, df_test_id, df_test_ood = load_dataframes(problem_name, dim, sigma)
    df_dict = create_df_dict(df_train, df_test_id, df_test_ood)
    
    # Get variable description if needed
    describe = ""
    if use_var_desc:
        try:
            describe = create_describe(problem_name)
        except ValueError:
            print("Variable description not found, continuing without it.")
    
    # Get initial function strings
    init_func_str_list = eval(f"init_func_str_{dim}D")
    
    # Initialize evolution chain
    evo = evolution_chain(
        df_train=df_train,
        df_test_id=df_test_id,
        df_test_ood=df_test_ood,
        max_params=max_params,
        dim=dim,
        num_equations=num_equations,
        de_tolerance=de_tolerance,
        bfgs_tolerance=bfgs_tolerance,
        use_gt=use_gt,
        sampler_model_name=sampler_model_name,
        scientist_model_name=scientist_model_name,
        forget_probability=forget_prob,
    )
    
    # Create config
    config = RunnableConfig(
        recursion_limit=recursion_limit,
        configurable={
            "timeout": timeout,
            "max_retries": max_retries,
            "describe": describe,
            "use_var_desc": use_var_desc,
            "use_differential_evolution": use_differential_evolution
        }
    )
    
    # Generate log directory name
    logs_dir_name = generate_logs_dir_name(
        use_var_desc,
        use_differential_evolution,
        start_time,
        use_buffer,
        use_scientist=use_scientist,
        sigma=sigma,
        use_gt=use_gt,
        forget_prob=forget_prob
    )
    
    # Sanitize model name for directory (replace / with _)
    model_name_safe = sampler_model_name.replace("/", "_")
    
    # Create directories: logs/problem_name/model_name/logs_dir_name/
    problem_logs_dir = Path("logs") / problem_name / model_name_safe
    problem_logs_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = problem_logs_dir / logs_dir_name
    logs_dir.mkdir(parents=True, exist_ok=True)
    iteration_json_dir = logs_dir / "iteration_json"
    iteration_json_dir.mkdir(parents=True, exist_ok=True)
    
    # Print experiment configuration
    print_experiment_config(
        problem_name, dim, max_params, evolution_num, recursion_limit,
        timeout, max_retries, sampler_model_name,
        use_var_desc, use_differential_evolution,
        logs_dir, start_time, use_buffer, use_scientist=use_scientist,
        use_gt=use_gt, forget_prob=forget_prob, de_tolerance=de_tolerance,
        bfgs_tolerance=bfgs_tolerance, num_equations=num_equations
    )
    
    # Create initial state
    initial_state = create_initial_state(
        evo, df_dict, init_func_str_list, max_params, config
    )
    
    # Initialize multi-dimensional experience buffer
    buffer = MultiDimExperienceBuffer(dim=dim) if use_buffer else None
    
    # Collect config info for report
    config_info = {
        "use_var_desc": use_var_desc,
        "use_differential_evolution": use_differential_evolution,
        "use_buffer": use_buffer,
        "use_scientist": use_scientist,
        "use_gt": use_gt,
        "problem_name": problem_name,
        "max_params": max_params,
        "dim": dim,
        "evolution_num": evolution_num,
        "recursion_limit": recursion_limit,
        "timeout": timeout,
        "max_retries": max_retries,
        "sampler_model_name": sampler_model_name,
        "scientist_model_name": scientist_model_name,
        "num_equations": num_equations,
        "de_tolerance": de_tolerance,
        "bfgs_tolerance": bfgs_tolerance,
        "forget_prob": forget_prob
    }
    
    # Execute evolution
    result_list, buffer = run_evolution(
        evo, initial_state, config, evolution_num,
        logs_dir, iteration_json_dir, problem_name, buffer, use_buffer,
        use_scientist, evo.get_func_names(),
        config_info=config_info
    )
    
    print(f"\nEvolution completed! Total {len(result_list)} results saved.")
    print(f"Log directory: {logs_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ODE evolution experiment")
    
    # Required parameters
    parser.add_argument(
        "--use_var_desc",
        type=lambda x: str(x).lower() in ['true', '1', 'yes', 'y'],
        default=False,
        help="Whether to use variable description (true/false, default: false)"
    )
    parser.add_argument(
        "--use_differential_evolution",
        type=lambda x: str(x).lower() in ['true', '1', 'yes', 'y'],
        default=True,
        help="Whether to use Differential Evolution for optimization (true/false, default: true)"
    )
    parser.add_argument(
        "--problem_name",
        type=str,
        required=True,
        help="Problem name (e.g., ode_053)"
    )
    parser.add_argument(
        "--max_params",
        type=int,
        required=True,
        help="Maximum number of parameters"
    )
    parser.add_argument(
        "--dim",
        type=int,
        required=True,
        help="Dimension (1, 2, 3, 4)"
    )
    parser.add_argument(
        "--evolution_num",
        type=int,
        required=True,
        help="Number of evolution iterations"
    )
    
    # Optional parameters
    parser.add_argument(
        "--recursion_limit",
        type=int,
        default=15,
        help="Recursion limit (default: 15)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Timeout in seconds (default: 180)"
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=2,
        help="Maximum retry count (default: 2)"
    )
    parser.add_argument(
        "--sampler_model_name",
        type=str,
        default="google/gemini-2.5-flash-lite",
        help="Model name for Sampler LLM (default: google/gemini-2.5-flash-lite)"
    )
    parser.add_argument(
        "--scientist_model_name",
        type=str,
        default="google/gemini-2.5-flash-lite",
        help="Model name for Scientist LLM (default: google/gemini-2.5-flash-lite)"
    )
    parser.add_argument(
        "--use_buffer",
        type=lambda x: str(x).lower() in ['true', '1', 'yes', 'y'],
        default=False,
        help="Whether to use experience buffer (true/false, default: false)"
    )
    parser.add_argument(
        "--use_scientist",
        type=lambda x: str(x).lower() in ['true', '1', 'yes', 'y'],
        default=False,
        help="Whether to run scientist for insight generation (true/false, default: false)"
    )
    parser.add_argument(
        "--num_equations",
        type=int,
        default=DEFAULT_NUM_EQUATIONS,
        help=f"Number of candidate equations to generate (default: {DEFAULT_NUM_EQUATIONS})"
    )
    parser.add_argument(
        "--de_tolerance",
        type=float,
        default=DE_TOLERANCE,
        help="Tolerance for Differential Evolution (overrides config)"
    )
    parser.add_argument(
        "--bfgs_tolerance",
        type=float,
        default=BFGS_TOLERANCE,
        help="Tolerance for BFGS Optimization (overrides config)"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.0,
        help="Noise level (0.0, 0.001, 0.01, 0.1, default: 0.0)"
    )
    parser.add_argument(
        "--use_gt",
        type=lambda x: x.lower() == 'true',
        default=False,
        help="Use ground truth (x*_t_gt) as target instead of gradient-based (x*_t)"
    )
    parser.add_argument(
        "--forget_prob",
        type=float,
        default=REMOVED_TERMS_FORGET_PROBABILITY,
        help=f"Probability to forget removed terms for re-exploration (default: {REMOVED_TERMS_FORGET_PROBABILITY})"
    )

    
    args = parser.parse_args()
    
    main(
        use_var_desc=args.use_var_desc,
        use_differential_evolution=args.use_differential_evolution,
        use_buffer=args.use_buffer,
        use_scientist=args.use_scientist,
        problem_name=args.problem_name,
        max_params=args.max_params,
        dim=args.dim,
        evolution_num=args.evolution_num,
        recursion_limit=args.recursion_limit,
        timeout=args.timeout,
        max_retries=args.max_retries,
        sampler_model_name=args.sampler_model_name,
        scientist_model_name=args.scientist_model_name,
        num_equations=args.num_equations,
        de_tolerance=args.de_tolerance,
        bfgs_tolerance=args.bfgs_tolerance,
        sigma=args.sigma,
        use_gt=args.use_gt,
        forget_prob=args.forget_prob,
    )

