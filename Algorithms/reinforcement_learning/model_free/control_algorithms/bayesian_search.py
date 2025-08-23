# python Algorithms/reinforcement_learning/model_free/control_algorithms/bayesian_search.py \
#   --algo "Algorithms.reinforcement_learning.model_free.control_algorithms.q_learning_function_approximation_tests:Q_Learning_Function_Approximation" \
#   --param-defaults '{"epsilon": 0.3, "epsilon_decay": 0.995, "epsilon_min": 0.05, "gamma": 0.99, "alpha": 0.5, "alpha_decay": 0.999, "alpha_min": 0.001, "step_size": 0.001, "step_size_decay": 0.995, "hidden": 20, "action_sampler": "v1"}' \
#   --param-ranges '{"epsilon": {"type": "float", "min": 0.05, "max": 1.0}, "step_size": {"type": "float", "suggest": "log", "min": 1e-5, "max": 1e-2}, "step_size_decay": {"type": "float", "min": 0.9, "max": 0.99999999}, "hidden": {"type": "categorical", "choices": [50, 100, 200, 400]}, "action_sampler": {"type": "categorical", "choices": ["v1", "v2"]}}' \
#   --env-params '{"step_penalty": -0.15, "hole_penalty": -5.0, "goal_reward": 20.0}' \
#   --num-trials 50 \
#   --num-jobs 10 \
#   --num-seeds 2 \
#   --study-name "q_learning_fn_approx_optuna"

# optuna-dashboard sqlite:///q_learning_fn_approx_optuna.db


import plotly.io as pio
# optional: install kaleido if you want PNGs (pip install kaleido)

import optuna
import numpy as np
from typing import Dict, Any
import sys
import os
import argparse
import json

# Add the current directory to path to import from grid_search
sys.path.insert(0, os.path.dirname(__file__))
from grid_search import _run_single, _load_json_arg

NUM_SEEDS = 2  # seeds per config
NUM_TRIALS = 50
NUM_JOBS = 10

def objective(trial, algo_cls_path, env_cls_path, env_kwargs, param_defaults, param_ranges):
    """Optuna objective function that samples parameters and runs experiments."""
    
    # Sample parameters based on the provided ranges
    sampled_params = {}
    for param_name, param_config in param_ranges.items():
        param_type = param_config.get("type", "float")
        if param_type == "float":
            if "log" in param_config.get("suggest", ""):
                sampled_params[param_name] = trial.suggest_loguniform(
                    param_name, 
                    param_config["min"], 
                    param_config["max"]
                )
            else:
                sampled_params[param_name] = trial.suggest_float(
                    param_name, 
                    param_config["min"], 
                    param_config["max"]
                )
        elif param_type == "categorical":
            sampled_params[param_name] = trial.suggest_categorical(
                param_name, 
                param_config["choices"]
            )
        elif param_type == "int":
            sampled_params[param_name] = trial.suggest_int(
                param_name, 
                param_config["min"], 
                param_config["max"]
            )

    # run the same config with multiple seeds and average
    results = []
    for seed_offset in range(NUM_SEEDS):
        seed = int(trial.number * 1000 + seed_offset)
        
        # Merge default params with sampled params
        algo_kwargs = dict(param_defaults)
        algo_kwargs.update(sampled_params)
        
        # Use _run_single from grid_search.py
        result = _run_single(
            env_cls_path=env_cls_path,
            algo_cls_path=algo_cls_path,
            env_kwargs=env_kwargs,
            algo_kwargs=algo_kwargs,
            num_iterations=10000,
            seed=seed,
            config_index=0,
        )
        results.append(result.metrics["final_policy_avg_gamma_return"])
    
    # use median to be robust to noise
    score = float(np.median(results))
    # optional: report intermediate to Optuna dashboard
    trial.set_user_attr("raw_returns", results)
    trial.set_user_attr("sampled_params", sampled_params)
    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna hyperparameter optimization for RL algorithms")
    parser.add_argument("--algo", required=True, help="Algorithm class path, e.g., 'Algorithms.reinforcement_learning.model_free.control_algorithms.q_learning_function_approximation_tests:Q_Learning_Function_Approximation'")
    parser.add_argument("--env", default="Algorithms.reinforcement_learning.model_free.problems.model_free_frozen_lake:ModelFreeFrozenLake", help="Environment class path")
    parser.add_argument("--param-defaults", default="{}", help="JSON string or path mapping param -> default values for algorithm __init__")
    parser.add_argument("--param-ranges", required=True, help="JSON string or path mapping param -> range config for Optuna sampling")
    parser.add_argument("--env-params", default="{}", help="JSON string or path for environment constructor params")
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations per run")
    parser.add_argument("--num-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--num-jobs", type=int, default=10, help="Number of parallel jobs for Optuna")
    parser.add_argument("--num-seeds", type=int, default=2, help="Number of seeds per trial")
    parser.add_argument("--study-name", type=str, default="optuna_study", help="Name for the Optuna study")
    
    args = parser.parse_args()
    
    # Update global constants
    NUM_SEEDS = args.num_seeds
    NUM_TRIALS = args.num_trials
    NUM_JOBS = args.num_jobs
    
    # Load parameters
    param_defaults = _load_json_arg(args.param_defaults)
    param_ranges = _load_json_arg(args.param_ranges)
    env_params = _load_json_arg(args.env_params)
    
    # Create runs directory if it doesn't exist
    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    # Create objective function with fixed parameters
    def objective_with_params(trial):
        return objective(trial, args.algo, args.env, env_params, param_defaults, param_ranges)
    
    study = optuna.create_study(
        direction="maximize", 
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        study_name=args.study_name,
        storage=f"sqlite:///{runs_dir}/{args.study_name}.db",
        load_if_exists=True,
    )
    study.optimize(
        objective_with_params, 
        n_trials=NUM_TRIALS, 
        n_jobs=NUM_JOBS,
        show_progress_bar=True,
    )  # n_jobs can be >1 with care
    # after study.optimize(...)
    # 1) optimization history (interactive)
    fig = optuna.visualization.plot_optimization_history(study)
    fig.write_html(f"{runs_dir}/{args.study_name}_optimization_history.html")

    # 2) parameter importance
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_html(f"{runs_dir}/{args.study_name}_param_importances.html")

    # 3) slice / parallel coordinate
    fig3 = optuna.visualization.plot_slice(study)
    fig3.write_html(f"{runs_dir}/{args.study_name}_slice.html")

    print("Best:", study.best_params, study.best_value)
    # You can then inspect importances:
    print("Param importance:", optuna.importance.get_param_importances(study))
