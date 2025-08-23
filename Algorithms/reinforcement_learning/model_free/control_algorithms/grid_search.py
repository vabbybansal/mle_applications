# python Algorithms/reinforcement_learning/model_free/control_algorithms/grid_search.py \
#   --algo Algorithms.reinforcement_learning.model_free.control_algorithms.q_learning_function_approximation:Q_Learning_Function_Approximation \
#   --param-defaults '{"epsilon":[0.3328218309968913], "hidden":[200], "epsilon_decay":[9948418061], "epsilon_min":[0.5], "alpha":[0.5], "alpha_decay":[0.999], "alpha_min":[0.001], "step_size_decay":[0.9999999897], "step_size":0.001}' \
#   --param-grid '{"step_size_decay":[0.9, 0.99, 0.999, 0.99999, 1.0], "epsilon":[0.1, 0.5, 0.7, 0.9, 0.99, 0.999, 0.99999, 1.0],"hidden":[50, 100, 200], "epsilon_decay":[0.9, 0.99, 0.995, 0.999, 0.9999, 0.99999]}' \
#   --fixed-params '{"gamma":0.99}' \
#   --env Algorithms.reinforcement_learning.model_free.problems.model_free_frozen_lake:ModelFreeFrozenLake \
#   --env-params '{"step_penalty":-0.15,"hole_penalty":-5.0,"goal_reward":20.0}' \
#   --iterations 10000 \
#   --ray-num-cpus 8 \
#   --plot --plot-metric final_policy_avg_gamma_return \
#   --runs-per-config 10 --seed-base 123

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from importlib import import_module
from inspect import signature, isclass
from typing import Any, Dict, List, Tuple

import numpy as np
import ray  # type: ignore


# Ensure repo root is importable so fully-qualified module paths work
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "../../../.."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _import_from_path(path: str) -> Any:
    """
    Import an attribute (usually a class) from a module path.
    Accepts either "pkg.mod:Attr" or dotted "pkg.mod.Attr".
    """
    if ":" in path:
        mod_name, attr_name = path.split(":", 1)
    else:
        parts = path.rsplit(".", 1)
        if len(parts) != 2:
            raise ValueError(f"Invalid import path '{path}'. Use 'module:Attr' or 'module.Attr'.")
        mod_name, attr_name = parts
    module = import_module(mod_name)
    try:
        return getattr(module, attr_name)
    except AttributeError as e:
        raise ImportError(f"Attribute '{attr_name}' not found in module '{mod_name}'.") from e


def _load_json_arg(value: str) -> Dict[str, Any]:
    """Load JSON either directly from a JSON string or from a file path if it exists."""
    if value is None:
        return {}
    value = value.strip()
    if not value:
        return {}
    if os.path.exists(value):
        with open(value, "r") as f:
            return json.load(f)
    # try parse as JSON string
    return json.loads(value)


def _set_global_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)


def _filter_kwargs_for_callable(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Return only those kwargs accepted by callable_obj's signature."""
    try:
        sig = signature(callable_obj)
    except Exception:
        return kwargs
    allowed = set(p.name for p in sig.parameters.values())
    return {k: v for k, v in kwargs.items() if k in allowed}


@dataclass
class RunResult:
    config_index: int
    seed: int
    params: Dict[str, Any]
    metrics: Dict[str, Any]


def _run_single(
    env_cls_path: str,
    algo_cls_path: str,
    env_kwargs: Dict[str, Any],
    algo_kwargs: Dict[str, Any],
    num_iterations: int,
    seed: int,
    config_index: int,
) -> RunResult:
    """Run a single experiment with given environment and algorithm configuration."""
    # Resolve classes
    env_cls = _import_from_path(env_cls_path)
    algo_cls = _import_from_path(algo_cls_path)
    if not isclass(env_cls) or not isclass(algo_cls):
        raise TypeError("Provided env/algo paths must point to classes.")

    _set_global_seeds(seed)

    # Instantiate environment
    env_init_kwargs = _filter_kwargs_for_callable(env_cls, env_kwargs)
    env = env_cls(**env_init_kwargs)

    # Instantiate algorithm with filtered kwargs
    algo_init_kwargs = dict(algo_kwargs)
    algo_init_kwargs["env"] = env
    algo_init_kwargs = _filter_kwargs_for_callable(algo_cls, algo_init_kwargs)
    algo = algo_cls(**algo_init_kwargs)

    # Fit
    fit_kwargs = _filter_kwargs_for_callable(getattr(algo, "fit"), {"num_iterations": num_iterations})
    _ = algo.fit(**fit_kwargs)

    # Collect metrics directly from the algorithm logger
    final_metrics: Dict[str, Any] = algo.logger["final_metrics"]

    # Use metrics as-is
    out_metrics = final_metrics

    return RunResult(
        config_index=config_index,
        seed=seed,
        params=algo_kwargs,
        metrics=out_metrics,
    )


def plot_results(results_records: List[Dict[str, Any]], plot_metric: str, start_time: float = None) -> None:
    """Plot scatter plots for each varied parameter."""
    import matplotlib.pyplot as plt
    from collections import defaultdict
    import os
    from datetime import datetime
    
    # Find varied parameters
    values_by_param: Dict[str, set] = defaultdict(set)
    for rec in results_records:
        for k, v in rec["config"].items():
            values_by_param[k].add(v)
    varied_params = [k for k, s in values_by_param.items() if len(s) > 1]
    
    if not varied_params:
        return
    
    n = len(varied_params)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(5 * cols + 1, 4 * rows))
    if n == 1:
        axs = [axs]
    else:
        axs = axs.flatten()

    for idx, p in enumerate(varied_params):
        ax = axs[idx]
        group_values = sorted(list(values_by_param[p]))
        
        for gv in group_values:
            vals = [float(rec["metrics"][plot_metric]) for rec in results_records if rec["config"][p] == gv]
            x_coords = [gv] * len(vals)
            ax.scatter(x_coords, vals, alpha=0.6, facecolors='none', edgecolors='black')

        ax.set_xlabel(p)
        ax.set_ylabel(plot_metric)
        ax.set_title(f"Param: {p}")
        ax.grid(True, alpha=0.3)

    for j in range(len(varied_params), len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout()
    
    # Create runs directory if it doesn't exist
    runs_dir = os.path.join(os.path.dirname(__file__), "runs")
    os.makedirs(runs_dir, exist_ok=True)
    
    # Generate filename with date and time
    if start_time is not None:
        dt = datetime.fromtimestamp(start_time)
        timestamp = dt.strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    filename = f"grid_search_{timestamp}.png"
    filepath = os.path.join(runs_dir, filename)
    
    # Save the plot
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Plot saved to: {filepath}")
    
    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generic grid search for RL control algorithms")
    parser.add_argument("--algo", required=True, help="Algorithm class path, e.g., 'Algorithms.reinforcement_learning.model_free.control_algorithms.q_learning_function_approximation:Q_Learning_Function_Approximation'")
    parser.add_argument("--env", default="Algorithms.reinforcement_learning.model_free.problems.model_free_frozen_lake:ModelFreeFrozenLake", help="Environment class path")
    parser.add_argument("--param-grid", required=True, help="JSON string or path mapping param -> list of values for algorithm __init__")
    parser.add_argument("--param-defaults", default="{}", help="JSON string or path mapping param -> default list (overridden by --param-grid)")
    parser.add_argument("--fixed-params", default="{}", help="JSON string or path for fixed algorithm params (applied to every run)")
    parser.add_argument("--env-params", default="{}", help="JSON string or path for environment constructor params")
    parser.add_argument("--iterations", type=int, default=10000, help="Training iterations per run")
    parser.add_argument("--ray-num-cpus", type=int, default=None, help="Number of CPUs to use with Ray (local)")
    parser.add_argument("--ray-address", type=str, default=None, help="Ray address for connecting to a cluster")
    parser.add_argument("--plot", action="store_true", help="Generate summary plots per varied parameter")
    parser.add_argument("--plot-metric", type=str, default="final_policy_avg_gamma_return", help="Metric key to plot")
    parser.add_argument("--runs-per-config", type=int, default=1, help="Number of runs per configuration")
    parser.add_argument("--seed-base", type=int, default=None, help="Base seed; seeds increment per run and config")
    parser.add_argument("--output-json", action="store_true", help="Output results as JSON instead of human-readable format")

    args = parser.parse_args()

    algo_cls_path: str = args.algo
    env_cls_path: str = args.env
    param_grid: Dict[str, Any] = _load_json_arg(args.param_grid)
    fixed_params: Dict[str, Any] = _load_json_arg(args.fixed_params)
    param_defaults: Dict[str, Any] = _load_json_arg(args.param_defaults)
    env_params: Dict[str, Any] = _load_json_arg(args.env_params)
    num_iterations: int = int(args.iterations)
    runs_per_config: int = int(args.runs_per_config)
    if args.seed_base is None:
        seed_base = int(time.time() * 1e6) % (2**31 - 1)
    else:
        seed_base = int(args.seed_base)

    # Prepare configs (one-parameter-at-a-time)
    # - param_defaults: dict of param -> list or scalar (baseline)
    # - param_grid: keys to vary; each key's values are tried while others stay at baseline
    def _as_list(v: Any) -> List[Any]:
        if isinstance(v, (list, tuple)):
            return list(v)
        return [v]
    def _first(v: Any) -> Any:
        return _as_list(v)[0]
    # Build baseline config
    base_cfg: Dict[str, Any] = dict(fixed_params)
    for k, v in (param_defaults or {}).items():
        base_cfg[k] = _first(v)

    configs: List[Dict[str, Any]] = []
    if param_grid:
        for k, values in (param_grid or {}).items():
            for v in _as_list(values):
                cfg = dict(base_cfg)
                cfg[k] = v
                configs.append(cfg)
    else:
        # If no grid specified, run the baseline once
        configs.append(base_cfg)

    # Run configs with Ray
    t0 = time.time()
    results_records: List[Dict[str, Any]] = []
    
    # Init Ray
    if args.ray_address:
        runtime_env = {"working_dir": _REPO_ROOT}
        ray.init(address=args.ray_address, num_cpus=args.ray_num_cpus, runtime_env=runtime_env)
    else:
        if args.ray_num_cpus is None:
            total_cpus = os.cpu_count() or 1
            use_cpus = total_cpus - 1
        else:
            use_cpus = args.ray_num_cpus
        ray.init(num_cpus=max(1, use_cpus))

    run_remote = ray.remote(num_cpus=1)(_run_single)
    refs = []
    ref_to_info: Dict[Any, Tuple[int, int, int, Dict[str, Any]]] = {}
    for ci, cfg in enumerate(configs):
        for ri in range(runs_per_config):
            seed = seed_base + ci * max(1, runs_per_config) + ri
            ref = run_remote.remote(
                env_cls_path=env_cls_path,
                algo_cls_path=algo_cls_path,
                env_kwargs=env_params,
                algo_kwargs=cfg,
                num_iterations=num_iterations,
                seed=seed,
                config_index=ci,
            )
            refs.append(ref)
            ref_to_info[ref] = (ci, ri, seed, cfg)
    
    # Collect all results
    results = ray.get(refs)
    for ref, res in zip(refs, results):
        ci, ri, seed, cfg = ref_to_info[ref]
        results_records.append({"config": cfg, "metrics": res.metrics, "config_index": ci, "run_index": ri, "seed": seed})
    ray.shutdown()

    duration = time.time() - t0
    total_runs = len(configs) * max(1, runs_per_config)
    
    if args.output_json:
        # Output results as JSON for programmatic use
        import json
        output_data = {
            "total_runs": total_runs,
            "duration_minutes": duration / 60.0,
            "results": results_records
        }
        print(json.dumps(output_data))
    else:
        print(f"\nCompleted {total_runs} runs in {duration/60.0:.2f} minutes")

    # Plot results if requested
    if args.plot and results_records:
        plot_results(results_records, args.plot_metric, t0)


if __name__ == "__main__":
    main()


