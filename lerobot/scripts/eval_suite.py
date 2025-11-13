"""Evaluate a policy on a full suite of LIBERO tasks by running rollouts and computing metrics.

Usage examples:

# One task (no gc)
python lerobot/scripts/eval_suite.py \
    --policy.path=outputs/train/diffPo_libero_v3/checkpoints/last/pretrained_model \
    --env.type=libero \
    --env.suite_name=libero_object \
    --task_ids=0 \
    --eval.batch_size=10
"""

import json
import logging
import time
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from termcolor import colored
from tqdm import tqdm

from lerobot.common.envs.factory import make_env
from lerobot.common.policies.factory import make_policy
from lerobot.common.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.scripts.eval import eval_policy

from libero.libero import benchmark


@dataclass
class EvalSuiteConfig(EvalPipelineConfig):
    """Configuration for evaluating on a suite of LIBERO tasks."""
    suite_name: str = "libero_object"  # Suite name or "all" for all suites (separate from env's task_suite_name)
    task_ids: Optional[str] = None  # Comma-separated task IDs, or None for all tasks in suite
    save_individual_results: bool = True  # Save results for each task separately
    aggregate_results: bool = True  # Compute and save aggregated results across tasks


def get_available_suites() -> List[str]:
    """Get list of available LIBERO task suites."""
    benchmark_dict = benchmark.get_benchmark_dict()
    # Filter out libero_100
    available_suites = [suite for suite in benchmark_dict.keys() if suite != "libero_100"]
    return available_suites


def get_suite_tasks(task_suite_name: str) -> List[Dict[str, Any]]:
    """Get task information for a given suite."""
    benchmark_dict = benchmark.get_benchmark_dict()
    if task_suite_name not in benchmark_dict:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    
    task_suite = benchmark_dict[task_suite_name]()
    tasks = []
    for task_id, task in enumerate(task_suite.tasks):
        tasks.append({
            "task_id": task_id,
            "name": task.name,
            "description": task.language,
            "suite": task_suite_name
        })
    return tasks


def eval_suite(
    cfg: EvalSuiteConfig,
    policy_path: str,
    output_dir: Path,
    device: torch.device
) -> Dict[str, Any]:
    """Evaluate a policy on a suite of tasks."""
    
    # Determine which suites and tasks to evaluate
    if cfg.suite_name == "all":
        suites_to_eval = get_available_suites()
    else:
        suites_to_eval = [cfg.suite_name]

    # Parse task IDs if provided
    task_ids_filter = None
    if cfg.task_ids is not None:
        task_ids_filter = [int(tid.strip()) for tid in cfg.task_ids.split(",")]
    
    # Collect all tasks to evaluate
    all_tasks = []
    for suite_name in suites_to_eval:
        suite_tasks = get_suite_tasks(suite_name)
        if task_ids_filter is not None:
            suite_tasks = [task for task in suite_tasks if task["task_id"] in task_ids_filter]
        all_tasks.extend(suite_tasks)
    
    logging.info(f"Evaluating on {len(all_tasks)} tasks across {len(suites_to_eval)} suites")
    
    # Results storage
    all_results = []
    suite_results = {}
    
    # Evaluate each task
    for task_info in tqdm(all_tasks, desc="Evaluating tasks"):
        task_suite_name = task_info["suite"]
        task_id = task_info["task_id"]
        task_name = task_info["name"]
        
        logging.info(f"Evaluating task: {task_suite_name}_{task_id} - {task_name}")
        
        # Create environment configuration for this specific task
        env_cfg = deepcopy(cfg.env)
        env_cfg.task = f"{task_suite_name}_{task_id}"
        env_cfg.task_suite_name = task_suite_name
        env_cfg.task_id = task_id
        
        # Create environment
        env = make_env(env_cfg, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)
        
        # Create policy (reuse same policy for all tasks)
        if task_info == all_tasks[0]:  # Create policy only once
            # Use the original env_cfg from the loaded policy config, not the current task env_cfg
            policy = make_policy(cfg=cfg.policy, env_cfg=cfg.env)
            policy.eval()
        
        # Evaluate on this task
        start_time = time.time()
        with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
            task_info_result = eval_policy(
                env,
                policy,
                cfg.eval.n_episodes,
                max_episodes_rendered=50,
                videos_dir=output_dir / "videos" / f"{task_suite_name}_{task_id}",
                start_seed=cfg.seed,
            )
        
        eval_time = time.time() - start_time
        
        # Add task metadata to results
        task_result = {
            "task_info": task_info,
            "eval_time_s": eval_time,
            **task_info_result
        }
        
        all_results.append(task_result)
        
        # Group by suite for aggregation
        if task_suite_name not in suite_results:
            suite_results[task_suite_name] = []
        suite_results[task_suite_name].append(task_result)
        
        # Save individual task result if requested
        if cfg.save_individual_results:
            task_output_dir = output_dir / "individual_tasks"
            task_output_dir.mkdir(parents=True, exist_ok=True)
            with open(task_output_dir / f"{task_suite_name}_{task_id}_result.json", "w") as f:
                json.dump(task_result, f, indent=2)
        
        # Log current task results
        agg_result = task_result["aggregated"]
        logging.info(
            f"Task {task_suite_name}_{task_id}: "
            f"Success rate: {agg_result['pc_success']:.1f}%, "
            f"Avg reward: {agg_result['avg_sum_reward']:.3f}, "
            f"Time: {eval_time:.1f}s"
        )
        
        env.close()
    
    # Compute aggregated results
    aggregated_results = {}
    
    if cfg.aggregate_results:
        # Overall aggregation across all tasks
        all_success_rates = [task["aggregated"]["pc_success"] for task in all_results]
        all_avg_rewards = [task["aggregated"]["avg_sum_reward"] for task in all_results]
        all_eval_times = [task["eval_time_s"] for task in all_results]
        
        aggregated_results["overall"] = {
            "num_tasks": len(all_results),
            "num_suites": len(suites_to_eval),
            "avg_success_rate": np.mean(all_success_rates),
            "std_success_rate": np.std(all_success_rates),
            "min_success_rate": np.min(all_success_rates),
            "max_success_rate": np.max(all_success_rates),
            "avg_reward": np.mean(all_avg_rewards),
            "std_reward": np.std(all_avg_rewards),
            "total_eval_time_s": np.sum(all_eval_times),
            "avg_eval_time_per_task_s": np.mean(all_eval_times),
            "suites_evaluated": suites_to_eval
        }
        
        # Per-suite aggregation
        for suite_name, suite_tasks in suite_results.items():
            suite_success_rates = [task["aggregated"]["pc_success"] for task in suite_tasks]
            suite_avg_rewards = [task["aggregated"]["avg_sum_reward"] for task in suite_tasks]
            suite_eval_times = [task["eval_time_s"] for task in suite_tasks]
            
            aggregated_results[f"suite_{suite_name}"] = {
                "suite_name": suite_name,
                "num_tasks": len(suite_tasks),
                "avg_success_rate": np.mean(suite_success_rates),
                "std_success_rate": np.std(suite_success_rates),
                "min_success_rate": np.min(suite_success_rates),
                "max_success_rate": np.max(suite_success_rates),
                "avg_reward": np.mean(suite_avg_rewards),
                "std_reward": np.std(suite_avg_rewards),
                "total_eval_time_s": np.sum(suite_eval_times),
                "task_success_rates": suite_success_rates,
                "task_names": [task["task_info"]["name"] for task in suite_tasks]
            }
    
    # Final results structure
    final_results = {
        "config": asdict(cfg),
        "policy_path": str(policy_path),
        "evaluation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "individual_tasks": all_results,
        "aggregated": aggregated_results
    }
    
    return final_results


@parser.wrap()
def eval_suite_main(cfg: EvalSuiteConfig):
    # Set default env type to libero if not specified
    if cfg.env is None or not hasattr(cfg.env, 'type') or cfg.env.type is None:
        from lerobot.common.envs.configs import LiberoEnv
        cfg.env = LiberoEnv()
        cfg.env.type = "libero"
    
    # Fix environment features to match what the policy expects
    if hasattr(cfg.policy, 'input_features') and cfg.policy.input_features:
        from lerobot.configs.types import FeatureType, PolicyFeature
        
        # Create proper features dict with individual cameras
        features = {}
        features_map = {}
        
        for key, policy_feature in cfg.policy.input_features.items():
            if key == 'observation.images.cam_libero.color':
                features['agentview'] = policy_feature
                features_map['agentview'] = key
            elif key == 'observation.images.cam_libero.wrist':
                features['wristview'] = policy_feature  
                features_map['wristview'] = key
            elif key == 'observation.images.cam_libero.transformed_depth':
                features['agentview_depth'] = policy_feature
                features_map['agentview_depth'] = key
            elif key == 'observation.images.cam_libero.goal_gripper_proj':
                features['agentview_goal_gripper_proj'] = policy_feature
                features_map['agentview_goal_gripper_proj'] = key
            elif key == 'observation.state':
                features['agent_pos'] = policy_feature
                features_map['agent_pos'] = key
        
        # Add action from output features
        if hasattr(cfg.policy, 'output_features') and 'action' in cfg.policy.output_features:
            features['action'] = cfg.policy.output_features['action']
            features_map['action'] = 'action'
        
        cfg.env.features = features
        cfg.env.features_map = features_map
    logging.info("LIBERO Evaluation Suite")
    logging.info("Policy config loaded:")
    logging.info(f"Policy type: {cfg.policy.type}")
    logging.info(f"Policy input features: {cfg.policy.input_features.keys()}")
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = False
    
    # Set seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    
    # Create output directory
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check LIBERO availability
    if benchmark is None:
        raise ImportError("LIBERO benchmark not available. Please install libero.")

    # Log available suites
    available_suites = get_available_suites()
    logging.info(f"Available LIBERO suites: {available_suites}")
    
    if cfg.suite_name != "all" and cfg.suite_name not in available_suites:
        raise ValueError(f"Unknown task suite: {cfg.suite_name}. Available: {available_suites}")

    # Run evaluation  
    policy_path = getattr(cfg.policy, 'pretrained_path', cfg.policy.path if hasattr(cfg.policy, 'path') else 'unknown')
    results = eval_suite(cfg, policy_path, output_dir, device)

    # Print summary
    if cfg.aggregate_results and "overall" in results["aggregated"]:
        overall = results["aggregated"]["overall"]
        print("\n" + "="*60)
        print("LIBERO EVALUATION SUITE RESULTS")
        print("="*60)
        print(f"Policy: {policy_path}")
        print(f"Suites evaluated: {', '.join(overall['suites_evaluated'])}")
        print(f"Total tasks: {overall['num_tasks']}")
        print(f"Episodes per task: {cfg.eval.n_episodes}")
        print(f"Overall success rate: {overall['avg_success_rate']:.1f}% ± {overall['std_success_rate']:.1f}%")
        print(f"Success rate range: {overall['min_success_rate']:.1f}% - {overall['max_success_rate']:.1f}%")
        print(f"Average reward: {overall['avg_reward']:.3f} ± {overall['std_reward']:.3f}")
        print(f"Total evaluation time: {overall['total_eval_time_s']:.1f}s ({overall['total_eval_time_s']/60:.1f}m)")
        print()
        
        # Print per-suite results
        for suite_name in overall['suites_evaluated']:
            suite_key = f"suite_{suite_name}"
            if suite_key in results["aggregated"]:
                suite_result = results["aggregated"][suite_key]
                print(f"{suite_name}: {suite_result['avg_success_rate']:.1f}% ± {suite_result['std_success_rate']:.1f}% "
                      f"({suite_result['num_tasks']} tasks)")
        
        print("="*60)

    # Save results (convert Path objects to strings for JSON serialization)
    def convert_paths_to_strings(obj):
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {k: convert_paths_to_strings(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_paths_to_strings(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_paths_to_strings(results)
    
    results_file = output_dir / "eval_suite_results.json"
    with open(results_file, "w") as f:
        json.dump(results_serializable, f, indent=2)
    
    logging.info(f"Results saved to: {results_file}")
    logging.info("End of LIBERO evaluation suite")


if __name__ == "__main__":
    init_logging()
    eval_suite_main()
