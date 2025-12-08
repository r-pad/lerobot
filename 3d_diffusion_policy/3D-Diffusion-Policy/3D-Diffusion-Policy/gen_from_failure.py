import os
import sys
import traceback
import argparse
import yaml
import json
import random
from tqdm import tqdm
import time
import datetime
import multiprocessing as mp
import numpy as np
from manipulation.utils import build_up_env, load_env, save_numpy_as_gif
import pybullet as p

def _check_failure(
    q: mp.Queue,
    exp_path: str,
    angle_threshold: float = 0.8,
    env_name: str = 'articulated'
):  
    config_file = os.path.join(exp_path, 'task_config.yaml')
    states_path = os.path.join(exp_path, 'states')
    all_states = [f for f in os.listdir(states_path) if f.startswith('state_') and f.endswith('.pkl')]
    if len(all_states) == 0:
        q.put(-1)
        return -1
    all_grasped_handles = [False for _ in range(2)]
    current_grasped_handle = False
    last_joint_angle = None
    # Load the initial state to get the initial joint angle
    init_state_path = os.path.join(states_path, 'state_0.pkl')
    env, _ = build_up_env(
        task_config=config_file,
        env_name=env_name,
        restore_state_file=init_state_path,
        render=False, 
        randomize=False,
    )
    env.reset()
    info = env._get_info()
    initial_joint_angle = info['initial_joint_angle']
    # print(f"Initial joint angle: {initial_joint_angle}")
    env.close()
    for state_idx in range(len(all_states)):
        state_path = os.path.join(states_path, f'state_{state_idx}.pkl')
        if not os.path.exists(state_path):
            raise ValueError(f"State file {state_path} does not exist.")
        env, _ = build_up_env( 
            task_config=config_file,
            env_name=env_name,
            restore_state_file=state_path,
            render=False, 
            randomize=False,
        )
        env.reset()
        info = env._get_info()
        env.close()
        joint_angle = info['opened_joint_angle']
        # initial_joint_angle = info['initial_joint_angle']
        # print(f"Opened joint angle at state {state_idx}: {joint_angle - initial_joint_angle}")
        if joint_angle - initial_joint_angle > angle_threshold:
            q.put(-1)
            return -1  # If the joint angle exceeds the threshold, success
        if last_joint_angle is not None:
            grasped_handle = abs(joint_angle - last_joint_angle) > 1e-6
        else:
            grasped_handle = False
        last_joint_angle = joint_angle
        all_grasped_handles.append(grasped_handle)
        if current_grasped_handle:
            if not any(all_grasped_handles[-1:]):
                # If the current state is not grasped, consider it a failure
                print(f"Failure detected at state {state_idx} in {exp_path}.")
                q.put(state_idx)
                return state_idx
        else:
            if all(all_grasped_handles[-2:]):
                # If the last 2 states are grasped, consider the current state successful
                current_grasped_handle = True
    q.put(-1)  # If all states are successful, return -1
    return -1  # If all states are successful, return -1

def check_failure(
    exp_path: str,
    angle_threshold: float = 0.8,
    env_name: str = 'articulated'
):
    """
    Check if the experiment has failed based on the joint angles.
    Returns the index of the first failure state, or -1 if no failure is detected.
    """
    q = mp.Queue()
    p = mp.Process(target=_check_failure, args=(q, exp_path, angle_threshold, env_name))
    p.start()
    p.join()
    failure_idx = q.get()
    return failure_idx


def get_failure_exps(
    all_exps_path: str,
):
    all_exps = [d for d in os.listdir(all_exps_path) if os.path.isdir(os.path.join(all_exps_path, d))]
    # all_exps = all_exps[:2]  # Limit to the first 100 experiments for performance
    json_path = os.path.join(all_exps_path, "opened_joint_angles.json")
    with open(json_path, 'r') as f:
        data = json.load(f)
    expert_opened_joint_angles = []
    for entry in data.values():
        expert_door_joint_angle = entry["expert_door_joint_angle"]
        initial_joint_angle = entry["initial_joint_angle"]
        expert_opened_joint_angles.append(expert_door_joint_angle - initial_joint_angle)
    avg_expert_opened_joint_angle = sum(expert_opened_joint_angles) / len(expert_opened_joint_angles)
    threshold = 0.7 * avg_expert_opened_joint_angle
    # print(f"Using threshold {threshold} for failure detection based on expert opened joint angles.")

    # all_exps = ['2025-05-29-15-31-26_rollout']
    failure_exps = []
    failure_idxs = []
    with tqdm(total=len(all_exps), desc="Checking experiments for failures") as pbar:
        for exp in all_exps:
            exp_path = os.path.join(all_exps_path, exp)
            failure_idx = check_failure(exp_path, angle_threshold=threshold)
            if failure_idx != -1:
                failure_exps.append(exp_path)
                failure_idxs.append(failure_idx)
            pbar.update(1)
                
    return failure_exps, failure_idxs

def get_all_init_states(
    all_exps,
    start_idxs,
):
    all_configs = []
    all_states = []
    for exp, start_idx in zip(all_exps, start_idxs):
        config_file = os.path.join(exp, 'task_config.yaml')
        states_path = os.path.join(exp, 'states')
        num_states = len([f for f in os.listdir(states_path) if f.startswith('state_') and f.endswith('.pkl')])
        if start_idx < 0 or start_idx >= num_states:
            raise ValueError(f"Start index {start_idx} is out of bounds for experiment {exp}.")
        all_configs += [config_file for _ in range(num_states - start_idx)]
        all_states += [os.path.join(states_path, f'state_{i}.pkl') for i in range(start_idx, num_states)]
    return all_configs, all_states

# def execute(
#     # q: mp.Queue,
#     exp_path: str,
#     config_path: str,
#     state_path: str,
#     env_name: str = 'articulated',
# ):  
#     print("Starting _execute...")
#     env, _ = build_up_env(config_path, env_name, restore_state_file=state_path)
#     env.primitive_save_path = exp_path

#     np.random.seed(time.time_ns() % 2**32)
#     env.reset()
#     rgb = env.render()
#     import imageio
#     imageio.imwrite("first_frame.png", rgb)

#     print("Executing environment...")

#     rgbs, states = env.execute()
#     env.close()

#     if len(states) > 10:
#         with open(os.path.join(exp_path, "last_state_files.txt"), 'w') as f:
#             f.write("\n".join(str(states[-1])))

#         save_numpy_as_gif(np.array(rgbs), os.path.join(exp_path, "all.gif"))
#         print("Execution succeeded.")
#         # q.put(True)
#         return True
#     else:
#         print("Execution failed: too few states.")
#         # q.put(False)
#         return False


def _execute(
    q: mp.Queue,
    exp_path: str,
    config_path: str,
    state_path: str,
    env_name: str = 'articulated',
):  
    # mkdir for the experiment path if it does not exist
    os.makedirs(exp_path, exist_ok=True)

    # redirect stdout and stderr to a log file
    log_path = os.path.join(exp_path, 'execution_log.txt')
    sys.stdout = open(log_path, 'w')
    sys.stderr = sys.stdout

    try:
        print("Starting _execute...")
        env, _ = build_up_env(config_path, env_name, restore_state_file=state_path)
        env.primitive_save_path = exp_path

        np.random.seed(time.time_ns() % 2**32)
        env.reset()

        print("Executing environment...")

        rgbs, states = env.execute()
        env.close()

        if len(states) > 10:
            with open(os.path.join(exp_path, "last_state_files.txt"), 'w') as f:
                f.write("\n".join(str(states[-1])))

            save_numpy_as_gif(np.array(rgbs), os.path.join(exp_path, "all.gif"))
            print("Execution succeeded.")
            q.put(True)
        else:
            print("Execution failed: too few states.")
            q.put(False)

    except Exception:
        print("Exception occurred:")
        traceback.print_exc()
        q.put(False)

    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        sys.stdout.close()

def execute(
    exp_path: str,
    config_path: str,
    state_path: str,
    env_name: str = 'articulated',
):
    q = mp.Queue()
    p = mp.Process(target=_execute, args=(q, exp_path, config_path, state_path, env_name))
    p.start()
    p.join()
    success = q.get()
    return success

def gen_from_failure(
    all_exps_path: str,
    output_path: str,
    num_exps: int = 50,
    resume: bool = True,
):
    if not os.path.exists(all_exps_path):
        raise ValueError(f"Path {all_exps_path} does not exist.")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    if os.path.exists(os.path.join(output_path, "failure_exps.txt")) and resume:
        print("Failure experiments already exist in the output directory. Skipping failure detection.")
        with open(os.path.join(output_path, "failure_exps.txt"), 'r') as f:
            failure_exps = []
            failure_idxs = []
            for line in f:
                exp, idx = line.strip().split()
                failure_exps.append(exp)
                failure_idxs.append(int(idx))
    else:
        print("Detecting failure experiments...")
        failure_exps, failure_idxs = get_failure_exps(all_exps_path)
        if len(failure_exps) == 0:
            print("No failure experiments found.")
            return
        print(f"Found {len(failure_exps)} failure experiments.")
        # save the failure experiments and their indices
        with open(os.path.join(output_path, "failure_exps.txt"), 'w') as f:
            for exp, idx in zip(failure_exps, failure_idxs):
                f.write(f"{exp} {idx}\n")
    all_configs, all_states = get_all_init_states(failure_exps, failure_idxs)
    if len(all_configs) == 0:
        print("No initial states found for the failure experiments.")
        return

    cnt = 0
    while cnt < num_exps or len(all_configs) > 0:
        ts = time.time()
        time_string = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d-%H-%M-%S')
        exp_path = os.path.join(output_path, time_string)
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
        # randomly select a config and state
        idx = random.randint(0, len(all_configs) - 1)
        config = all_configs.pop(idx)
        state = all_states.pop(idx)
        os.system("cp {} {}".format(config, os.path.join(exp_path, "task_config.yaml")))
        print(f"Executing experiment with config {config} and state {state}...")
        success = execute(exp_path, config, state)
        if not success:
            print(f"Execution failed for config {config} and state {state}.")
        else:
            print(f"Experiment saved to {exp_path}.")
            cnt += 1
        if len(all_configs) == 0:
            print("No more initial states to process.")
            break
        elif cnt >= num_exps:
            print(f"Generated {cnt} experiments. Stopping.")
            break

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_dir', type=str, default=None)
    parser.add_argument('--data_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--category', type=str, default=None)
    parser.add_argument('--obj_id', type=str, default=None)
    args = parser.parse_args()

    if args.exp_dir is not None:
        exp_dir = args.exp_dir
        category = exp_dir.split('/')[-2]
        obj_id = exp_dir.split('/')[-1]
    else:
        category = args.category
        obj_id = args.obj_id
    
    all_exps_path = os.path.join(args.data_dir, category, obj_id)
    if not os.path.exists(all_exps_path):
        raise ValueError(f"Path {all_exps_path} does not exist.")
    output_path = os.path.join(args.output_dir, category, obj_id)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    gen_from_failure(
        all_exps_path=all_exps_path,
        output_path=output_path
    )
