import sys
import os
import time
import argparse
import json
import numpy as np
import multiprocessing

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def run_single_experiment_process(alg_name, alg_key, seed, num_episodes, experiment_name, render_mode=None, env_id="RELintersection-v0"):
    """
    Subprocess entrypoint that runs a single algorithm and seed configuration.
    All imports are local to ensure compatibility with Windows process spawning.
    """
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

    import random
    import numpy as np
    import torch

    from highwayenv.utils import patch_intersection_env, register_intersection_env, register_roundabout_env, register_double_intersection_env
    from src.experiment import scenarios_config as sc
    from src.experiment.experiment_config import Experiment
    from src.training.training_handler import run_experiment

    # Set up console log redirection to prevent mixed console output
    log_dir = os.path.join(experiment_name, "process_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, f"{alg_name}_S{seed}.log")

    sys.stdout = open(log_file_path, "w", encoding="utf-8")
    sys.stderr = sys.stdout

    print(f"[{alg_name} | Seed {seed}] Initializing environment patches...")
    patch_intersection_env()
    register_intersection_env()
    register_roundabout_env()
    register_double_intersection_env()

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    config = Experiment(
        ALGORITHM=alg_key,
        RENDER_MODE=render_mode,
        ENV_ID=env_id,
        EXPERIMENT_ID=f"Compare_{alg_name}_S{seed}",
        CYCLES=1,
        EPISODES_PER_CYCLE=num_episodes
    )
    config.SEED = seed
    config.LOAD_PREVIOUS_WEIGHT = False
    config.EXPERIMENT_PATH = os.path.join(experiment_name, f"Compare_{alg_name}_S{seed}")
    config.SAVE_MODEL_DIRECTORY = f"{config.EXPERIMENT_PATH}/trained_model"

    env_config = sc.full_env_config_exp5

    print(f"[{alg_name} | Seed {seed}] Starting training run...")
    start_time = time.time()
    try:
        _, history, _ = run_experiment(config, env_config)

        # Save history to a JSON file (extremely safe and robust cross-process)
        histories_dir = os.path.join(experiment_name, "histories")
        os.makedirs(histories_dir, exist_ok=True)
        history_path = os.path.join(histories_dir, f"{alg_name}_S{seed}.json")

        # Convert numpy numbers to standard python types for JSON serialization
        serializable_history = {}
        for k, v in history.items():
            if isinstance(v, list):
                serializable_history[k] = [float(x) if isinstance(x, (np.floating, float)) else x for x in v]
            else:
                serializable_history[k] = v

        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(serializable_history, f, indent=4)

        elapsed = time.time() - start_time
        print(f"[{alg_name} | Seed {seed}] Finished successfully in {elapsed:.2f} seconds!")
        print(f"[{alg_name} | Seed {seed}] History saved to {history_path}")
    except Exception as e:
        print(f"[{alg_name} | Seed {seed}] FAILED with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.close()


def main():
    parser = argparse.ArgumentParser(description="Parallelized Multi-Seed MARL Comparison")
    parser.add_argument("--episodes", type=int, default=900 * 3,
                        help="Number of episodes per seed")
    parser.add_argument("--seeds", type=str, default="42,100,2026", help="Comma-separated seeds")
    parser.add_argument("--window", type=int, default=10, help="Rolling average window size")
    parser.add_argument("--render-mode", type=str, default="rgb_array", choices=["human", "rgb_array", "none", "None"],
                        help="Gymnasium render mode")
    parser.add_argument("--experiment-name", type=str, default="experiment_default_name",
                        help="The name of experiment for logging and results dir")
    parser.add_argument("--env", type=str, default="RELintersection-v0", help="Environment ID to run")
    args = parser.parse_args()

    experiment_name = os.path.join("experiments_results", args.experiment_name)
    seeds = [int(s) for s in args.seeds.split(",")]
    num_episodes = args.episodes
    render_mode = args.render_mode
    env_id = args.env
    if render_mode in ["none", "None", ""]:
        render_mode = None

    print("==================================================")
    print("      Parallel Cooperative MARL Experiment        ")
    print(f"      Seeds: {seeds} | Episodes: {num_episodes}   ")
    print(f"      Experiment Directory: {experiment_name}     ")
    print(f"      Environment: {env_id}                       ")
    print("==================================================")

    algorithms = {
        "MAPS": "experiment",
        "VN-MA-DDPG": "vn_maddpg",
        "MA-GA-DDPG": "ma_ga_ddpg",
        "IPPO": "ippo",
        "COMA": "coma",
        "VDN": "vdn",
        "Social-Attention": "social_attention"
    }

    # Ensure histories and logs directory is clean/setup
    os.makedirs(os.path.join(experiment_name, "histories"), exist_ok=True)
    os.makedirs(os.path.join(experiment_name, "process_logs"), exist_ok=True)

    # Spawn a pool of processes to avoid memory exhaustion
    max_concurrent = 4
    print(f"\nInitializing process pool with {max_concurrent} concurrent workers...")
    pool = multiprocessing.Pool(processes=max_concurrent)

    async_results = []

    # Queue a task for each seed and algorithm only if not already completed
    for alg_name, alg_key in algorithms.items():
        for seed in seeds:
            history_path = os.path.join(experiment_name, "histories", f"{alg_name}_S{seed}.json")
            is_complete = False
            if os.path.exists(history_path):
                try:
                    with open(history_path, "r", encoding="utf-8") as f:
                        hist = json.load(f)
                    if len(hist.get("episode_rewards", [])) >= num_episodes:
                        is_complete = True
                except Exception:
                    pass

            if is_complete:
                print(
                    f"Skipping {alg_name} (Seed {seed}) - full training run of {num_episodes} episodes already completed.")
                continue

            print(f"Queueing process for {alg_name} (Seed {seed}) in pool...")
            res = pool.apply_async(
                run_single_experiment_process,
                args=(alg_name, alg_key, seed, num_episodes, experiment_name, render_mode, env_id)
            )
            async_results.append((alg_name, seed, res))

    if async_results:
        print(f"\nSuccessfully queued {len(async_results)} training tasks in the process pool!")
        print(f"All outputs are being redirected to logs in {experiment_name}/process_logs/.")
        print("Waiting for runs to complete... (This will run in sequence of max 4 concurrent runs)")

        # Wait for all pool tasks to complete
        try:
            for idx, (alg_name, seed, res) in enumerate(async_results):
                res.get()  # Blocks until this task is finished.
                print(f"  Task {idx + 1}/{len(async_results)} finished: {alg_name} (Seed {seed}).")
        except KeyboardInterrupt:
            print("\nTermination requested. Terminating process pool...")
            pool.terminate()
            pool.join()
            print("All processes terminated successfully.")
            return
        else:
            pool.close()
            pool.join()
    else:
        print(
            f"\nAll requested experiments in {experiment_name} are already completed! Proceeding to aggregate directly...")
        pool.close()
        pool.join()

    print(f"\nAll processes finished! Results saved to {experiment_name}/histories/.")
    print("To analyze the results and view plots, open and run the Jupyter Notebook: analyze_results.ipynb")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()