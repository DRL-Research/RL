# Summary Report: Multi-Seed Comparison of MAPS, VN-MA-DDPG, and MA-GA-DDPG

## 1. Executive Summary

This work added a full comparison pipeline to the project for three multi-agent reinforcement learning methods:

- `MAPS` (`ALGORITHM="experiment"`)
- `VN-MA-DDPG`
- `MA-GA-DDPG`

The main goal was to produce a fair, reproducible, multi-seed comparison with a final graph that shows:

- Success rate
- Collision rate
- Reward per episode
- Average travel time in successful episodes

Each method was trained with three random seeds, and the final graph aggregates the results using a moving average and variance shading. The output of this work is a repeatable experiment flow, standardized per-episode logging across all methods, and a final comparison figure that can be used for reporting and discussion.

The final full-run result shows that:

- `VN-MA-DDPG` performed best overall
- `MA-GA-DDPG` performed strongly and was competitive
- `MAPS` underperformed significantly in this environment

## 2. Purpose of the Work

Before this work, the project could run individual methods, but it did not have a unified mechanism to:

- train all three methods under the same setup,
- repeat training over multiple seeds,
- save comparable per-episode metrics for all methods,
- and generate one final graph with variance across seeds.

This assignment closed that gap by building a comparison-oriented workflow from training through final visualization.

## 3. Main Deliverables

The work delivered the following practical outputs:

- A unified multi-seed comparison runner
- Standardized progress CSV files for all compared methods
- A combined `2x2` comparison plot with variance shading
- Reproducible seed handling
- Fixes to MAPS training and episode outcome tracking

Final generated artifacts from the full run:

- Plot: `experiments/02_06_2026-14_54_29_FullAssignmentComparison_comparison/plots/multi_seed_algorithm_comparison.png`
- Summary file: `experiments/02_06_2026-14_54_29_FullAssignmentComparison_comparison/comparison_summary.json`
- Full run log: `full_assignment_comparison.log`

## 4. Code Changes

### 4.1 New Comparison Runner

File:

- `src/experiment/comparison_runner.py`

Purpose:

- Creates a fresh experiment configuration for each algorithm and each seed
- Runs the experiments automatically
- Collects the output CSV files
- Calls the plotting utility to create the final comparison graph
- Saves a summary JSON file describing all executed runs

This is the main orchestration layer for the assignment workflow.

### 4.2 New Comparison Plotting Utility

File:

- `src/plotting_utils/comparison_plotting.py`

Purpose:

- Reads progress CSV files from multiple runs
- Extracts comparable metrics
- Applies a trailing moving average
- Aggregates the runs across seeds
- Plots mean curves with shaded standard deviation

The final graph includes four subplots:

- Success Rate (%) - Moving Avg
- Collision Rate (%) - Moving Avg
- Reward per Episode - Moving Avg
- Avg Travel Time (Steps) - Moving Avg

### 4.3 New Shared Experiment Utilities

File:

- `src/training/experiment_utils.py`

Purpose:

- Sets random seeds consistently for Python, NumPy, and PyTorch
- Derives deterministic per-episode seeds from a run seed
- Writes standardized progress CSV files

This file was added to avoid duplicating seed and CSV logic across several algorithms.

### 4.4 Experiment Configuration Extensions

File:

- `src/experiment/experiment_config.py`

New fields:

- `SEED`
- `SHOW_PLOTS`

Purpose:

- `SEED` makes runs reproducible
- `SHOW_PLOTS` allows comparison mode to save plots without opening windows during long runs

### 4.5 MAPS Training and Logging Changes

Files:

- `src/training/episode_utils.py`
- `src/training/training_loop_utils.py`
- `src/training/training_handler.py`

Main changes:

- MAPS episodes now return structured per-episode results
- MAPS now tracks:
  - reward
  - success
  - collision
  - episode length
- MAPS writes `comparison_logs/progress.csv`
- MAPS clears global rollout buffers before repeated runs
- MAPS now participates fully in the same comparison flow as the baselines

This was necessary because MAPS previously did not expose its episode outcomes in a comparison-ready format.

### 4.6 Baseline Logging and Seeding Updates

Files:

- `src/baseline/vn_maddpg.py`
- `src/baseline/ma_ga_ddpg.py`

Main changes:

- Added multi-seed support
- Added writing to `comparison_logs/progress.csv`
- Preserved existing `baseline_logs/progress.csv`

The baseline algorithms already tracked the needed metrics, so these changes focused mainly on standardization and reproducibility.

### 4.7 Entry Point Improvement

File:

- `main.py`

Main change:

- Added `run_mode = "single" | "comparison"`

Purpose:

- `single` keeps normal behavior
- `comparison` launches the multi-seed comparison flow

## 5. Important Bug Fixes

### 5.1 MAPS Master Training Shape Fix

While testing the comparison flow, MAPS exposed an existing bug in its master PPO update logic.

Problem:

- The master rollout buffer returned tensors with an extra environment dimension
- PPO evaluation produced incompatible tensor shapes for:
  - values
  - log probabilities
  - advantages

Observed error:

- `The size of tensor a (4) must match the size of tensor b (...)`

Cause:

- The master action is a 4-dimensional continuous embedding
- The PPO loss path expected one scalar log-probability per timestep
- The current manual update path produced per-dimension log-probabilities instead

Fix:

- Removed the extra environment dimension before calling `evaluate_actions()`
- Reshaped values, log probabilities, returns, and advantages consistently
- Ensured the master PPO loss was computed on aligned tensors

Impact:

- MAPS master updates now run instead of silently failing
- The final MAPS curve in the full comparison is based on actual training

### 5.2 MAPS Collision Detection Fix

Problem:

- Some MAPS episodes with collisions were being labeled as successful

Cause:

- Episode outcome logic relied too much on `info["crashed"]`
- That signal was not always sufficient for correct final episode labeling

Fix:

- Added a direct check of controlled vehicle crash flags in the wrapped environment

Impact:

- Success and collision metrics are now trustworthy for MAPS
- The comparison plot reflects correct outcome statistics

## 6. Workflow of the New System

The new workflow is:

1. Start from one base experiment configuration
2. Select the compared algorithms
3. Select the seed list
4. For each algorithm-seed pair:
   - create an isolated experiment config
   - disable rendering
   - train the method
   - write a standardized progress CSV
5. After all runs finish:
   - load all CSV files
   - smooth each run with a moving average
   - aggregate across seeds
   - create the final comparison plot
   - save a JSON summary of the experiment set

This makes the comparison reusable and easy to extend later.

## 7. Experimental Setup

Full comparison configuration:

- Algorithms:
  - `MAPS`
  - `VN-MA-DDPG`
  - `MA-GA-DDPG`
- Seeds:
  - `11`
  - `22`
  - `33`
- Episodes per cycle:
  - `300`
- Cycles:
  - `3`
- Total episodes per run:
  - `900`
- Total runs:
  - `9`
- Moving average window:
  - `50`

This produced a long-horizon comparison with visible variance across seeds.

## 8. Results

### 8.1 Overall Ranking

Based on the final full run:

1. `VN-MA-DDPG`
2. `MA-GA-DDPG`
3. `MAPS`

### 8.2 VN-MA-DDPG

Observed behavior:

- Success rate quickly rose to nearly `100%`
- Collision rate dropped to almost `0%`
- Reward stayed clearly positive
- Average travel time was the shortest, around `7-8` steps

Interpretation:

- This method was the strongest and most stable in the tested setup

### 8.3 MA-GA-DDPG

Observed behavior:

- Success rate stabilized around `90-95%`
- Collision rate stayed low
- Reward was generally good, though weaker than `VN-MA-DDPG`
- Average travel time was also short, around `8` steps

Interpretation:

- This method performed well and remained competitive
- It was slightly weaker and slightly more variable than `VN-MA-DDPG`

### 8.4 MAPS

Observed behavior:

- Early training showed some temporary improvement
- Later training degraded significantly
- Success rate dropped to roughly `10-15%`
- Collision rate rose to roughly `85-90%`
- Reward remained strongly negative
- Successful travel time remained much higher than the other methods

Interpretation:

- In the current implementation and environment, MAPS does not compete well with the two MADDPG-based methods

## 9. Interpretation of the Final Plot

The final graph should be read as follows:

- The solid line is the mean performance across three seeds
- The shaded region is the standard deviation across the three runs

This allows evaluation of:

- average performance,
- training trend,
- and stability between runs

The variance is much more meaningful in the full run than in the short demo run because the final experiment used enough episodes for the moving average to smooth the curves properly.

## 10. Reproducibility and Reuse

One of the main benefits of this work is that the comparison process is now reusable.

It is now easy to:

- add more seeds,
- compare additional algorithms,
- rerun the same setup with identical seeds,
- or regenerate the final figure from saved CSV files.

This makes the code useful not only for the current assignment, but also for future research comparisons in the same project.

## 11. Limitations

The following points are worth noting:

- Results are specific to the current environment and reward setup
- MAPS now runs correctly, but still performs poorly in this configuration
- The current comparison uses standard deviation shading, which is informative but can still look wide if instability is high
- The final ranking reflects this implementation and not necessarily the theoretical potential of each algorithm under different tuning

## 12. Recommended Next Steps

Recommended follow-up work:

- review MAPS architecture and reward interaction to understand its performance collapse,
- tune hyperparameters for MAPS more systematically,
- optionally compare confidence intervals instead of standard deviation in the plot,
- and repeat the study on additional traffic scenarios to test generalization.

## 13. Conclusion

This work successfully added a complete multi-seed comparison workflow to the project. It standardized logging across methods, fixed important MAPS training and evaluation issues, and produced a full comparison graph suitable for reporting. The final results show that `VN-MA-DDPG` is the best-performing method in the tested setup, `MA-GA-DDPG` is also strong, and `MAPS` currently lags behind both in stability and final performance.

From a project perspective, the most important outcome is not only the final ranking, but also the fact that the repository now includes a reliable framework for running future comparative experiments in a consistent and reproducible way.
