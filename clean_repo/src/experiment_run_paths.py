# Code and comments only in English.
"""
Parent directory for fresh training outputs (learning runs, grids, main_final).

Each script creates a timestamped subfolder under this root, e.g.:
  experiment_runs/grid_<date>/H01_baseline/ ...
  experiment_runs/learning_<date>/ ...
  experiment_runs/final_<date>/ ...

Legacy outputs may still live under ``experiments/`` (see .gitignore).
Change ``EXPERIMENT_RUNS_ROOT`` if you want a different folder name.
"""

EXPERIMENT_RUNS_ROOT = "experiment_runs"
