## Execution Flow

```text
main.py
│
└── run_experiment()
    │
    ├── MasterModel
    │   └── Generates latent embeddings (master_model.py)
    │
    ├── Agent (Wrapper)
    │   ├── Wraps environment and injects master embedding
    │   └── Uses:
    │       └── gym.make('RELintersection-v0')
    │           └── IntersectionEnv (intersection_class.py)
    │               ├── CustomControlledVehicle (CustomControlledVehicle.py)
    │               └── action_factory (custom_action.py)
    │
    ├── Model (Agent PPO)
    │   └── PPO agent initialized with experiment config (model_handler.py)
    │
    ├── Training Loop (training_loop.py)
    │   ├── Runs episodes (run_episode)
    │   │   ├── MasterModel → embedding
    │   │   └── AgentModel → state + embedding → action
    │   └── Updates buffers + trains agent/master alternately
    │
    ├── Logging & Saving
    │   └── Saves models, logs to CSV/TensorBoard
    │
    └── Plotting
        └── PlottingUtils → generates reward/loss graphs

