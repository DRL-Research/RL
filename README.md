# Proto Plan Embeddings for Hierarchical Multi Agent Coordination at Unsignalized Intersections

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![Gym 0.21.0](https://img.shields.io/badge/gym-0.21.0-orange.svg)](https://github.com/openai/gym)
[![Stable-Baselines3 1.6.2](https://img.shields.io/badge/stable--baselines3-1.6.2-blueviolet.svg)](https://github.com/DLR-RM/stable-baselines3)
[![PyTorch 2.3.1](https://img.shields.io/badge/pytorch-2.3.1-red.svg)](https://pytorch.org/)

A state-of-the-art cooperative Multi-Agent Reinforcement Learning (MARL) framework designed for autonomous intersection coordination. The architecture addresses the challenge of exponential action-space growth in multi-vehicle coordination by utilizing a **Dual-Loop Latent Master model**. 

In this system, a centralized **Master Model** observes the overall environment and projects global coordination guidelines into a low-dimensional continuous **latent embedding**. Local vehicle controllers (**Agent PPOs**) receive their individual state augmented with this master embedding to perform collision-free, high-throughput cooperative navigation.

---

## 🚀 Key Features

* **Scalable Dual-Loop Coordination**: Master embedding acts as a coordination protocol, eliminating action-space dimensionality explosion when scaling the number of controlled cars.
* **Deep Attention Feature Extraction**: The `MasterModel` uses a custom `AttentionPolicyNetwork` implementing multi-head attention to capture dependencies across vehicle states.
* **Custom Highway-Env Layouts**: Leverages highly customized environments including a 4-way intersection layout (`RELintersection-v0`), double intersections, and roundabouts.
* **Complex Multi-Scenario Rotation**: Automatically generates up to 100 scenario permutations (25 base vehicle distributions × 4 clockwise rotations) to enhance agent generalization.
* **Cooperative Reward Engineering**: Combined rewards balancing collision penalties, speed limits, vehicle starvation prevention, and destination arrival bonuses.
* **Extensive Baselines**: Includes implementations of COMA, IDM, IPPO, MA-GA-DDPG, VDN, and VN-MA-DDPG for comprehensive comparison.

---

## 📊 System Architecture

### 1. Dual-Loop Latent Action Flow
The diagram below illustrates how global environment observations are compressed by the Master Model into coordinate-latent vectors, which are then fused with local car observations to guide individual vehicle decisions.

```mermaid
graph TD
    subgraph Env ["RELintersection-v0 (Gymnasium Environment)"]
        EnvState["Global Intersection State (N × 4 Features)"]
        LocalObs["Local Car State (x, y, vx, vy)"]
    end

    subgraph Master ["Master Model (SB3 PPO)"]
        Attention["AttentionPolicyNetwork (Multi-Head Attention)"]
        MasterPPO["Master Policy Net"]
        LatentEmb["Latent Coordination Embedding (Size: 4)"]
    end

    subgraph Agent ["Agent Model (Local PPO Controller)"]
        ConcatState["Concatenated State (Local Obs + Latent Emb = Size: 8)"]
        AgentPPO["Local Actor-Critic Net"]
        DiscreteAction["Discrete Action (0: Slow/Stop, 1: Fast)"]
    end

    %% Relationships
    EnvState -->|"Flattened State Vector"| Attention
    Attention --> MasterPPO
    MasterPPO -->|"Generates"| LatentEmb

    LocalObs --> ConcatState
    LatentEmb --> ConcatState
    ConcatState --> AgentPPO
    AgentPPO -->|"Applies Throttle Control"| DiscreteAction
    DiscreteAction -->|"Controls Individual Vehicles"| Env
```

### 2. Training Iteration Cycle
The sequence below outlines the step-by-step state propagation and alternate optimization cycle during training.

```mermaid
sequenceDiagram
    autonumber
    participant Env as RELintersection-v0
    participant MM as MasterModel (SB3 PPO)
    participant AM as AgentModel (Local PPO)
    participant Trainer as training_loop (training_handler.py)

    Note over Trainer: Starts Training Cycle
    Trainer->>Env: Reset environment & retrieve initial state
    Env-->>Trainer: Initial Global State
    Trainer->>MM: Prepare global state & forward
    MM-->>Trainer: Latent Coordination Embedding (Size: 4)
    
    Note over Trainer: For each controlled car: concat local state with Master Embedding
    Trainer->>AM: Predict action (Slow / Fast throttle control)
    AM-->>Trainer: Multi-Agent Actions (Throttle controls)
    
    Trainer->>Env: Step environment with action tuple
    Env-->>Trainer: Next State, Reward, Terminated, Truncated, Info
    
    Note over Trainer: Store step experience in RolloutBuffer & update weights alternately
```

---

## 📂 Codebase Directory Structure

```text
RL/
├── highwayenv/                 # Custom highway-env wrappers and configurations
│   ├── CustomControlledVehicle.py # Custom kinematic vehicle properties
│   ├── custom_action.py        # Custom action spaces and discrete action factories
│   ├── intersection_class.py   # Core RELintersection-v0 Gymnasium environment
│   ├── double_intersection_class.py # Double intersection layout environment
│   ├── roundabout_class.py     # Roundabout layout environment
│   └── utils.py                # Environment patching and registration utilities
├── src/                        # Core codebase logic
│   ├── baseline/               # Baseline MARL and rule-based algorithms
│   │   ├── coma.py             # COMA baseline
│   │   ├── idm.py              # IDM (Intelligent Driver Model) baseline
│   │   ├── ippo.py             # IPPO baseline
│   │   ├── ma_ga_ddpg.py       # MA-GA-DDPG algorithm
│   │   ├── vdn.py              # VDN baseline
│   │   └── vn_maddpg.py        # VN-MA-DDPG algorithm
│   ├── experiment/             # Simulation scenarios & environment configs
│   │   ├── experiment_config.py# Central hyperparameter and experiment class configuration
│   │   ├── scenarios.py        # Base lane layout and vehicle coordinates
│   │   └── scenarios_config.py # Complex multi-agent intersection scenarios
│   ├── model/                  # Neural network models
│   │   ├── agent_handler.py    # Local vehicle controller and Driver env wrapper
│   │   ├── master_model.py     # Master model with SimpleResNetExtractor
│   │   └── model_handler.py    # Model wrappers (PPO, DQN, A2C)
│   ├── plotting_utils/         # Data plotting and graphics tools
│   │   └── plotting_utils.py   # Training summary, loss curves, and reward plotting
│   └── training/               # Training loop orchestrators and buffers
│       ├── episode_utils.py    # Per-episode transitions and step execution
│       ├── general_utils.py    # Model initialization and logger setup
│       ├── training_handler.py # Orchestrates run modes (Training vs Inference)
│       └── training_loop_utils.py # Cycles and buffers optimization routines
├── run_parallel_experiment.py  # Parallel multi-seed experiment runner
├── requirements.txt            # System dependencies
└── README.md                   # Repository documentation
```

---

## ⚙️ Installation & Setup

### 1. Prerequisite Environments
Make sure you have **Python 3.10** or higher installed.

### 2. Clone and Install Dependencies
Navigate into the repository directory and configure the environment:

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On Windows Command Prompt:
.venv\Scripts\activate.bat
# On Linux/macOS:
source .venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

---

## 🎮 How to Run

### Run Parallel Experiments

This is the recommended workflow to run, evaluate, and compare all cooperative MARL algorithms across multiple seeds.

Execute the parallel runner to train/evaluate the algorithms across validation seeds (`42`, `100`, `2026`). It automatically skips already completed runs.
```bash
python run_parallel_experiment.py --episodes 900
```
- `--episodes`: Number of episodes per seed (default: `900`).
- `--seeds`: Comma-separated list of seeds (default: `42,100,2026`).

The results will be saved as JSON history files under `experiments/histories/`.

---

## 🛠️ Hyperparameter & Configuration Guide

You can find and modify all model configs inside [experiment_config.py](file:///Users/gil/PycharmProjects/RL/src/experiment/experiment_config.py).

### General & Training Settings
| Configuration Parameter | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `CYCLES` | `int` | `3` | Number of training cycles to perform |
| `EPISODES_PER_CYCLE` | `int` | `300` | Episodes run per cycle |
| `EPISODE_AMOUNT_FOR_TRAIN` | `int` | `2` | Number of episodes collected before backpropagation |
| `ONLY_INFERENCE` | `bool` | `False` | Toggle to run ONLY evaluation inference |
| `RENDER_MODE` | `str` | `"rgb_array"` | Gymnasium rendering mode (`"human"` or `"rgb_array"`) |

### Model Architecture Parameters
| Configuration Parameter | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `MODEL_TYPE` | `str` | `"PPO"` | Core model type (`"PPO"`, `"DQN"`, `"A2C"`) |
| `LEARNING_RATE` | `float` | `0.005` | Learning rate for optimizers |
| `N_STEPS` | `int` | `64` | Experience steps collected per rollout buffer |
| `BATCH_SIZE` | `int` | `32` | Minibatch size for policy updates |
| `EMBEDDING_SIZE` | `int` | `4` | Dimension of the latent coordination Master embedding |

### Environment Reward Profiles
| Reward Variable | Default Value | Description |
| :--- | :--- | :--- |
| `REACHED_TARGET_REWARD` | `50` | Bonus rewarded when a vehicle successfully crosses the intersection |
| `COLLISION_REWARD` | `-300` | Penalty enforced when vehicles collide with other structures or cars |
| `STARVATION_REWARD` | `-5` | Penalty when speed falls below fixed throttle limit (keeps cars moving) |
| `HIGH_SPEED_REWARD` | `5` | Reward granted when speed exceeds throttle target without crashes |
