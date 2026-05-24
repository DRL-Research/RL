# AutoResearch Operating Manual: Cooperative Multi-Agent Intersection Coordination

Welcome to the AutoResearch environment for Cooperative Multi-Agent Reinforcement Learning (MARL). 
Your objective is to optimize the **Dual-Loop Latent Master model** to safely and efficiently coordinate autonomous vehicles at a busy intersection.

---

## 🎯 The Research Goal

Your goal is to optimize the cooperative driving policy to maximize safety and coordination efficiency. The performance is assessed across three fixed validation seeds (`42`, `100`, `2026`), evaluating:
1. **Validation Success Rate (Primary Metric)**: The percentage of episodes where all vehicles successfully exit the intersection without colliding. Target: **> 90%** (100% is ideal).
2. **Mean Episode Reward (Secondary Metric)**: The average cooperative reward accumulated by the vehicles per episode.

---

## 📜 Rules & Constraints

To maintain a valid, scientific, and cheat-free benchmark:
1. **Mutable Sandbox**: You are **ONLY** permitted to modify [train.py](file:///c:/PycharmProjects/RL/train.py).
2. **Immutable Harness**: Do **NOT** modify [prepare.py](file:///c:/PycharmProjects/RL/prepare.py) or any files inside the `src/` or `highwayenv/` directories.
3. **Ratchet Loop Mechanics**:
   - Make a targeted modification to `train.py`.
   - Run the experiment and validation: `python train.py`.
   - Read the console logs and [results.tsv](file:///c:/PycharmProjects/RL/results.tsv).
   - If the script exits with **0** (improved metrics), **keep and commit** the changes!
   - If the script exits with **1** (no improvement), **discard** the changes using `git checkout train.py` and formulate a new hypothesis.

---

## 🚀 Research & Optimization Directions

### 1. Neural Architecture Innovation (`SimpleResNetExtractor`)
The `SimpleResNetExtractor` class in `train.py` extracts coordinate feature representation for the `MasterModel`. You have full freedom to innovate here:
- **Depth & Skip Connections**: Add more residual blocks (`res_block3`, `res_block4`) to construct a deeper ResNet.
- **Layer Widths**: Try wider linear layers (e.g., 256, 512, 128) to increase feature capacity.
- **Activations**: Replace standard `nn.ReLU()` with modern activations like `nn.GELU()`, `nn.SiLU()` (Swish), or `nn.LeakyReLU()`.
- **Normalization & Regularization**: Introduce `nn.LayerNorm(128)` or `nn.Dropout(0.1)` inside the residual blocks to prevent gradient explosion and stabilize updates.

### 2. Hyperparameter Tuning
Hyperparameters at the top of `train.py` significantly affect training stability and speed:
- **Learning Rate (`LEARNING_RATE`)**: Tune the learning rate (e.g., `0.001`, `0.005`, `0.0005`).
- **Batch Size (`BATCH_SIZE`)** & **Rollout Buffer Steps (`N_STEPS`)**: Balance policy updates. Common configurations include `BATCH_SIZE=64 / N_STEPS=128`.
- **Embedding Size (`EMBEDDING_SIZE`)**: Modify the size of the Master Model's latent embedding (e.g. increase to `8` or decrease to `2`) to see how communication bandwidth affects coordination.

### 3. Environment Reward Shaping
Reward shaping guides cooperative policies. Adjust the coefficients in `train.py`:
- **Collision Penalty (`COLLISION_REWARD`, baseline `-300`)**: A heavier penalty makes vehicles highly cautious; too heavy, and they may refuse to cross (starvation).
- **Target Exit Reward (`REACHED_TARGET_REWARD`, baseline `50`)**: Increase this to encourage cars to cross and arrive at destinations quickly.
- **Starvation Penalty (`STARVATION_REWARD`, baseline `-5`)**: Penalizes cars that stay still. Increase this if cars get stuck at green lights, or decrease it if they collide trying to rush.
- **High Speed Reward (`HIGH_SPEED_REWARD`, baseline `5`)**: Rewards faster throughput.

---

## 📊 Baselines

| Model / Configuration | Success Rate | Avg Reward | Description |
| :--- | :--- | :--- | :--- |
| **Baseline (Original)** | ~33% - 44% | ~-25.00 | Default hyperparameters with standard Relu ResNet extractor |

Good luck! Run `python train.py` to get your first benchmark and start proposing improvements!
