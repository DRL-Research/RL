# SafeR-ADAIM Design Mapping

## High-Level Architecture
1. **Environment Adaptation** – `src/safe_radaim/env_wrapper.py` wraps the
   patched `IntersectionEnv` to expose the SafeR-ADAIM state, action and
   cost interfaces. It installs the new
   `ExpectedVelocityMultiAgentAction` (`highwayenv/custom_action.py`) so the
   policy can output desired velocities directly.
2. **Neural Modules** – `src/safe_radaim/networks.py` implements the
   Gaussian policy, reward value and cost value networks that mirror the
   architecture specified in the paper.
3. **Algorithm Core** – `src/safe_radaim/algorithm.py` houses the
   Risk Situation-aware Constrained Policy Optimisation (RSCPO) update.
   It computes gradients, classifies the risk case, solves the dual for the
   moderate-risk regime and performs constrained line-search.
4. **Trajectory Loop** – `src/safe_radaim/training.py` collects
   on-policy trajectories, estimates reward/cost advantages and orchestrates
   policy/value updates via `SafeRADAIMAgent`.
5. **Entrypoint** – `safe_radaim_main.py` builds the wrapped environment,
   initialises the agent with `SafeRADAIMConfig`, runs training through
   `SafeRADAIMTrainer` and saves the resulting PyTorch weights under
   `experiments/`.

## Key Data Flows
1. `gym.make('RELintersection-v0')` → `SafeIntersectionEnv` translates raw
   vehicle kinematics into `[d, v, β_onehot]` feature vectors and derives
   dense/sparse costs.
2. `SafeRADAIMTrainer` (per epoch):
   * collects `steps_per_epoch` transitions using `SafeRADAIMAgent.act`;
   * converts them into a `TrajectoryBatch` (states, actions, rewards,
     costs, log-probs, advantages, targets);
   * calls `agent.update_values` then `agent.update_policy`.
3. `agent.update_policy`:
   * computes policy gradient `g`, cost gradient `b` and Hessian-vector
     products via conjugate gradients;
   * classifies the risk regime and computes the step direction;
   * runs line-search constrained by KL and estimated post-update cost;
   * writes the updated parameters back into the policy network.

## External Interfaces
* **Configuration** – `src/safe_radaim/config.py` exposes tunable defaults
  (reward/cost coefficients, cost limit, network widths). `SafeRADAIMConfig`
  can be customised via CLI arguments in `safe_radaim_main.py`.
* **Outputs** – serialized PyTorch weights (`safe_radaim_policy.pt`,
  `safe_radaim_value.pt`, `safe_radaim_cost_value.pt`) and stdout logging of
  epoch metrics (reward, cost, KL, risk case).

## Integration Points
* The wrapper preserves compatibility with the existing experiment config
  (`src/experiment/scenarios_config.py`) so SafeR-ADAIM reuses the same
  scenario definitions.
* Existing PPO-based pipeline remains untouched; SafeR-ADAIM lives alongside
  the original training stack, enabling side-by-side comparisons.
