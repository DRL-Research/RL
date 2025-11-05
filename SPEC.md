# SafeR-ADAIM Specification

## Problem Statement
Autonomous intersection management (AIM) must coordinate connected and
automated vehicles (CAVs) through all-direction turn lane (ADTL)
intersections without traffic signals. Traditional optimization-based
methods ensure safety but are computationally expensive, while reward-only
reinforcement learning (RL) approaches often neglect safety constraints.
Safe Reinforced All-Directional AIM (SafeR-ADAIM) combines risk-sensitive
policy optimization with multi-objective reward design to deliver safe,
efficient and comfortable intersection crossing in unpredictable traffic.

## Environment Abstraction
* **State** `S=[d, v, β]` – for each controlled vehicle: distance to exit
  along route (`d`), current speed (`v`) and turn intention (`β` as left,
  straight or right one-hot encoded). State is concatenated for all
  controlled vehicles.
* **Action** `A=[v'_i]` – desired velocity per vehicle in metres per
  second, bounded between `[v_min, v_max]`. The environment converts the
  desired speed into throttle/brake inputs via the existing PID controller.
* **Reward** – sum of three components evaluated per episode:
  * Efficiency `R_eff`: dense speed incentive with sparse bonuses for
    individual and collective completion.
  * Comfort `R_comfort`: penalties proportional to acceleration and jerk
    magnitudes.
  * Safety `R_safety = -C_total` uses the same costs defined below.
* **Cost** – separate dense collision-risk cost (distance threshold) and
  sparse collision cost. The cumulative cost is constrained during policy
  updates.

## Algorithmic Overview
SafeR-ADAIM is built on Risk Situation-aware Constrained Policy
Optimization (RSCPO):
1. Collect on-policy trajectories using a Gaussian policy.
2. Estimate reward and cost advantages with GAE.
3. Compute policy gradient `g` and cost gradient `b`.
4. Approximate the Fisher information via the Hessian of the KL divergence
   using conjugate gradients.
5. Classify the current policy risk level with
   `c_hat = J_C(π) - d` and `F = δ - (c_hat^2)/(b^T H^{-1} b)`.
   * **Risk-free** (`c_hat < 0`, `F > 0`): TRPO-style step.
   * **Moderate risk** (`c_hat > 0`, `F > 0`): solve the dual to blend
     reward and cost gradients while respecting constraints.
   * **High risk** (`F ≤ 0`): step along the cost gradient to re-enter the
     feasible region.
6. Use backtracking line-search to satisfy KL and cost limits.
7. Update reward and cost value networks via MSE regression.

## Hyper-parameters (adapted defaults)
* Policy/value hidden widths: `[128, 128]` (policy/value), `[128, 128, 128, 128]` (cost value).
* Rollout length: `2048` steps per epoch, `γ=0.99`, `λ=0.95` for both
  reward and cost.
* Trust-region radius: `δ = 0.01`.
* Cost limit per epoch: `0.01` (tunable per scenario).
* Dense reward/cost coefficients derived from the paper (ε_v, ε_t,
  ε_pass_single, ε_pass_all, ε_a, ε_j, ε_d, ε_c).

## Expected Outputs
* Learned Gaussian policy mapping the SafeR-ADAIM state into desired
  velocities.
* Reward, cost and KL diagnostics for each epoch.
* Serialized PyTorch weights for policy, reward value and cost value
  networks.
