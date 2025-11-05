# SafeR-ADAIM Migration Plan

## Objectives
* Introduce the SafeR-ADAIM control stack alongside the existing PPO
  training code without disrupting current workflows.
* Provide a clear path for replacing or augmenting the legacy agent with
  the SafeR-ADAIM agent in experiments.

## Steps
1. **Environment Preparation**
   * Ensure `highwayenv` patches are loaded via `patch_intersection_env()`
     and `register_intersection_env()`.
   * Use `SafeIntersectionEnv` to wrap any existing intersection scenarios.
     The wrapper accepts the same configuration dictionaries already used in
     `scenarios_config`.
2. **Training Pipeline**
   * Instantiate `SafeRADAIMAgent` with observation/action dimensions derived
     from the wrapped environment.
   * Configure rollout length, velocity bounds and reward/cost coefficients
     through `SafeRADAIMConfig`.
   * Use `SafeRADAIMTrainer` to run training epochs. The trainer collects
     trajectories, updates the value networks and then applies the RSCPO step.
3. **Evaluation / Deployment**
   * After training, load the serialized weights saved in `experiments/`.
   * Re-wrap the environment in inference mode and feed observations through
     the trained policy to obtain desired velocities.
4. **Coexistence with PPO**
   * The original PPO training loop (`main.py`) remains untouched. Teams can
     run PPO-based experiments in parallel with SafeR-ADAIM by choosing the
     appropriate entrypoint.
   * Shared assets (scenario definitions, vehicle classes, logging directory)
     remain compatible across both pipelines.

## Validation
* Run `python safe_radaim_main.py --epochs 1 --steps-per-epoch 128` to
  perform a smoke test. Confirm that artefacts are saved and KL/cost metrics
  are reported.
* Compare collision rates between PPO and SafeR-ADAIM by reusing the same
  environment seeds, leveraging the unified wrappers for a fair comparison.
