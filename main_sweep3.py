"""
Sweep 3 — Combining Z07 (ent=0.05) + Z10 (clip=0.2) findings.

Sweep 2 winners:
  Z07_ent05   → ent=0.05  (reward growing to +30, 85% arrival, 391 crashes)
  Z10_clip02  → clip=0.2  (reward growing to +45, 85% arrival, 329 crashes)
  Z06_lr1e3   → lr=1e-3   (fewest crashes 297, reward +40)

Both Z07 and Z10 changed ONE thing from the champion and both improved.
We now test their combination and identify which factor drives each improvement.

5 configs × 1200 episodes:
  W01 — Z07 + Z10 combined          (ent=0.05, clip=0.2, lr=3e-3)
  W02 — Z07 + Z10 + Z06 combined    (ent=0.05, clip=0.2, lr=1e-3) ← best guess
  W03 — only clip=0.2, ent=0.01     (isolate: is clip=0.2 alone enough?)
  W04 — only ent=0.05, clip=0.1     (isolate: is ent=0.05 alone enough?)
  W05 — W02 + epochs=10             (push gradient updates further)

All other params locked to confirmed-best from sweep 2:
  gamma=0.90, gae=0.90, arch=wide, ep_for_train=3,
  cotrain=False, starvation=0, master_lr=3e-4,
  collision=-50, vf_coef=1.0, n_ppo_epochs=5 (except W05)
"""

import os, json, csv, logging
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as _cm

from highwayenv.utils import patch_intersection_env, register_intersection_env
from src import project_globals
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.training.training_handler import training_loop
from src.model.model_handler import save_models
from src.training.general_utils import initialize_models, setup_experiment_dirs, setup_loggers

logging.basicConfig(level=logging.WARNING)

# ── helpers ────────────────────────────────────────────────────────────────────

def _smooth(values, window=40):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    window = max(1, min(window, len(arr)))
    kernel = np.ones(window) / window
    pad    = window // 2
    padded = np.concatenate([np.full(pad, np.nan), arr, np.full(pad, np.nan)])
    s = np.convolve(np.where(np.isnan(padded), 0, padded), kernel, mode='valid')
    c = np.convolve((~np.isnan(padded)).astype(float), kernel, mode='valid')
    s = s / np.where(c > 0, c, 1)
    s[c == 0] = np.nan
    return s[:len(arr)]


def _nanmean_all(lst):
    arr = np.array([v for v in (lst or []) if v is not None and not np.isnan(v)], dtype=float)
    return float(np.nanmean(arr)) if len(arr) else None


def _nanmean_last(lst, n=10):
    tail = [v for v in (lst or [])[-n:] if v is not None and not np.isnan(v)]
    return float(np.nanmean(np.array(tail, dtype=float))) if tail else None


# ── champion defaults for this sweep ──────────────────────────────────────────
# (locked params confirmed in sweep 2)
_BASE = dict(
    ent_coef=0.05,
    collision_reward=-50,
    arrived_reward=50,
    reward_mode='global',
    cotrain_cycles=False,
    agent_lr=3e-3,
    master_lr=3e-4,
    clip_range=0.2,
    gamma=0.90,
    gae_lambda=0.90,
    agent_net_arch='wide',
    ep_for_train=3,
    vf_coef=1.0,
    n_ppo_epochs=5,
    starvation_reward=0,
    high_speed_reward=5,
)


# ── grid ───────────────────────────────────────────────────────────────────────

SWEEP_CONFIGS = [
    # W01: Z07 + Z10 directly combined (ent=0.05, clip=0.2) — base agent_lr=3e-3
    {**_BASE, 'label': 'W01_ent05_clip02'},

    # W02: W01 + Z06 best lr (1e-3)  ← main hypothesis for best overall
    {**_BASE, 'label': 'W02_lr1e3', 'agent_lr': 1e-3},

    # W03: only clip=0.2 changed, ent stays 0.01 (isolate clip contribution)
    {**_BASE, 'label': 'W03_clip02_ent01', 'ent_coef': 0.01},

    # W04: only ent=0.05, clip stays 0.1 (isolate ent contribution)
    {**_BASE, 'label': 'W04_ent05_clip01', 'clip_range': 0.1},

    # W05: W02 (best combo) with n_ppo_epochs=10
    {**_BASE, 'label': 'W05_epochs10', 'agent_lr': 1e-3, 'n_ppo_epochs': 10},
]


# ── config runner ──────────────────────────────────────────────────────────────

def run_config(cfg: dict, sweep_root: str):
    label = cfg['label']

    print(f"\n{'='*70}")
    print(f"  CONFIG: {label}")
    print(f"  ent={cfg['ent_coef']}  clip={cfg['clip_range']}  "
          f"agent_lr={cfg['agent_lr']:.2e}  master_lr={cfg['master_lr']:.2e}")
    print(f"  gamma={cfg['gamma']}  gae={cfg['gae_lambda']}  "
          f"arch={cfg['agent_net_arch']}  ep4train={cfg['ep_for_train']}")
    print(f"  cotrain={cfg['cotrain_cycles']}  vf={cfg['vf_coef']}  "
          f"epochs={cfg['n_ppo_epochs']}  starv={cfg['starvation_reward']}")
    print(f"{'='*70}\n")

    project_globals.reset_globals()

    section  = label.split('_')[0]
    exp_path = os.path.join(sweep_root, section, label)

    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID=f'grid_{label}',
        LOAD_MODEL_DIRECTORY='',
        EPOCHS=1,
        CYCLES=4,
        ENT_COEF=cfg['ent_coef'],
        COLLISION_REWARD=cfg['collision_reward'],
        REACHED_TARGET_REWARD=cfg['arrived_reward'],
        STARVATION_REWARD=cfg['starvation_reward'],
        HIGH_SPEED_REWARD=cfg['high_speed_reward'],
        AGENT_REWARD_MODE=cfg['reward_mode'],
        COTRAIN_CYCLES=cfg['cotrain_cycles'],
        AGENT_LR=cfg['agent_lr'],
        MASTER_LR=cfg['master_lr'],
        CLIP_RANGE=cfg['clip_range'],
        GAMMA=cfg['gamma'],
        GAE_LAMBDA=cfg['gae_lambda'],
        AGENT_NET_ARCH=cfg['agent_net_arch'],
        EPISODE_AMOUNT_FOR_TRAIN=cfg['ep_for_train'],
        VF_COEF=cfg['vf_coef'],
        N_PPO_EPOCHS=cfg['n_ppo_epochs'],
    )
    exp.EXPERIMENT_PATH      = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, 'trained_model')

    env_config = sc.make_env_config_exp7(
        collision_reward=cfg['collision_reward'],
        arrived_reward=cfg['arrived_reward'],
        starvation_reward=cfg['starvation_reward'],
        high_speed_reward=cfg['high_speed_reward'],
    )

    setup_experiment_dirs(exp.EXPERIMENT_PATH)
    master_model, agent_model, wrapped_env = initialize_models(exp, env_config)
    agent_logger, master_logger = setup_loggers(exp.EXPERIMENT_PATH)
    agent_model.set_logger(agent_logger)
    master_model.set_logger(master_logger)

    agent_model, master_model, collisions, _, _, results = training_loop(
        experiment=exp, env=wrapped_env,
        agent_model=agent_model, master_model=master_model,
    )

    save_models(agent_model, master_model, exp.SAVE_MODEL_DIRECTORY)
    try:
        wrapped_env.env.highway_env.close()
    except Exception:
        pass

    save_config_plots(results, exp_path, label)
    arr_final = np.nanmean([v for v in results.get('arrival_rates', [np.nan])[-20:] if v is not None] or [np.nan])
    print(f"\n  [{label}] done — collisions={collisions}  arr_last20={arr_final:.1f}%")
    return collisions, results


# ── per-config 4-panel plot ────────────────────────────────────────────────────

def save_config_plots(results: dict, exp_path: str, label: str):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Config: {label}', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(2)]

    def _ps(ax, vals, color, lbl, alpha_raw=0.15):
        if not vals:
            return
        raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        x   = np.arange(1, len(raw) + 1)
        ax.plot(x, raw, color=color, alpha=alpha_raw, linewidth=0.6)
        ax.plot(x, _smooth(vals), color=color, linewidth=2.0, label=lbl)

    _ps(axes[0], results.get('arrival_rates', []),   '#2196F3', 'Arrival %')
    axes[0].set_title('Arrival Rate'); axes[0].set_ylabel('%')
    axes[0].set_ylim(-5, 105); axes[0].grid(True, alpha=0.25); axes[0].legend(fontsize=8)

    _ps(axes[1], results.get('episode_rewards', []), '#4CAF50', 'Reward')
    axes[1].set_title('Episode Reward'); axes[1].set_ylabel('Reward')
    axes[1].grid(True, alpha=0.25); axes[1].legend(fontsize=8)

    for key, col, lbl_s in [
        ('master_policy_losses', '#E53935', 'Policy'),
        ('master_value_losses',  '#FB8C00', 'Value'),
        ('master_total_losses',  '#8E24AA', 'Total'),
    ]:
        _ps(axes[2], results.get(key, []), col, lbl_s, 0.2)
    axes[2].set_title('Master Losses'); axes[2].set_ylabel('Loss')
    axes[2].grid(True, alpha=0.25); axes[2].legend(fontsize=8)

    for key, col, lbl_s in [
        ('agent_group0_total_losses', '#00ACC1', 'G0'),
        ('agent_group1_total_losses', '#43A047', 'G1'),
    ]:
        _ps(axes[3], results.get(key, []), col, lbl_s, 0.2)
    axes[3].set_title('Agent Losses'); axes[3].set_ylabel('Loss')
    axes[3].grid(True, alpha=0.25); axes[3].legend(fontsize=8)

    for ax in axes:
        ax.set_xlabel('Episode' if axes.index(ax) < 2 else 'Training call')

    plt.savefig(os.path.join(exp_path, 'results.png'), dpi=120, bbox_inches='tight')
    plt.close()


# ── comparison plot (all 5 configs together) ───────────────────────────────────

_COLORS = [_cm.tab10(i) for i in np.linspace(0, 1, 10, endpoint=False)]


def save_comparison_plot(all_results, sweep_configs, sweep_root):
    done = [c for c in sweep_configs if c['label'] in all_results]
    if not done:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Sweep 3 — All Configs Comparison', fontweight='bold', fontsize=13)

    for i, c in enumerate(done):
        lbl = c['label']
        res = all_results[lbl]
        col = _COLORS[i]
        ncol = res.get('total_collisions', '?')
        tag  = f"{lbl} ({ncol}col)"
        arr  = res.get('arrival_rates', [])
        rew  = res.get('episode_rewards', [])
        for ax, vals in [(axes[0], arr), (axes[1], rew)]:
            if vals:
                raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
                ax.plot(raw, color=col, alpha=0.1, linewidth=0.6)
                ax.plot(_smooth(vals), color=col, linewidth=2.0, label=tag)

    axes[0].set_title('Arrival Rate (%)'); axes[0].set_ylabel('%')
    axes[0].set_ylim(-5, 105); axes[0].set_xlabel('Episode')
    axes[1].set_title('Episode Reward'); axes[1].set_ylabel('Reward')
    axes[1].set_xlabel('Episode')
    for ax in axes:
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, loc='best')

    plt.tight_layout()
    plt.savefig(os.path.join(sweep_root, 'sweep3_comparison.png'), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [comparison] {sweep_root}/sweep3_comparison.png")


# ── incremental JSON/CSV ───────────────────────────────────────────────────────

def save_incremental_outputs(sweep_root, all_results, sweep_configs):
    summary = {}
    for cfg in sweep_configs:
        lbl = cfg['label']
        if lbl not in all_results:
            continue
        res      = all_results[lbl]
        arr_vals = [v for v in res.get('arrival_rates', []) if v is not None]
        last20   = arr_vals[-20:] if len(arr_vals) >= 20 else arr_vals
        summary[lbl] = {
            'total_collisions':        res.get('total_collisions', None),
            'arrival_rate_avg_pct':    float(np.mean(arr_vals))  if arr_vals else None,
            'arrival_rate_last20_pct': float(np.mean(last20))    if last20   else None,
            'master_policy_loss_avg':  _nanmean_all(res.get('master_policy_losses', [])),
            'master_policy_loss_last': _nanmean_last(res.get('master_policy_losses', [])),
            'master_value_loss_avg':   _nanmean_all(res.get('master_value_losses', [])),
            'master_value_loss_last':  _nanmean_last(res.get('master_value_losses', [])),
            'agent_g0_loss_avg':       _nanmean_all(res.get('agent_group0_total_losses', [])),
            'agent_g1_loss_avg':       _nanmean_all(res.get('agent_group1_total_losses', [])),
            'cfg': {k: v for k, v in cfg.items() if k != 'label'},
        }

    with open(os.path.join(sweep_root, 'grid_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    stat_cols = ['total_collisions', 'arrival_rate_avg_pct', 'arrival_rate_last20_pct',
                 'master_policy_loss_avg', 'master_policy_loss_last',
                 'master_value_loss_avg',  'master_value_loss_last',
                 'agent_g0_loss_avg',      'agent_g1_loss_avg']
    all_keys = sorted({k for cfg in sweep_configs for k in cfg if k != 'label'})
    with open(os.path.join(sweep_root, 'grid_summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + stat_cols + all_keys)
        for lbl, d in summary.items():
            row = [lbl] + [d.get(c) for c in stat_cols]
            for k in all_keys:
                row.append(d['cfg'].get(k, ''))
            writer.writerow(row)

    save_comparison_plot(all_results, sweep_configs, sweep_root)
    print(f"  → outputs updated in {sweep_root}/")


# ── resume ─────────────────────────────────────────────────────────────────────

def _reconstruct_all_results(sweep_root, sweep_configs):
    json_path = os.path.join(sweep_root, 'grid_summary.json')
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        raw = json.load(f)
    return {
        lbl: {
            'arrival_rates':             [d.get('arrival_rate_last20_pct')],
            'episode_rewards':           [],
            'total_collisions':          d.get('total_collisions', 0),
            'master_policy_losses':      [],
            'master_value_losses':       [],
            'master_total_losses':       [],
            'agent_group0_total_losses': [],
            'agent_group1_total_losses': [],
        }
        for lbl, d in raw.items()
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    patch_intersection_env()
    register_intersection_env()

    ts         = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')
    sweep_root = os.path.join('experiments', f'sweep3_{ts}')
    os.makedirs(sweep_root, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"  SWEEP-3  root: {sweep_root}")
    print(f"  {len(SWEEP_CONFIGS)} configs × 1200 episodes")
    print(f"  Goal: combine Z07 (ent=0.05) + Z10 (clip=0.2) findings")
    print(f"{'#'*70}\n")

    all_results = _reconstruct_all_results(sweep_root, SWEEP_CONFIGS)
    completed   = set(all_results.keys())

    for i, cfg in enumerate(SWEEP_CONFIGS, 1):
        lbl = cfg['label']
        if lbl in completed:
            print(f"[{i}/{len(SWEEP_CONFIGS)}] SKIP: {lbl}")
            continue

        print(f"[{i}/{len(SWEEP_CONFIGS)}]")
        try:
            collisions, results = run_config(cfg, sweep_root)
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] Config {lbl} failed: {exc}")
            traceback.print_exc()
            results    = {k: [] for k in ['arrival_rates', 'episode_rewards',
                          'master_policy_losses', 'master_value_losses',
                          'master_total_losses', 'agent_group0_total_losses',
                          'agent_group1_total_losses']}
            collisions = -1

        results['total_collisions'] = collisions
        all_results[lbl] = results
        save_incremental_outputs(sweep_root, all_results, SWEEP_CONFIGS)

    print(f"\n{'#'*70}")
    print(f"  SWEEP-3 complete.  Results: {sweep_root}")
    print(f"{'#'*70}")
