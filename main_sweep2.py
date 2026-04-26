"""
Focused follow-up sweep — 10 configs.

Based on analysis of the first 55-config sweep (sweep_08_04_2026-13_31_53):

CHAMPION parameters (one winner per section):
  A → ent=0.01,  collision=-50                   (85 % arrival)
  B → agent_lr=3e-3                               (94 % arrival ← biggest lever)
  C → master_lr=3e-4                              (85 %)
  D → clip=0.1                                    (75 %)
  E → gamma=0.90, gae=0.90                        (73 %)
  F → arch='wide'                                 (~90 %)
  G → ep_for_train=3                              (~91 %)
  H → cotrain=False                               (~79 %)
  I → vf_coef=1.0  AND  n_ppo_epochs=5           (~78 % each)
  J → starvation=0 (speed-only rewards)           (267 crashes, POSITIVE reward)

Grid design: one "all-best" champion + 9 single-axis ablations so we know
exactly what each factor contributes in combination.

Section labels start with "Z" to keep them separate from the first sweep.
"""

# ── re-use everything from main_sweep.py ──────────────────────────────────────
import os, sys, json, csv, logging
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

# ── Copy utility helpers from main_sweep.py ───────────────────────────────────

def _smooth(values, window=40):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if len(arr) < window:
        window = max(1, len(arr))
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
    arr  = np.array(tail, dtype=float)
    return float(np.nanmean(arr)) if len(arr) else None


# ── single-config runner ───────────────────────────────────────────────────────

def run_config(cfg: dict, sweep_root: str):
    label     = cfg['label']
    ent       = cfg.get('ent_coef',              0.01)
    coll_r    = cfg.get('collision_reward',       -50)
    arr_r     = cfg.get('arrived_reward',          50)
    mode      = cfg.get('reward_mode',         'global')
    cotrain   = cfg.get('cotrain_cycles',        False)
    agent_lr  = cfg.get('agent_lr',              3e-3)
    master_lr = cfg.get('master_lr',             3e-4)

    print(f"\n{'='*70}")
    print(f"  CONFIG: {label}")
    print(f"  ent={ent}  coll={coll_r}  arr={arr_r}  mode={mode}  cotrain={cotrain}")
    print(f"  clip={cfg.get('clip_range',0.1)}  gamma={cfg.get('gamma',0.90)}"
          f"  gae={cfg.get('gae_lambda',0.90)}  arch={cfg.get('agent_net_arch','wide')}"
          f"  ep4train={cfg.get('ep_for_train',3)}")
    print(f"  agent_lr={agent_lr:.2e}  master_lr={master_lr:.2e}"
          f"  vf_coef={cfg.get('vf_coef',1.0)}  n_ppo_epochs={cfg.get('n_ppo_epochs',5)}")
    print(f"  starvation={cfg.get('starvation_reward',0)}  high_speed={cfg.get('high_speed_reward',5)}")
    print(f"{'='*70}\n")

    project_globals.reset_globals()

    starvation_r = cfg.get('starvation_reward',  0)
    high_speed_r = cfg.get('high_speed_reward',  5)

    section  = label.split('_')[0]
    exp_path = os.path.join(sweep_root, section, label)

    exp = Experiment(
        RENDER_MODE=None,
        EXPERIMENT_ID=f'grid_{label}',
        LOAD_MODEL_DIRECTORY='',
        EPOCHS=1,
        CYCLES=4,
        ENT_COEF=ent,
        COLLISION_REWARD=coll_r,
        REACHED_TARGET_REWARD=arr_r,
        STARVATION_REWARD=starvation_r,
        HIGH_SPEED_REWARD=high_speed_r,
        AGENT_REWARD_MODE=mode,
        COTRAIN_CYCLES=cotrain,
        AGENT_LR=agent_lr,
        MASTER_LR=master_lr,
        CLIP_RANGE=cfg.get('clip_range',          0.1),
        GAMMA=cfg.get('gamma',                    0.90),
        GAE_LAMBDA=cfg.get('gae_lambda',          0.90),
        AGENT_NET_ARCH=cfg.get('agent_net_arch',  'wide'),
        EPISODE_AMOUNT_FOR_TRAIN=cfg.get('ep_for_train', 3),
        VF_COEF=cfg.get('vf_coef',                1.0),
        N_PPO_EPOCHS=cfg.get('n_ppo_epochs',       5),
    )
    exp.EXPERIMENT_PATH      = exp_path
    exp.SAVE_MODEL_DIRECTORY = os.path.join(exp_path, 'trained_model')

    env_config = sc.make_env_config_exp7(
        collision_reward=coll_r,
        arrived_reward=arr_r,
        starvation_reward=starvation_r,
        high_speed_reward=high_speed_r,
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

    arr_final = np.nanmean(results.get('arrival_rates', [np.nan])[-20:])
    print(f"\n  [{label}] done — collisions={collisions}  arr_last20={arr_final:.1f}%")
    return collisions, results


# ── per-config 4-panel plot ────────────────────────────────────────────────────

def save_config_plots(results: dict, exp_path: str, label: str):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Config: {label}', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    ax_arr = fig.add_subplot(gs[0, 0])
    ax_rew = fig.add_subplot(gs[0, 1])
    ax_mst = fig.add_subplot(gs[1, 0])
    ax_agt = fig.add_subplot(gs[1, 1])

    def _plot_series(ax, vals, color, lbl, alpha_raw=0.15):
        if not vals:
            return
        raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        x   = np.arange(1, len(raw) + 1)
        ax.plot(x, raw, color=color, alpha=alpha_raw, linewidth=0.6)
        ax.plot(x, _smooth(vals), color=color, linewidth=2.0, label=lbl)

    _plot_series(ax_arr, results.get('arrival_rates', []), '#2196F3', 'Arrival %')
    ax_arr.set_title('Success / Arrival Rate'); ax_arr.set_xlabel('Episode')
    ax_arr.set_ylabel('%'); ax_arr.set_ylim(-5, 105)
    ax_arr.grid(True, alpha=0.25); ax_arr.legend(fontsize=8)

    _plot_series(ax_rew, results.get('episode_rewards', []), '#4CAF50', 'Reward')
    ax_rew.set_title('Episode Reward'); ax_rew.set_xlabel('Episode')
    ax_rew.set_ylabel('Reward'); ax_rew.grid(True, alpha=0.25); ax_rew.legend(fontsize=8)

    for key, col, lbl_s in [
        ('master_policy_losses', '#E53935', 'Master Policy'),
        ('master_value_losses',  '#FB8C00', 'Master Value'),
        ('master_total_losses',  '#8E24AA', 'Master Total'),
    ]:
        _plot_series(ax_mst, results.get(key, []), col, lbl_s, alpha_raw=0.2)
    ax_mst.set_title('Master Losses'); ax_mst.set_xlabel('Training call')
    ax_mst.set_ylabel('Loss'); ax_mst.grid(True, alpha=0.25); ax_mst.legend(fontsize=8)

    for key, col, lbl_s in [
        ('agent_group0_total_losses', '#00ACC1', 'Agent G0 (cars 0-2)'),
        ('agent_group1_total_losses', '#43A047', 'Agent G1 (cars 3-5)'),
    ]:
        _plot_series(ax_agt, results.get(key, []), col, lbl_s, alpha_raw=0.2)
    ax_agt.set_title('Agent Losses'); ax_agt.set_xlabel('Training call')
    ax_agt.set_ylabel('Loss'); ax_agt.grid(True, alpha=0.25); ax_agt.legend(fontsize=8)

    out = os.path.join(exp_path, 'results.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [plot] {out}")


# ── comparison plots ───────────────────────────────────────────────────────────

_COLORS = [_cm.tab10(i) for i in np.linspace(0, 1, 10, endpoint=False)]


def _section_plot(sec, cfgs, all_results, out_path):
    done = [c for c in cfgs if c['label'] in all_results]
    if not done:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Section {sec} — Arrival Rate & Reward', fontweight='bold')

    for i, c in enumerate(done):
        lbl  = c['label']
        res  = all_results[lbl]
        col  = _COLORS[i % len(_COLORS)]
        ncol = res.get('total_collisions', '?')
        tag  = f"{lbl} ({ncol}col)"

        arr  = res.get('arrival_rates', [])
        rew  = res.get('episode_rewards', [])

        if arr:
            raw = np.array([v if v is not None else np.nan for v in arr], dtype=float)
            ax1.plot(raw, color=col, alpha=0.12, linewidth=0.6)
            ax1.plot(_smooth(arr), color=col, linewidth=1.8, label=tag)
        if rew:
            raw = np.array([v if v is not None else np.nan for v in rew], dtype=float)
            ax2.plot(raw, color=col, alpha=0.12, linewidth=0.6)
            ax2.plot(_smooth(rew), color=col, linewidth=1.8, label=tag)

    for ax, title, ylabel in [(ax1, 'Arrival Rate (%)', '%'), (ax2, 'Episode Reward', 'Reward')]:
        ax.set_title(title); ax.set_xlabel('Episode'); ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=1, loc='best')
    if any(res.get('arrival_rates') for res in all_results.values()):
        ax1.set_ylim(-5, 105)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [section plot] {out_path}")


def _ranking_plot(all_results, sweep_configs, out_path):
    rows = []
    for cfg in sweep_configs:
        lbl = cfg['label']
        if lbl not in all_results:
            continue
        arr_vals = all_results[lbl].get('arrival_rates', [])
        last20   = [v for v in arr_vals[-20:] if v is not None]
        rows.append((lbl, np.mean(last20) if last20 else 0,
                     all_results[lbl].get('total_collisions', 0)))
    if not rows:
        return
    rows.sort(key=lambda x: -x[1])
    labels  = [r[0] for r in rows]
    arrivals = [r[1] for r in rows]
    crashes  = [r[2] for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(12, len(rows)*0.9), 9))
    fig.suptitle('Grid Ranking — All Completed Configs', fontweight='bold')
    colors = [_COLORS[i % len(_COLORS)] for i in range(len(rows))]
    ax1.bar(range(len(rows)), arrivals, color=colors)
    ax1.set_xticks(range(len(rows))); ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax1.set_ylabel('Arrival Rate last-20 (%)'); ax1.set_ylim(0, 105)
    ax1.axhline(80, color='green', linestyle='--', alpha=0.5, label='80 % target')
    ax1.legend(fontsize=8); ax1.grid(axis='y', alpha=0.3)

    ax2.bar(range(len(rows)), crashes, color=colors)
    ax2.set_xticks(range(len(rows))); ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Total Crashes'); ax2.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [ranking plot] {out_path}")


# ── incremental outputs ────────────────────────────────────────────────────────

def save_incremental_outputs(sweep_root, all_results, sweep_configs):
    summary = {}
    for cfg in sweep_configs:
        lbl = cfg['label']
        if lbl not in all_results:
            continue
        res       = all_results[lbl]
        arr_vals  = [v for v in res.get('arrival_rates', []) if v is not None]
        last20    = arr_vals[-20:] if len(arr_vals) >= 20 else arr_vals
        summary[lbl] = {
            'total_collisions':        res.get('total_collisions', None),
            'arrival_rate_avg_pct':    float(np.mean(arr_vals))       if arr_vals  else None,
            'arrival_rate_last20_pct': float(np.mean(last20))         if last20    else None,
            'master_policy_loss_avg':  _nanmean_all(res.get('master_policy_losses', [])),
            'master_policy_loss_last': _nanmean_last(res.get('master_policy_losses', [])),
            'master_value_loss_avg':   _nanmean_all(res.get('master_value_losses', [])),
            'master_value_loss_last':  _nanmean_last(res.get('master_value_losses', [])),
            'agent_g0_loss_avg':       _nanmean_all(res.get('agent_group0_total_losses', [])),
            'agent_g0_loss_last':      _nanmean_last(res.get('agent_group0_total_losses', [])),
            'agent_g1_loss_avg':       _nanmean_all(res.get('agent_group1_total_losses', [])),
            'agent_g1_loss_last':      _nanmean_last(res.get('agent_group1_total_losses', [])),
            'cfg': {k: v for k, v in cfg.items() if k != 'label'},
        }

    with open(os.path.join(sweep_root, 'grid_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # CSV
    stat_cols = ['total_collisions', 'arrival_rate_avg_pct', 'arrival_rate_last20_pct',
                 'master_policy_loss_avg', 'master_policy_loss_last',
                 'master_value_loss_avg',  'master_value_loss_last',
                 'agent_g0_loss_avg',      'agent_g0_loss_last',
                 'agent_g1_loss_avg',      'agent_g1_loss_last']
    all_keys = sorted({k for cfg in sweep_configs for k in cfg if k != 'label'})
    with open(os.path.join(sweep_root, 'grid_summary.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + stat_cols + all_keys)
        for lbl, d in summary.items():
            row = [lbl] + [d.get(c) for c in stat_cols]
            for k in all_keys:
                row.append(d['cfg'].get(k, ''))
            writer.writerow(row)

    # Section + ranking plots
    sections = {}
    for cfg in sweep_configs:
        sec = cfg['label'].split('_')[0]
        sections.setdefault(sec, []).append(cfg)
    for sec, cfgs in sections.items():
        if any(c['label'] in all_results for c in cfgs):
            _section_plot(sec, cfgs, all_results,
                          os.path.join(sweep_root, f'grid_section_{sec}.png'))
    _ranking_plot(all_results, sweep_configs,
                  os.path.join(sweep_root, 'grid_ranking.png'))
    print(f"  → outputs updated in {sweep_root}/")


# ── resume helper ──────────────────────────────────────────────────────────────

def _reconstruct_all_results(sweep_root, sweep_configs):
    """Re-load results from existing grid_summary.json (arrival/crash only)."""
    json_path = os.path.join(sweep_root, 'grid_summary.json')
    if not os.path.exists(json_path):
        return {}
    with open(json_path) as f:
        raw = json.load(f)
    out = {}
    for lbl, d in raw.items():
        out[lbl] = {
            'arrival_rates':    [d.get('arrival_rate_last20_pct')],
            'episode_rewards':  [],
            'total_collisions': d.get('total_collisions', 0),
            'master_policy_losses': [],
            'master_value_losses':  [],
            'master_total_losses':  [],
            'agent_group0_total_losses': [],
            'agent_group1_total_losses': [],
        }
    return out


# ══════════════════════════════════════════════════════════════════════════════
# SWEEP CONFIGS  (10 configs)
#
# CHAMPION defaults (all best from sweep 1):
#   ent=0.01, collision=-50, arrived=50
#   agent_lr=3e-3, master_lr=3e-4
#   clip=0.1, gamma=0.90, gae=0.90, arch='wide'
#   ep_for_train=3, cotrain=False
#   vf_coef=1.0, n_ppo_epochs=5
#   starvation=0, high_speed=5
#
# Each Z## config changes exactly ONE axis from the champion so we can
# isolate what each factor contributes in the combined setting.
# ══════════════════════════════════════════════════════════════════════════════

SWEEP_CONFIGS = [

    # ── Z01: FULL CHAMPION — all best combined ─────────────────────────────────
    dict(label='Z01_champion'),   # all defaults = champion

    # ── Z02: cotrain ON (ablate cotrain=False) ─────────────────────────────────
    dict(label='Z02_cotrain',     cotrain_cycles=True),

    # ── Z03: gamma=0.99, gae=0.95 (ablate lower discount) ─────────────────────
    dict(label='Z03_gamma99',     gamma=0.99, gae_lambda=0.95),

    # ── Z04: arch=medium (ablate wide network) ─────────────────────────────────
    dict(label='Z04_medium',      agent_net_arch='medium'),

    # ── Z05: ep_for_train=5 (ablate ep=3) ─────────────────────────────────────
    dict(label='Z05_ep5',         ep_for_train=5),

    # ── Z06: agent_lr=1e-3 (ablate agent_lr down from 3e-3) ───────────────────
    dict(label='Z06_lr1e3',       agent_lr=1e-3),

    # ── Z07: ent=0.05 (ablate entropy) ────────────────────────────────────────
    dict(label='Z07_ent05',       ent_coef=0.05),

    # ── Z08: n_ppo_epochs=1 (ablate multi-epoch PPO) ──────────────────────────
    dict(label='Z08_epochs1',     n_ppo_epochs=1),

    # ── Z09: original step rewards (starvation=-5) ────────────────────────────
    dict(label='Z09_starv',       starvation_reward=-5, high_speed_reward=5),

    # ── Z10: clip=0.2 (ablate tighter clipping) ───────────────────────────────
    dict(label='Z10_clip02',      clip_range=0.2),
]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    patch_intersection_env()
    register_intersection_env()

    ts         = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')
    sweep_root = os.path.join('experiments', f'sweep2_{ts}')
    os.makedirs(sweep_root, exist_ok=True)
    print(f"\n{'#'*70}")
    print(f"  SWEEP-2  root: {sweep_root}")
    print(f"  {len(SWEEP_CONFIGS)} configs × 1200 episodes each")
    print(f"{'#'*70}\n")

    all_results = _reconstruct_all_results(sweep_root, SWEEP_CONFIGS)
    completed   = set(all_results.keys())

    for i, cfg in enumerate(SWEEP_CONFIGS, 1):
        lbl = cfg['label']
        if lbl in completed:
            print(f"[{i}/{len(SWEEP_CONFIGS)}] SKIP (already done): {lbl}")
            continue

        print(f"[{i}/{len(SWEEP_CONFIGS)}]")
        try:
            collisions, results = run_config(cfg, sweep_root)
        except Exception as exc:
            import traceback
            print(f"\n[ERROR] Config {lbl} failed: {exc}")
            traceback.print_exc()
            results    = {'arrival_rates': [], 'episode_rewards': [],
                          'master_policy_losses': [], 'master_value_losses': [],
                          'master_total_losses': [],
                          'agent_group0_total_losses': [], 'agent_group1_total_losses': []}
            collisions = -1

        results['total_collisions'] = collisions
        all_results[lbl] = results
        save_incremental_outputs(sweep_root, all_results, SWEEP_CONFIGS)

    print(f"\n{'#'*70}")
    print(f"  SWEEP-2 complete.  Results: {sweep_root}")
    print(f"{'#'*70}")
