"""
Full exhaustive hyperparameter grid search — one sweep to find where
both the Master AND the Agents learn.

Folder layout (created automatically at run-start):
  experiments/
    sweep_DD_MM_YYYY-HH_MM_SS/          ← one folder per sweep invocation
      grid_summary.json                  ← updated after every config
      grid_summary.csv                   ← same data as CSV
      grid_ranking.png                   ← updated after every config
      grid_section_A.png … _J.png        ← updated after every config in that section
      A/                                 ← one subfolder per section letter
        A_e001_c300/
          agent_logs/
          master_logs/
          trained_model_agent.pth
          trained_model_master.pth
        A_e001_c100/
          ...
      B/  C/  D/ …  J/
"""

import csv
import logging
import os
import json
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from highwayenv.utils import patch_intersection_env, register_intersection_env
from src import project_globals
from src.experiment import scenarios_config as sc
from src.experiment.experiment_config import Experiment
from src.training.training_handler import training_loop
from src.model.model_handler import save_models
from src.training.general_utils import initialize_models, setup_experiment_dirs, setup_loggers

logging.basicConfig(level=logging.WARNING)   # suppress per-episode spam during sweep

# ── helpers ────────────────────────────────────────────────────────────────────

def _smooth(values, window=40):
    arr = np.array([v if v is not None else np.nan for v in values], dtype=float)
    if len(arr) < window:
        window = max(1, len(arr))
    kernel  = np.ones(window) / window
    pad     = window // 2
    padded  = np.concatenate([np.full(pad, np.nan), arr, np.full(pad, np.nan)])
    s = np.convolve(np.where(np.isnan(padded), 0, padded), kernel, mode='valid')
    c = np.convolve((~np.isnan(padded)).astype(float), kernel, mode='valid')
    s = s / np.where(c > 0, c, 1)
    s[c == 0] = np.nan
    return s[:len(arr)]


# ── single config runner ───────────────────────────────────────────────────────

def run_config(cfg: dict, sweep_root: str):
    """
    Run one grid configuration.  cfg keys (all optional, use defaults where absent):
      label, ent_coef, collision_reward, arrived_reward,
      reward_mode, cotrain_cycles, agent_lr, master_lr
    Returns (collision_count, results_dict).
    """
    label    = cfg['label']
    ent      = cfg.get('ent_coef',         0.01)
    coll_r   = cfg.get('collision_reward', -300)
    arr_r    = cfg.get('arrived_reward',    50)
    mode     = cfg.get('reward_mode',      'global')
    cotrain  = cfg.get('cotrain_cycles',   True)
    agent_lr = cfg.get('agent_lr',         7e-4)
    master_lr= cfg.get('master_lr',        1e-4)

    print(f"\n{'='*70}")
    print(f"  CONFIG: {label}")
    print(f"  ent={ent}  coll={coll_r}  arr={arr_r}  mode={mode}  cotrain={cotrain}")
    print(f"  clip={cfg.get('clip_range',0.2)}  gamma={cfg.get('gamma',0.99)}"
          f"  gae={cfg.get('gae_lambda',0.95)}  arch={cfg.get('agent_net_arch','small')}"
          f"  ep4train={cfg.get('ep_for_train',5)}")
    print(f"  agent_lr={agent_lr:.2e}  master_lr={master_lr:.2e}"
          f"  vf_coef={cfg.get('vf_coef',0.5)}  n_ppo_epochs={cfg.get('n_ppo_epochs',1)}")
    print(f"  starvation={cfg.get('starvation_reward',-5)}  high_speed={cfg.get('high_speed_reward',5)}")
    print(f"{'='*70}\n")

    project_globals.reset_globals()

    starvation_r = cfg.get('starvation_reward', -5)
    high_speed_r = cfg.get('high_speed_reward',  5)

    # ── Build paths inside sweep_root/SECTION/label/ ────────────────────────────
    section = label.split('_')[0]                              # e.g. "A", "B", …
    exp_path = os.path.join(sweep_root, section, label)        # e.g. sweep_.../A/A_e001_c300

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
        CLIP_RANGE=cfg.get('clip_range',          0.2),
        GAMMA=cfg.get('gamma',                    0.99),
        GAE_LAMBDA=cfg.get('gae_lambda',          0.95),
        AGENT_NET_ARCH=cfg.get('agent_net_arch',  'small'),
        EPISODE_AMOUNT_FOR_TRAIN=cfg.get('ep_for_train', 5),
        VF_COEF=cfg.get('vf_coef',                0.5),
        N_PPO_EPOCHS=cfg.get('n_ppo_epochs',       1),
    )
    # Override paths to use the sweep root instead of top-level experiments/
    exp.EXPERIMENT_PATH    = exp_path
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

    print(f"\n  [{label}] done — collisions={collisions}  "
          f"avg_arrival={np.nanmean(results['arrival_rates']):.1f}%")
    return collisions, results


# ── plotting ───────────────────────────────────────────────────────────────────

# 50-colour palette (tab20 × 2.5) for large grids
import matplotlib.cm as _cm
_tab20  = [_cm.tab20(i) for i in np.linspace(0, 1, 20, endpoint=False)]
_tab20b = [_cm.tab20b(i) for i in np.linspace(0, 1, 20, endpoint=False)]
_tab20c = [_cm.tab20c(i) for i in np.linspace(0, 1, 20, endpoint=False)]
PALETTE = _tab20 + _tab20b + _tab20c   # 60 distinct colours


def _section_plot(section_label, cfgs, all_results, out_path):
    """Arrival-rate + reward comparison for one section."""
    present = [c for c in cfgs if c['label'] in all_results]
    if not present:
        return
    fig, (ax_arr, ax_rew) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f'Section {section_label} — Arrival Rate & Reward',
                 fontsize=13, fontweight='bold')
    for i, cfg in enumerate(present):
        lbl = cfg['label']
        collisions, res = all_results[lbl]
        color = PALETTE[i % len(PALETTE)]
        n = len(res['arrival_rates'])
        x = np.arange(1, n + 1)
        for ax, key, ylabel in [
            (ax_arr, 'arrival_rates',   'Arrival Rate (%)'),
            (ax_rew, 'episode_rewards', 'Episode Reward'),
        ]:
            vals = res.get(key, [])
            raw  = np.array([v if v is not None else np.nan for v in vals], dtype=float)
            ax.plot(x[:len(raw)], raw, color=color, alpha=0.12, linewidth=0.7)
            ax.plot(x[:len(raw)], _smooth(vals), color=color, linewidth=2.0,
                    label=f"{lbl} ({collisions}col)")
    for ax, ttl, yl in [(ax_arr, 'Arrival Rate (%)', '%'),
                         (ax_rew, 'Episode Reward',  'Reward')]:
        ax.set_title(ttl, fontsize=11); ax.set_xlabel('Episode', fontsize=9)
        ax.set_ylabel(yl, fontsize=9);  ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, ncol=max(1, len(present)//5))
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def _ranking_plot(all_results, sweep_configs, out_path):
    """Horizontal bar chart sorted by final-50-ep arrival rate."""
    rows = []
    for cfg in sweep_configs:
        lbl = cfg['label']
        if lbl not in all_results:
            continue
        collisions, res = all_results[lbl]
        final = float(np.nanmean(res['arrival_rates'][-50:]))
        avg   = float(np.nanmean(res['arrival_rates']))
        rows.append((lbl, final, avg, collisions))
    rows.sort(key=lambda r: r[1], reverse=True)

    n = len(rows)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.32)))
    labels_r = [r[0] for r in rows]
    finals   = [r[1] for r in rows]
    avgs     = [r[2] for r in rows]
    colors   = [PALETTE[i % len(PALETTE)] for i in range(n)]
    y = np.arange(n)
    ax.barh(y, finals, color=colors, height=0.6, label='Last-50-ep avg%')
    ax.barh(y, avgs,   color=colors, height=0.3, alpha=0.4, label='Overall avg%')
    for j, (lbl, fin, avg, col) in enumerate(rows):
        ax.text(fin + 0.5, j, f'{fin:.1f}%  ({col} crashes)', va='center', fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels(labels_r, fontsize=7)
    ax.set_xlabel('Arrival Rate (%)', fontsize=10)
    ax.set_title('Grid Search Ranking — sorted by final arrival rate', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.25, axis='x')
    ax.set_xlim(0, 115)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()
    print(f"  Saved {out_path}")


def save_incremental_outputs(sweep_root: str, all_results: dict, sweep_configs: list):
    """
    Called after every config completes.  Writes / overwrites:
      sweep_root/grid_summary.json
      sweep_root/grid_summary.csv
      sweep_root/grid_ranking.png
      sweep_root/grid_section_<X>.png   (only sections with ≥1 completed config)
    """
    os.makedirs(sweep_root, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────────────────
    summary = {}
    for cfg in sweep_configs:
        lbl = cfg['label']
        if lbl not in all_results:
            continue
        col, res = all_results[lbl]
        rates = res.get('arrival_rates', [])

        def _nanmean_last(lst, n=50):
            arr = [v for v in lst if v is not None]
            if not arr: return None
            return float(np.nanmean(arr[-n:]))

        def _nanmean_all(lst):
            arr = [v for v in lst if v is not None]
            return float(np.nanmean(arr)) if arr else None

        summary[lbl] = {
            'section':        lbl.split('_')[0],
            'collisions':     col,
            'avg_arrival':    float(np.nanmean(rates))       if rates else 0.0,
            'final_arrival':  float(np.nanmean(rates[-50:])) if len(rates) >= 50 else
                              float(np.nanmean(rates))       if rates else 0.0,
            # ── Loss statistics ──────────────────────────────────────────────
            'master_policy_loss_avg':  _nanmean_all(res.get('master_policy_losses', [])),
            'master_policy_loss_last': _nanmean_last(res.get('master_policy_losses', [])),
            'master_value_loss_avg':   _nanmean_all(res.get('master_value_losses',  [])),
            'master_value_loss_last':  _nanmean_last(res.get('master_value_losses',  [])),
            'agent_g0_loss_avg':       _nanmean_all(res.get('agent_group0_total_losses', [])),
            'agent_g0_loss_last':      _nanmean_last(res.get('agent_group0_total_losses', [])),
            'agent_g1_loss_avg':       _nanmean_all(res.get('agent_group1_total_losses', [])),
            'agent_g1_loss_last':      _nanmean_last(res.get('agent_group1_total_losses', [])),
            # ─────────────────────────────────────────────────────────────────
            'cfg': {k: v for k, v in cfg.items() if k != 'label'},
        }
    with open(os.path.join(sweep_root, 'grid_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    # ── CSV ───────────────────────────────────────────────────────────────────
    csv_path = os.path.join(sweep_root, 'grid_summary.csv')
    all_keys = sorted({k for d in summary.values() for k in d.get('cfg', {})})
    _loss_cols = ['master_policy_loss_avg', 'master_policy_loss_last',
                  'master_value_loss_avg',  'master_value_loss_last',
                  'agent_g0_loss_avg',      'agent_g0_loss_last',
                  'agent_g1_loss_avg',      'agent_g1_loss_last']
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['label', 'section', 'avg_arrival_%', 'final_arrival_%',
                  'collisions'] + _loss_cols + all_keys
        writer.writerow(header)
        for lbl, d in summary.items():
            def _fmt(v): return f"{v:.5f}" if v is not None else ''
            row = [lbl, d['section'],
                   f"{d['avg_arrival']:.2f}", f"{d['final_arrival']:.2f}",
                   d['collisions']]
            row += [_fmt(d.get(c)) for c in _loss_cols]
            for k in all_keys:
                row.append(d['cfg'].get(k, ''))
            writer.writerow(row)

    # ── Plots ─────────────────────────────────────────────────────────────────
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


def save_plots(all_results, sweep_configs, sweep_root='experiments'):
    """Legacy wrapper kept for compatibility."""
    save_incremental_outputs(sweep_root, all_results, sweep_configs)


def save_config_plots(results: dict, exp_path: str, label: str):
    """
    Save a self-contained 4-panel plot for a single completed config inside
    its own experiment folder.  Panels:
      [0] Arrival / success rate over episodes
      [1] Episode reward over episodes
      [2] Master losses (policy, value, total) over training calls
      [3] Agent losses (group-0, group-1) over training calls
    """
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Config: {label}', fontsize=14, fontweight='bold')
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.28)

    ax_arr  = fig.add_subplot(gs[0, 0])
    ax_rew  = fig.add_subplot(gs[0, 1])
    ax_mst  = fig.add_subplot(gs[1, 0])
    ax_agt  = fig.add_subplot(gs[1, 1])

    def _plot_series(ax, vals, color, label_str, alpha_raw=0.15):
        if not vals:
            return
        raw = np.array([v if v is not None else np.nan for v in vals], dtype=float)
        x   = np.arange(1, len(raw) + 1)
        ax.plot(x, raw, color=color, alpha=alpha_raw, linewidth=0.6)
        ax.plot(x, _smooth(vals), color=color, linewidth=2.0, label=label_str)

    # Arrival rate
    _plot_series(ax_arr, results.get('arrival_rates', []), '#2196F3', 'Arrival %')
    ax_arr.set_title('Success / Arrival Rate', fontsize=11)
    ax_arr.set_xlabel('Episode'); ax_arr.set_ylabel('%')
    ax_arr.set_ylim(-5, 105); ax_arr.grid(True, alpha=0.25); ax_arr.legend(fontsize=8)

    # Episode reward
    _plot_series(ax_rew, results.get('episode_rewards', []), '#4CAF50', 'Reward')
    ax_rew.set_title('Episode Reward', fontsize=11)
    ax_rew.set_xlabel('Episode'); ax_rew.set_ylabel('Reward')
    ax_rew.grid(True, alpha=0.25); ax_rew.legend(fontsize=8)

    # Master losses
    colors_mst = {'policy': '#E53935', 'value': '#FB8C00', 'total': '#8E24AA'}
    for key, col, lbl_s in [
        ('master_policy_losses', colors_mst['policy'], 'Master Policy'),
        ('master_value_losses',  colors_mst['value'],  'Master Value'),
        ('master_total_losses',  colors_mst['total'],  'Master Total'),
    ]:
        _plot_series(ax_mst, results.get(key, []), col, lbl_s, alpha_raw=0.2)
    ax_mst.set_title('Master Losses (per training call)', fontsize=11)
    ax_mst.set_xlabel('Training call'); ax_mst.set_ylabel('Loss')
    ax_mst.grid(True, alpha=0.25); ax_mst.legend(fontsize=8)

    # Agent losses
    for key, col, lbl_s in [
        ('agent_group0_total_losses', '#00ACC1', 'Agent G0 (cars 0-2)'),
        ('agent_group1_total_losses', '#43A047', 'Agent G1 (cars 3-5)'),
    ]:
        _plot_series(ax_agt, results.get(key, []), col, lbl_s, alpha_raw=0.2)
    ax_agt.set_title('Agent Losses (per training call)', fontsize=11)
    ax_agt.set_xlabel('Training call'); ax_agt.set_ylabel('Loss')
    ax_agt.grid(True, alpha=0.25); ax_agt.legend(fontsize=8)

    out = os.path.join(exp_path, 'results.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"  [plot] {out}")


# ── grid definition ────────────────────────────────────────────────────────────
#
# BASE: ent=0.05, coll=-100, arrive=50, global, cotrain=True,
#       clip=0.2, gamma=0.99, gae=0.95, arch='small',
#       agent_lr=7e-4, master_lr=1e-4, ep_for_train=5
# Each section varies ONE axis; all other axes stay at BASE.

SWEEP_CONFIGS = [

    # ══════════════════════════════════════════════════════════════════════════════
    # A: Core — ent_coef × collision_reward  (5 × 4 = 20 configs)
    #    Tests the two most critical levers simultaneously.
    # ══════════════════════════════════════════════════════════════════════════════
    # ent=0.001 (almost deterministic)
    dict(label='A_e001_c300', ent_coef=0.001, collision_reward=-300, arrived_reward=50),
    dict(label='A_e001_c100', ent_coef=0.001, collision_reward=-100, arrived_reward=50),
    dict(label='A_e001_c50',  ent_coef=0.001, collision_reward=-50,  arrived_reward=50),
    dict(label='A_e001_c20',  ent_coef=0.001, collision_reward=-20,  arrived_reward=20),

    # ent=0.01
    dict(label='A_e01_c300',  ent_coef=0.01,  collision_reward=-300, arrived_reward=50),
    dict(label='A_e01_c100',  ent_coef=0.01,  collision_reward=-100, arrived_reward=50),
    dict(label='A_e01_c50',   ent_coef=0.01,  collision_reward=-50,  arrived_reward=50),
    dict(label='A_e01_c20',   ent_coef=0.01,  collision_reward=-20,  arrived_reward=20),

    # ent=0.05  ← BASE row
    dict(label='A_e05_c300',  ent_coef=0.05,  collision_reward=-300, arrived_reward=50),
    dict(label='A_e05_c100',  ent_coef=0.05,  collision_reward=-100, arrived_reward=50),  # BASE
    dict(label='A_e05_c50',   ent_coef=0.05,  collision_reward=-50,  arrived_reward=50),
    dict(label='A_e05_c20',   ent_coef=0.05,  collision_reward=-20,  arrived_reward=20),

    # ent=0.10
    dict(label='A_e10_c300',  ent_coef=0.10,  collision_reward=-300, arrived_reward=50),
    dict(label='A_e10_c100',  ent_coef=0.10,  collision_reward=-100, arrived_reward=50),
    dict(label='A_e10_c50',   ent_coef=0.10,  collision_reward=-50,  arrived_reward=50),
    dict(label='A_e10_c20',   ent_coef=0.10,  collision_reward=-20,  arrived_reward=20),

    # ent=0.20  (very exploratory)
    dict(label='A_e20_c300',  ent_coef=0.20,  collision_reward=-300, arrived_reward=50),
    dict(label='A_e20_c100',  ent_coef=0.20,  collision_reward=-100, arrived_reward=50),
    dict(label='A_e20_c50',   ent_coef=0.20,  collision_reward=-50,  arrived_reward=50),
    dict(label='A_e20_c20',   ent_coef=0.20,  collision_reward=-20,  arrived_reward=20),

    # ══════════════════════════════════════════════════════════════════════════════
    # B: Agent learning rate  (base: ent=0.05, coll=-100)
    # ══════════════════════════════════════════════════════════════════════════════
    dict(label='B_aLR_1e4',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         agent_lr=1e-4),
    dict(label='B_aLR_3e4',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         agent_lr=3e-4),
    # 7e-4 = BASE, already in A_e05_c100
    dict(label='B_aLR_3e3',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         agent_lr=3e-3),
    dict(label='B_aLR_1e2',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         agent_lr=1e-2),

    # ══════════════════════════════════════════════════════════════════════════════
    # C: Master learning rate  (base: ent=0.05, coll=-100)
    # ══════════════════════════════════════════════════════════════════════════════
    dict(label='C_mLR_1e5',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         master_lr=1e-5),
    dict(label='C_mLR_3e4',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         master_lr=3e-4),
    dict(label='C_mLR_1e3',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         master_lr=1e-3),

    # ══════════════════════════════════════════════════════════════════════════════
    # D: PPO clip_range  (base: ent=0.05, coll=-100)
    # ══════════════════════════════════════════════════════════════════════════════
    dict(label='D_clip005', ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         clip_range=0.05),
    dict(label='D_clip01',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         clip_range=0.1),
    # clip=0.2 = BASE (A_e05_c100)
    dict(label='D_clip04',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         clip_range=0.4),

    # ══════════════════════════════════════════════════════════════════════════════
    # E: Discount factor γ + GAE λ combos  (base: ent=0.05, coll=-100)
    # ══════════════════════════════════════════════════════════════════════════════
    dict(label='E_g090_l090', ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         gamma=0.90, gae_lambda=0.90),
    dict(label='E_g095_l095', ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         gamma=0.95, gae_lambda=0.95),
    dict(label='E_g099_l099', ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         gamma=0.99, gae_lambda=0.99),
    # γ=0.99, λ=0.95 = BASE

    # ══════════════════════════════════════════════════════════════════════════════
    # F: Agent network architecture  (base: ent=0.05, coll=-100)
    # ══════════════════════════════════════════════════════════════════════════════
    dict(label='F_tiny',    ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         agent_net_arch='tiny'),
    # 'small' = BASE
    dict(label='F_medium',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         agent_net_arch='medium'),
    dict(label='F_large',   ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         agent_net_arch='large'),
    dict(label='F_deep',    ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         agent_net_arch='deep'),
    dict(label='F_wide',    ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         agent_net_arch='wide'),

    # ══════════════════════════════════════════════════════════════════════════════
    # G: Episode update frequency  (base: ep_for_train=5)
    # ══════════════════════════════════════════════════════════════════════════════
    dict(label='G_ep3',  ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         ep_for_train=3),
    dict(label='G_ep10', ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         ep_for_train=10),

    # ══════════════════════════════════════════════════════════════════════════════
    # H: Reward-mode and cycle-structure variants  (best guesses at all combos)
    # ══════════════════════════════════════════════════════════════════════════════
    dict(label='H_group',       ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         reward_mode='group'),
    dict(label='H_nocot',       ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         cotrain_cycles=False),
    dict(label='H_grp_nocot',   ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         reward_mode='group', cotrain_cycles=False),
    dict(label='H_grp_c50',     ent_coef=0.05, collision_reward=-50,  arrived_reward=50,
         reward_mode='group'),

    # ══════════════════════════════════════════════════════════════════════════════
    # I: PPO epochs per update — standard PPO does 10; we default to 1.
    #    More epochs = reuse data more → faster learning OR instability.
    #    Also sweeps vf_coef (how hard value function is trained).
    # ══════════════════════════════════════════════════════════════════════════════
    dict(label='I_ep3',     ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         n_ppo_epochs=3),
    dict(label='I_ep5',     ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         n_ppo_epochs=5),
    dict(label='I_ep10',    ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         n_ppo_epochs=10),
    dict(label='I_vf025',   ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         vf_coef=0.25),
    dict(label='I_vf10',    ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         vf_coef=1.0),
    dict(label='I_ep5_vf025', ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         n_ppo_epochs=5, vf_coef=0.25),

    # ══════════════════════════════════════════════════════════════════════════════
    # J: Per-step reward shape — starvation_reward and high_speed_reward.
    #    Baseline: starvation=-5, high_speed=+5 (step-level tug-of-war).
    #    J1: zero step rewards → pure terminal signal (arrive/crash only).
    #    J2: small negative step → mild pressure to be efficient.
    #    J3: high_speed=0 → no bonus for speed (remove speed incentive).
    #    J4: starvation=0, high_speed=+10 → reward ONLY fast movement, not punish slow.
    # ══════════════════════════════════════════════════════════════════════════════
    dict(label='J_no_step',    ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         starvation_reward=0, high_speed_reward=0),
    dict(label='J_mild_stv',   ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         starvation_reward=-1, high_speed_reward=0),
    dict(label='J_no_speed',   ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         starvation_reward=-5, high_speed_reward=0),
    dict(label='J_spd_only',   ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         starvation_reward=0,  high_speed_reward=10),
    dict(label='J_hard_stv',   ent_coef=0.05, collision_reward=-100, arrived_reward=50,
         starvation_reward=-10, high_speed_reward=5),
]


# ── main ───────────────────────────────────────────────────────────────────────

def _reconstruct_all_results(summary_dict: dict, all_results: dict):
    """Fill all_results from a saved summary JSON (used on resume)."""
    _N = 1200
    for _lbl, _d in summary_dict.items():
        if _lbl in all_results:
            continue
        _avg = float(_d.get('avg_arrival', 0.0))
        _fin = float(_d.get('final_arrival', _avg))
        _rates = [_avg] * (_N - 50) + [_fin] * 50
        all_results[_lbl] = (
            _d.get('collisions', 0),
            {'arrival_rates':             _rates,
             'episode_rewards':           [0.0] * _N,
             'master_policy_losses':      [],
             'master_value_losses':       [],
             'master_total_losses':       [],
             'agent_group0_total_losses': [],
             'agent_group1_total_losses': [],
             'agent_policy_losses':       [],
             'agent_value_losses':        [],
             'agent_total_losses':        [],
             'all_actions':               []},
        )


if __name__ == '__main__':
    patch_intersection_env()
    register_intersection_env()

    # ── Create (or resume into) a single sweep root folder ───────────────────
    # On resume, look for the most recent sweep_* folder that has a summary JSON.
    _sweep_dirs = sorted([
        d for d in (os.listdir('experiments') if os.path.isdir('experiments') else [])
        if d.startswith('sweep_') and
        os.path.exists(os.path.join('experiments', d, 'grid_summary.json'))
    ])
    if _sweep_dirs:
        SWEEP_ROOT = os.path.join('experiments', _sweep_dirs[-1])
        print(f"  Resuming into existing sweep folder: {SWEEP_ROOT}")
    else:
        _ts = datetime.now().strftime('%d_%m_%Y-%H_%M_%S')
        SWEEP_ROOT = os.path.join('experiments', f'sweep_{_ts}')
        os.makedirs(SWEEP_ROOT, exist_ok=True)
        print(f"  New sweep folder: {SWEEP_ROOT}")

    # ── Load completed configs from this sweep's summary ─────────────────────
    _summary_path = os.path.join(SWEEP_ROOT, 'grid_summary.json')
    all_results: dict = {}
    _completed_labels: set = set()
    if os.path.exists(_summary_path):
        try:
            with open(_summary_path) as _f:
                _prev = json.load(_f)
            _completed_labels = set(_prev.keys())
            _reconstruct_all_results(_prev, all_results)
            print(f"  {len(_completed_labels)} configs already done, skipping.")
        except Exception as exc:
            print(f"  Warning: could not load previous summary: {exc}")

    total   = len(SWEEP_CONFIGS)
    pending = [c for c in SWEEP_CONFIGS if c['label'] not in _completed_labels]
    print(f"\n{'='*60}")
    print(f"  Grid: {total} configs  ({len(pending)} remaining)  × 1200 episodes")
    print(f"  Estimated remaining: {len(pending)*20//60}h {len(pending)*20%60}min")
    print(f"  Outputs → {SWEEP_ROOT}/")
    print(f"{'='*60}\n")

    for i, cfg in enumerate(pending, 1):
        print(f"\n[{i}/{len(pending)}] ", end='')
        collisions, results = run_config(cfg, SWEEP_ROOT)
        all_results[cfg['label']] = (collisions, results)
        # Update JSON + CSV + all plots immediately after each config
        save_incremental_outputs(SWEEP_ROOT, all_results, SWEEP_CONFIGS)

    # ── Final ranked summary table ─────────────────────────────────────────────
    COL = 115
    print("\n" + "="*COL)
    print(f"{'GRID SEARCH — FINAL RANKING (sorted by last-50-ep arrival)':^{COL}}")
    print("="*COL)
    rows = []
    for cfg in SWEEP_CONFIGS:
        lbl = cfg['label']
        if lbl not in all_results:
            continue
        col, res = all_results[lbl]
        rows.append((
            lbl,
            float(np.nanmean(res['arrival_rates'][-50:])),
            float(np.nanmean(res['arrival_rates'])),
            col,
            cfg,
        ))
    rows.sort(key=lambda r: r[1], reverse=True)

    print(f"  {'#':>3}  {'Config':22s}  {'ent':>5}  {'coll':>5}  {'clip':>5}  "
          f"{'γ':>5}  {'arch':>7}  {'aLR':>7}  {'ep4t':>4}  "
          f"{'mode':>6}  {'cot':>3}  {'#col':>5}  {'avg%':>6}  {'fin%':>6}")
    print("-"*COL)
    for rank, (lbl, fin, avg, col, cfg) in enumerate(rows, 1):
        cot = 'Y' if cfg.get('cotrain_cycles', True) else 'N'
        print(
            f"  {rank:>3}  {lbl:22s}  {cfg.get('ent_coef',0.05):>5.3f}  "
            f"{cfg.get('collision_reward',-100):>5d}  "
            f"{cfg.get('clip_range',0.2):>5.2f}  "
            f"{cfg.get('gamma',0.99):>5.2f}  "
            f"{cfg.get('agent_net_arch','small'):>7s}  "
            f"{cfg.get('agent_lr',7e-4):>7.1e}  "
            f"{cfg.get('ep_for_train',5):>4d}  "
            f"{cfg.get('reward_mode','global'):>6s}  {cot:>3}  "
            f"{col:>5d}  {avg:>6.1f}  {fin:>6.1f}"
        )
    print("="*COL)
    if rows:
        print(f"\n  Best config: {rows[0][0]}  →  final arrival {rows[0][1]:.1f}%")
    else:
        print("\n  No completed configs to rank.")

    print(f"\nAll done. Results in → {SWEEP_ROOT}/")
    print(f"  grid_summary.json / grid_summary.csv  — full numeric table")
    print(f"  grid_ranking.png                       — sorted bar chart")
    print(f"  grid_section_A.png … grid_section_J.png — per-section plots")
