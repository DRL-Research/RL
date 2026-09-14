from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from hdmaps.corridor import Corridor
from hdmaps.paths import agent_checkpoint, master_checkpoint, repo_root
from hdmaps.policies.master import load_models
from hdmaps.protocol import (
    SIM_CONNECTOR_M,
    SIM_DURATION,
    SIM_MAX_STEPS,
    TRAIN_CHECKPOINT_EVERY,
    TRAIN_EPISODES,
    TRAIN_MASTER_LR,
    TRAIN_MAX_AGENTS,
    TRAIN_N_INT,
    TRAIN_SEED,
    TRAIN_STAGE1_EPISODES,
    TRAIN_STAGE1_LR,
    TRAIN_STAGE1_SEED,
)
from hdmaps.scenarios.regional import generate_scenario, generate_single_intersection
from hdmaps.training.ppo import update_master
from hdmaps.training.rollout import run_training_episode


def sample_stage1(rng: np.random.Generator) -> dict:
    n_agents = int(rng.choice([1, 2, 3, 3, 4, 4]))
    mixed = bool(n_agents >= 3 and rng.random() < 0.5)
    sc = generate_single_intersection(n_agents, rng, mixed=mixed)
    sc['connector_length'] = SIM_CONNECTOR_M
    return {
        'n_int': 1,
        'n_agents': n_agents,
        'connector': SIM_CONNECTOR_M,
        'mixed': mixed,
        'scenario': sc,
    }


def sample_stage2(rng: np.random.Generator) -> dict:
    n_int = TRAIN_N_INT
    n_agents = int(rng.choice([4, 6, 6, 6]))
    connector = SIM_CONNECTOR_M
    sc = generate_scenario(n_int, n_agents, rng, varied=True)
    sc['connector_length'] = connector
    return {
        'n_int': n_int,
        'n_agents': n_agents,
        'connector': connector,
        'agents_per_int': n_agents // n_int,
        'scenario': sc,
    }


def _save_master(master, dest: Path, sidecar_src: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    master.save(str(dest / 'master.pth'))
    if sidecar_src.is_file():
        (dest / 'master_custom_params.pt').write_bytes(sidecar_src.read_bytes())


def _write_summary(dest: Path, summary: dict) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'train_log.json').write_text(json.dumps(summary, indent=2, default=float), encoding='utf-8')


def _load_summary(dest: Path) -> dict:
    path = dest / 'train_log.json'
    return json.loads(path.read_text(encoding='utf-8'))


def _stage_complete(dest: Path, n_episodes: int) -> bool:
    if not (dest / 'master.pth').is_file() or not (dest / 'train_log.json').is_file():
        return False
    try:
        data = _load_summary(dest)
    except Exception:
        return False
    return int(data.get('episodes', 0)) >= int(n_episodes)


def _latest_checkpoint(dest: Path) -> tuple[int, Path] | None:
    found: list[tuple[int, Path]] = []
    for path in dest.glob('master_ep*.pth'):
        try:
            found.append((int(path.stem.split('ep')[-1]), path))
        except ValueError:
            continue
    if not found:
        return None
    return max(found, key=lambda item: item[0])


def _save_rng(rng: np.random.Generator, dest: Path, episode: int) -> None:
    (dest / ('rng_ep%04d.json' % episode)).write_text(
        json.dumps(rng.bit_generator.state),
        encoding='utf-8',
    )


def _load_rng(rng: np.random.Generator, dest: Path, episode: int) -> None:
    path = dest / ('rng_ep%04d.json' % episode)
    if path.is_file():
        rng.bit_generator.state = json.loads(path.read_text(encoding='utf-8'))


def _history_within_cap(dest: Path) -> bool:
    paths = [dest / 'history.jsonl', dest / 'history_incomplete.jsonl']
    for path in paths:
        if not path.is_file():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if int(row.get('n_int', 0)) > TRAIN_N_INT or int(row.get('n_agents', 0)) > TRAIN_MAX_AGENTS:
                return False
    return True


def archive_overscale_stage(dest: Path) -> Path | None:
    dest = Path(dest)
    if not dest.exists() or _history_within_cap(dest):
        return None
    bak = dest.with_name(dest.name + '_overscale')
    n = 1
    while bak.exists():
        n += 1
        bak = dest.with_name('%s_overscale_%d' % (dest.name, n))
    dest.rename(bak)
    print('[pipeline] archived overscale %s -> %s' % (dest.name, bak.name), flush=True)
    return bak


def _truncate_history(history_path: Path, n_keep: int) -> None:
    if n_keep <= 0 or not history_path.is_file():
        if history_path.is_file():
            history_path.unlink()
        return
    lines = [line for line in history_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    history_path.write_text('\n'.join(lines[:n_keep]) + ('\n' if lines[:n_keep] else ''), encoding='utf-8')


def train_loop(
    *,
    n_episodes: int,
    seed: int,
    lr: float,
    sample_fn,
    agent_path: Path,
    master_path: Path,
    dest: Path,
    smoke: bool = False,
    label: str = 'train',
) -> dict:
    dest = Path(dest)
    if not smoke:
        archive_overscale_stage(dest)
    dest.mkdir(parents=True, exist_ok=True)
    history_path = dest / 'history.jsonl'
    rng = np.random.default_rng(seed)
    start_ep = 0
    load_master = Path(master_path)
    resume = None if smoke else _latest_checkpoint(dest)
    if resume is not None and resume[0] < n_episodes:
        start_ep, ckpt = resume
        load_master = ckpt
        _load_rng(rng, dest, start_ep)
        _truncate_history(history_path, start_ep)
        print('[%s] resume from %s (episode %d)' % (label, ckpt.name, start_ep), flush=True)
    elif history_path.is_file() and resume is None:
        bak = dest / 'history_incomplete.jsonl'
        history_path.replace(bak)
        print('[%s] archived incomplete history to %s' % (label, bak.name), flush=True)

    master, agent = load_models(str(agent_path), str(load_master))
    for param_group in master.model.policy.optimizer.param_groups:
        param_group['lr'] = lr
    sidecar_src = Path(master_path).parent / 'master_custom_params.pt'
    if not sidecar_src.is_file():
        sidecar_src = dest / 'master_custom_params.pt'
    cache: dict[tuple[int, int, int], Corridor] = {}
    t0 = time.time()

    for ep in range(start_ep, n_episodes):
        if smoke:
            spec = {
                'n_int': 1,
                'n_agents': 3,
                'connector': 150,
                'mixed': False,
                'agents_per_int': 3,
                'scenario': generate_single_intersection(3, rng, mixed=False),
            }
            spec['scenario']['connector_length'] = 150
        else:
            spec = sample_fn(rng)
        n_int = int(spec['n_int'])
        n_agents = int(spec['n_agents'])
        connector = int(spec['connector'])
        if (not smoke) and (n_int > TRAIN_N_INT or n_agents > TRAIN_MAX_AGENTS):
            raise ValueError(
                'training cap is %d intersections / %d agents; got n_int=%d n_agents=%d'
                % (TRAIN_N_INT, TRAIN_MAX_AGENTS, n_int, n_agents)
            )
        key = (n_int, n_agents, connector)
        if key not in cache:
            while len(cache) >= 4:
                cache.pop(next(iter(cache))).close()
            cache[key] = Corridor(n_int, n_agents, connector_length=connector, duration=SIM_DURATION)
        cell = cache[key]
        result, streams = run_training_episode(cell, spec['scenario'], master, agent, max_steps=SIM_MAX_STEPS)
        losses = []
        for stream in streams:
            info = update_master(master, stream)
            if info is not None:
                losses.append(info)
        row = {
            'episode': ep + 1,
            'n_int': n_int,
            'n_agents': n_agents,
            'connector': connector,
            'steps': result['steps'],
            'crashed': bool(result['crashed']),
            'arrived': int(result['arrived']),
            'reward': float(result['reward']),
            'n_updates': len(losses),
        }
        if 'mixed' in spec:
            row['mixed'] = bool(spec['mixed'])
        if 'agents_per_int' in spec:
            row['agents_per_int'] = int(spec['agents_per_int'])
        with history_path.open('a', encoding='utf-8') as fh:
            fh.write(json.dumps(row, default=float) + '\n')
        if (ep + 1) % 10 == 0 or ep == start_ep or (ep + 1) == n_episodes:
            elapsed = time.time() - t0
            print(
                '[%s] ep %d/%d n_int=%d n_agents=%d steps=%d arrived=%d crashed=%s n_updates=%d elapsed=%.0fs'
                % (
                    label,
                    ep + 1,
                    n_episodes,
                    n_int,
                    n_agents,
                    row['steps'],
                    row['arrived'],
                    row['crashed'],
                    row['n_updates'],
                    elapsed,
                ),
                flush=True,
            )
        if (ep + 1) % TRAIN_CHECKPOINT_EVERY == 0:
            ckpt = dest / ('master_ep%04d.pth' % (ep + 1))
            master.save(str(ckpt))
            _save_master(master, dest, sidecar_src)
            _save_rng(rng, dest, ep + 1)
            print('[%s] checkpoint %s' % (label, ckpt.name), flush=True)

    for cell in cache.values():
        cell.close()

    agent.save(str(dest / 'agent.pth'))
    _save_master(master, dest, sidecar_src)
    summary = {
        'stage': label,
        'episodes': n_episodes,
        'seed': seed,
        'lr': lr,
        'master_init': str(master_path),
        'agent_init': str(agent_path),
        'elapsed_sec': time.time() - t0,
        'resumed_from': start_ep,
        'mean_arrival': None,
        'crash_rate': None,
    }
    rows = [json.loads(line) for line in history_path.read_text(encoding='utf-8').splitlines() if line.strip()] if history_path.is_file() else []
    if rows:
        summary['mean_arrival'] = float(np.mean([100.0 * r['arrived'] / max(1, r['n_agents']) for r in rows]))
        summary['crash_rate'] = 100.0 * float(np.mean([1.0 if r['crashed'] else 0.0 for r in rows]))
        summary['max_n_int'] = int(max(r['n_int'] for r in rows))
        summary['max_n_agents'] = int(max(r['n_agents'] for r in rows))
    _write_summary(dest, summary)
    print('[%s] done mean_arrival=%.1f crash_rate=%.1f' % (label, summary['mean_arrival'] or 0.0, summary['crash_rate'] or 0.0), flush=True)
    return summary


def train(*, smoke: bool = False, out_dir: Path | None = None, episodes: int | None = None) -> dict:
    dest = Path(out_dir) if out_dir else (repo_root() / 'outputs' / 'train')
    n_episodes = 2 if smoke else int(episodes or TRAIN_EPISODES)
    return train_loop(
        n_episodes=n_episodes,
        seed=TRAIN_SEED,
        lr=TRAIN_MASTER_LR,
        sample_fn=sample_stage2,
        agent_path=agent_checkpoint(),
        master_path=master_checkpoint(),
        dest=dest,
        smoke=smoke,
        label='stage2',
    )


def train_pipeline(*, smoke: bool = False, out_dir: Path | None = None) -> dict:
    dest = Path(out_dir) if out_dir else (repo_root() / 'outputs' / 'train')
    stage1_dir = dest / 'stage1'
    stage2_dir = dest / 'stage2'
    n1 = 2 if smoke else TRAIN_STAGE1_EPISODES
    n2 = 2 if smoke else TRAIN_EPISODES
    if (not smoke) and _stage_complete(stage1_dir, n1):
        s1 = _load_summary(stage1_dir)
        print('[pipeline] skip stage1, using %s' % (stage1_dir / 'master.pth'), flush=True)
    else:
        s1 = train_loop(
            n_episodes=n1,
            seed=TRAIN_STAGE1_SEED,
            lr=TRAIN_STAGE1_LR,
            sample_fn=sample_stage1,
            agent_path=agent_checkpoint(),
            master_path=master_checkpoint(),
            dest=stage1_dir,
            smoke=smoke,
            label='stage1',
        )
    if not smoke:
        archive_overscale_stage(stage2_dir)
    if (not smoke) and _stage_complete(stage2_dir, n2):
        s2 = _load_summary(stage2_dir)
        print('[pipeline] skip stage2, using %s' % (stage2_dir / 'master.pth'), flush=True)
    else:
        s2 = train_loop(
            n_episodes=n2,
            seed=TRAIN_SEED,
            lr=TRAIN_MASTER_LR,
            sample_fn=sample_stage2,
            agent_path=agent_checkpoint(),
            master_path=stage1_dir / 'master.pth',
            dest=stage2_dir,
            smoke=smoke,
            label='stage2',
        )
    sidecar = stage2_dir / 'master_custom_params.pt'
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'master.pth').write_bytes((stage2_dir / 'master.pth').read_bytes())
    agent_src = stage2_dir / 'agent.pth'
    if not agent_src.is_file():
        agent_src = Path(agent_checkpoint())
    (dest / 'agent.pth').write_bytes(agent_src.read_bytes())
    if sidecar.is_file():
        (dest / 'master_custom_params.pt').write_bytes(sidecar.read_bytes())
    summary = {'stage1': s1, 'stage2': s2, 'master': str(dest / 'master.pth'), 'agent': str(dest / 'agent.pth')}
    _write_summary(dest, summary)
    return summary
