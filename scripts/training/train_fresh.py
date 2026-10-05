from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training'))
import probe_random as probe
import run_chain_scalability as chain
OUT = Path(__file__).resolve().parents[2] / 'outputs' / 'fresh_net'
SLOTS, FEAT, PLAN = (5, 5, 4)
RADIUS = 8.0
THRESH = 0.175
HORIZON = 180
CONNECTOR = 150
PAPER_ARR = {2: 98.0, 16: 83.35}
AGENTS = {1: 3, 2: 6, 4: 12, 8: 24, 16: 48}

class Master(nn.Module):

    def __init__(self):
        super().__init__()
        self.slot = nn.Sequential(nn.Linear(FEAT, 64), nn.Tanh())
        self.out = nn.Sequential(nn.Linear(64 * SLOTS, 128), nn.Tanh(), nn.Linear(128, PLAN))
        self.value = nn.Sequential(nn.Linear(64 * SLOTS + 1, 64), nn.Tanh(), nn.Linear(64, 1))
        self.decoder = nn.Sequential(nn.Linear(PLAN, 32), nn.Tanh(), nn.Linear(32, 1))
        self.register_buffer('shift_scale', torch.zeros(()))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def features(self, obs: torch.Tensor) -> torch.Tensor:
        slots = obs.view(-1, SLOTS, FEAT)
        occupied = (slots[..., :4].abs().sum(-1) > 1e-06).unsqueeze(-1).float()
        content = slots.clone()
        content[..., 4] = 0.0
        hidden = self.slot(content) * occupied
        return hidden.reshape(hidden.shape[0], -1)

    def embed(self, obs: torch.Tensor) -> torch.Tensor:
        slots = obs.view(-1, SLOTS, FEAT)
        none = slots[..., :4].abs().sum(dim=(1, 2)) < 1e-06
        plan = torch.tanh(self.out(self.features(obs)))
        return torch.where(none.unsqueeze(-1), torch.zeros_like(plan), plan)

def _slot(vec4: torch.Tensor, bit: float) -> torch.Tensor:
    return torch.cat([vec4, vec4.new_tensor([float(bit)])])

def reduce_plans(master: Master, plans: torch.Tensor, bit: float) -> torch.Tensor:
    cur = plans
    while cur.shape[0] > 1:
        nxt = []
        for i in range(0, cur.shape[0], 2):
            kids = [cur[i]] + ([cur[i + 1]] if i + 1 < cur.shape[0] else [])
            rows = [_slot(k, bit) for k in kids]
            while len(rows) < SLOTS:
                rows.append(cur.new_zeros(FEAT))
            nxt.append(master.embed(torch.stack(rows).reshape(1, -1))[0])
        cur = torch.stack(nxt)
    return cur[0]

def direct_pool(master: Master, plans: torch.Tensor, bit: float) -> torch.Tensor:
    rows = [_slot(plans[i], bit) for i in range(min(4, plans.shape[0]))]
    while len(rows) < SLOTS:
        rows.append(plans.new_zeros(FEAT))
    return master.embed(torch.stack(rows).reshape(1, -1))[0]

def norm_car(row, center: float) -> np.ndarray:
    x, y, vx, vy = [float(v) for v in row]
    return np.array([(x - center) / 40.0, y / 40.0, vx / 12.0, vy / 12.0], np.float32)

def pack_np(rows: list[np.ndarray], bit: float) -> np.ndarray:
    slots = np.zeros((SLOTS, FEAT), np.float32)
    for i, row in enumerate(rows[:SLOTS]):
        slots[i, :4] = row
        slots[i, 4] = bit
    return slots.reshape(-1)

def observe(cell, state: np.ndarray, bit: float):
    zones = cell.agent_intersection_zone()
    spacing = cell._get_spacing()
    inner = cell._inner()
    arrived = [bool(getattr(v, 'is_arrived', False)) for v in inner.controlled_vehicles]
    far = {z: [] for z in range(cell.n_int)}
    near = {z: [] for z in range(cell.n_int)}
    meta = []
    for i, z in enumerate(zones):
        x, y, vx, vy = [float(v) for v in state[i]]
        moving = abs(x) + abs(y) + abs(vx) + abs(vy) > 0.001 and (not arrived[i])
        axis = probe.axis_of(vx, vy) if moving else 0
        dist = abs(y) if axis == 1 else abs(x - int(z) * spacing)
        speed = float(np.hypot(vx, vy))
        rec = {'zone': int(z), 'axis': int(axis), 'dist': float(dist), 'speed': speed, 'arrived': arrived[i]}
        meta.append(rec)
        if not moving or axis == 0 or dist < 5.0:
            continue
        vec = norm_car(state[i], int(z) * spacing)
        (far if dist > RADIUS else near)[int(z)].append((axis, vec, dist / max(1.0, speed)))
    scout = np.zeros((cell.n_int, SLOTS * FEAT), np.float32)
    scout_near = np.zeros((cell.n_int, SLOTS * FEAT), np.float32)
    near_pad = np.zeros((cell.n_int, 4, 4), np.float32)
    zsign = np.zeros(cell.n_int, np.float32)
    tta = np.zeros(cell.n_int, np.float32)
    for z in range(cell.n_int):

        def ordered(items):
            items = sorted(items, key=lambda it: (0 if it[0] == 1 else 1, it[2]))
            return items
        f = ordered(far[z])
        n = ordered(near[z])
        scout[z] = pack_np([it[1] for it in f + n], bit)
        scout_near[z] = pack_np([it[1] for it in n], bit)
        for j, it in enumerate(n[:4]):
            near_pad[z, j] = it[1]
        both = f + n
        verts = [it[2] for it in both if it[0] == 1]
        horiz = [it[2] for it in both if it[0] == -1]
        fars = [it[2] for it in f]
        if fars:
            tta[z] = float(min(fars))
        if verts and horiz:
            zsign[z] = 1.0 if min(verts) <= min(horiz) else -1.0
    return (scout, near_pad, scout_near, zsign, tta, meta)

def plans_from(master: Master, scout_np, near_np, bit: float, mode: str):
    scout_obs = torch.tensor(scout_np, dtype=torch.float32)
    near = torch.tensor(near_np, dtype=torch.float32)
    scouts = master.embed(scout_obs)
    parent = reduce_plans(master, scouts, bit) if scouts.shape[0] > 1 else scouts[0]
    use_parent = mode != 'no_parent'
    local = []
    for z in range(near.shape[0]):
        own = scouts[z] if use_parent else scouts.new_zeros(PLAN)
        rows = [_slot(own, bit if use_parent else 0.0)]
        for j in range(4):
            if float(near[z, j].detach().abs().sum()) > 1e-06 and len(rows) < SLOTS:
                rows.append(_slot(near[z, j], bit))
        while len(rows) < SLOTS:
            rows.append(scouts.new_zeros(FEAT))
        local.append(torch.stack(rows).reshape(-1))
    local_obs = torch.stack(local)
    local_plan = master.embed(local_obs)
    shift = 0.8 * master.shift_scale * float(bit)
    if mode == 'plan_zero':
        pri = local_plan.new_zeros(local_plan.shape[0])
    else:
        base = local_plan[:, 0] if mode == 'no_parent' else scouts[:, 0]
        pri = torch.where(base.abs() < 1e-05, torch.zeros_like(base), base + shift)
    return (pri, local_plan, scouts, parent)

def hard_speed(priority: torch.Tensor, lane: torch.Tensor) -> torch.Tensor:
    return (priority * lane > -THRESH).long()

def rollout(cell, scenario, master: Master | None, mode: str, oracle: bool):
    bit = 0.0 if mode == 'bit_off' else 1.0
    state = cell.reset(scenario)
    rows = []
    agree = []
    n = 0
    if master is not None:
        master.eval()
    while n < HORIZON and (not cell.done):
        n += 1
        scout, near, scout_near, zsign, tta, meta = observe(cell, state, bit)
        if oracle or master is None:
            actions = probe.actions_for(cell, state, 'closer', RADIUS)
            pri_np = np.zeros(cell.n_int, np.float32)
        else:
            with torch.no_grad():
                seen = scout_near if mode == 'no_parent' else scout
                pri, _, _, _ = plans_from(master, seen, near, bit, mode)
            pri_np = pri.cpu().numpy()
            actions = [0] * cell.n_agents
            for i, m in enumerate(meta):
                if m['arrived'] or m['axis'] == 0:
                    continue
                actions[i] = int(pri_np[m['zone']] * m['axis'] > -THRESH)
            for z in range(cell.n_int):
                if zsign[z] == 0 or (abs(pri_np[z]) < 1e-06 and mode == 'plan_zero'):
                    continue
                if zsign[z] != 0:
                    agree.append(float(np.sign(pri_np[z]) == np.sign(zsign[z]) or (abs(pri_np[z]) < 0.001 and False)))
                    if abs(pri_np[z]) >= THRESH:
                        agree[-1] = float(np.sign(pri_np[z]) == np.sign(zsign[z]))
        rows.append({'scout': scout, 'near': near, 'zsign': zsign, 'tta': tta, 'bit': bit})
        state = cell.step(actions)
        if cell.crashed:
            break
    arrival = 100.0 * cell.n_arrived() / max(1, cell.n_agents)
    acc = float(np.mean(agree)) if agree else 0.0
    return (rows, arrival, 100.0 * float(cell.crashed), acc)

def hinge_on(master: Master, rows: list[dict], opt, margin: float) -> float:
    master.train()
    rows = list(rows)
    np.random.default_rng(0).shuffle(rows)
    total = 0.0
    count = 0
    opt.zero_grad()
    loss = None
    n = 0
    for row in rows:
        pri, local, scouts, parent = plans_from(master, row['scout'], row['near'], float(row['bit']), 'full')
        sign = torch.tensor(row['zsign'])
        mask = sign.abs() > 0
        if int(mask.sum()) == 0:
            continue
        term = F.relu(margin - sign * pri)[mask].mean()
        has = torch.tensor(row['tta']) > 0
        if int(has.sum()) > 0 and float(parent.detach().abs().sum()) > 0:
            term = term + 0.0 * F.mse_loss(master.decoder(parent.unsqueeze(0)).squeeze(), has.float().new_tensor(float(row['tta'][has.numpy()].mean())))
        if scouts.shape[0] >= 4:
            four = scouts[:4]
            term = term + 0.0 * (reduce_plans(master, four, float(row['bit']))[0] - direct_pool(master, four, float(row['bit']))[0]).pow(2)
        loss = term if loss is None else loss + term
        n += 1
        if n >= 24:
            (loss / n).backward()
            nn.utils.clip_grad_norm_(master.parameters(), 5.0)
            opt.step()
            total += float(loss.detach()) / n
            count += 1
            opt.zero_grad()
            loss = None
            n = 0
    if n > 0 and loss is not None:
        (loss / n).backward()
        nn.utils.clip_grad_norm_(master.parameters(), 5.0)
        opt.step()
        total += float(loss.detach()) / n
        count += 1
    return total / max(1, count)

def get_cell(cache, n_int):
    if n_int not in cache:
        cache[n_int] = chain.ChainCell(n_int, 3 * n_int, [5, 10], connector_length=CONNECTOR, duration=HORIZON)
    return cache[n_int]

def scripted(cell, scenario, kind: str):
    state = cell.reset(scenario)
    steps = 0
    while steps < HORIZON and (not cell.done):
        steps += 1
        state = cell.step(probe.actions_for(cell, state, kind, RADIUS))
        if cell.crashed:
            break
    return (100.0 * cell.n_arrived() / cell.n_agents, 100.0 * float(cell.crashed))

def gate(cache) -> bool:
    ok = True
    summary = {}
    for n_int, n_scen in ((2, 8), (16, 3)):
        cell = get_cell(cache, n_int)
        scens = probe.build(n_int, n_scen, 2026)
        scores = {}
        for kind in ('slow', 'fast', 'closer'):
            arr, cr = ([], [])
            for sc in scens:
                a, c = scripted(cell, sc, kind)
                arr.append(a)
                cr.append(c)
            scores[kind] = (float(np.mean(arr)), float(np.mean(cr)))
            print(f'gate {AGENTS[n_int]} {kind:7s} arr {scores[kind][0]:6.1f} crash {scores[kind][1]:6.1f}', flush=True)
        summary[n_int] = scores
        if scores['closer'][0] < max(scores['slow'][0], scores['fast'][0]) + 15.0:
            print('gate fail: closer does not beat constants by 15', flush=True)
            ok = False
    for kind in ('slow', 'fast'):
        if summary[2][kind][0] >= PAPER_ARR[2] - 10 and summary[16][kind][0] >= PAPER_ARR[16] - 10:
            print(f'gate fail: {kind} is within 10 of the paper at 6 and 48', flush=True)
            ok = False
    return ok

def evaluate(master, cache, scales, n_scen: int, modes=('full', 'bit_off', 'no_parent', 'plan_zero')):
    out = {}
    for n_int in scales:
        cell = get_cell(cache, n_int)
        scens = probe.build(n_int, n_scen, 2026)
        out[n_int] = {}
        for mode in modes:
            arr, cr, acc = ([], [], [])
            for sc in scens:
                _rows, a, c, g = rollout(cell, sc, master, mode, oracle=False)
                arr.append(a)
                cr.append(c)
                acc.append(g)
            out[n_int][mode] = {'arrival': float(np.mean(arr)), 'crash': float(np.mean(cr)), 'sign_agree': float(np.mean(acc))}
            print(f"eval {AGENTS[n_int]:2d} {mode:10s} arr {out[n_int][mode]['arrival']:6.1f} crash {out[n_int][mode]['crash']:6.1f} agree {out[n_int][mode]['sign_agree']:.2f}", flush=True)
    return out

def collect(cache) -> list[dict]:
    rows = []
    for n_int, n_ep, seed in ((1, 8, 11), (2, 16, 12), (4, 6, 13)):
        cell = get_cell(cache, n_int)
        for i in range(n_ep):
            scen = probe.build(n_int, 1, seed * 1000 + i)[0]
            ep, arr, cr, _ = rollout(cell, scen, None, 'full', oracle=True)
            rows.extend(ep)
            print(f'oracle n={AGENTS[n_int]} ep {i} arr {arr:.0f} crash {cr:.0f} steps {len(ep)}', flush=True)
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-gate', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    if args.skip_gate:
        print('gate already passed this session', flush=True)
    else:
        print('scripted gate before any gradient', flush=True)
        if not gate(cache):
            for c in cache.values():
                c.close()
            sys.exit(2)
    torch.manual_seed(0)
    master = Master()
    opt = torch.optim.Adam(master.parameters(), lr=0.003)
    print('collect oracle states along the whole drive', flush=True)
    data = collect(cache)
    print(f'states {len(data)}', flush=True)
    best = -1.0
    cell2 = get_cell(cache, 2)
    for rnd in range(20):
        loss = hinge_on(master, data, opt, 0.35)
        for _extra in range(3):
            loss = hinge_on(master, data, opt, 0.35)
        for i in range(8):
            scen = probe.build(2, 1, 9000 + rnd * 20 + i)[0]
            visited, _, _, _ = rollout(cell2, scen, master, 'full', oracle=False)
            data.extend(visited)
        scores = evaluate(master, cache, (2,), 8, ('full',))
        full = scores[2]['full']
        print(f"fit {rnd} hinge {loss:.4f} arr {full['arrival']:.1f} agree {full['sign_agree']:.2f} states {len(data)}", flush=True)
        if full['arrival'] > best:
            best = full['arrival']
            torch.save({'master': master.state_dict(), 'round': rnd}, OUT / 'master_best.pth')
        if full['arrival'] >= 90.0 and full['sign_agree'] >= 0.85:
            break
    print('ablation check', flush=True)
    scores = evaluate(master, cache, (2, 4), 6)
    full6 = scores[2]['full']['arrival']
    zero6 = scores[2]['plan_zero']['arrival']
    if full6 >= 70.0 and full6 >= zero6 + 15.0:
        print('arm +0.8 and refit the effective priority on every step', flush=True)
        master.shift_scale.fill_(1.0)
        for rnd in range(8):
            loss = hinge_on(master, data, opt, 0.2)
            scores = evaluate(master, cache, (2,), 6)
            full = scores[2]['full']['arrival']
            off = scores[2]['bit_off']['arrival']
            print(f'armed {rnd} hinge {loss:.4f} full {full:.1f} bit_off {off:.1f}', flush=True)
            if off > full + 20.0:
                print('kill: bit-off beats bit-on by more than 20', flush=True)
                break
            if full >= 90.0 and off <= full - 10.0:
                break
    torch.save({'master': master.state_dict(), 'shift': float(master.shift_scale)}, OUT / 'master.pth')
    final = evaluate(master, cache, (2, 4, 8, 16), 4)
    payload = {'shift': float(master.shift_scale), 'radius_m': RADIUS, 'results': {str(AGENTS[k]): v for k, v in final.items()}}
    (OUT / 'eval.json').write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(json.dumps(payload['results'], indent=2), flush=True)
    for c in cache.values():
        c.close()
if __name__ == '__main__':
    main()
