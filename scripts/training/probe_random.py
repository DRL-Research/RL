import sys
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / 'scripts' / 'evaluation'))
import highwayenv.CustomControlledVehicle as ccv
import run_chain_scalability as chain

def _act(self, action=None):
    if action == 'FASTER':
        self.target_speed = 10.0
        action = None
    elif action == 'SLOWER':
        self.target_speed = 5.0
        action = None
    self.follow_road()
    acc = self.speed_control(self.target_speed)
    steer = float(np.clip(self.steering_control(self.target_lane_index), -self.MAX_STEERING_ANGLE, self.MAX_STEERING_ANGLE))
    ccv.Vehicle.act(self, {'steering': steer, 'acceleration': acc})
ccv.CustomControlledVehicle.act = _act
OFF_CLOSE, OFF_FAR = (52.0, 46.0)

def build(n_int, n, seed):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        hop = int(rng.integers(2, 4))
        kw = {} if n_int <= 2 else dict(hop_window=hop, bidirectional=True, jitter=0.0)
        if n_int <= 2:
            kw = dict(jitter=0.0)
        scen = chain.generate_chain_scenario(n_int, 3 * n_int, rng, **kw)
        close_vert = {}
        for i in range(n_int):
            close_vert[i] = bool(rng.random() < 0.5)
        seen = {}
        agents = []
        for lane_key, dest, _off in scen['agents']:
            name = str(lane_key[0])
            inter = int(name.split('_')[0][1:])
            corner = int(name.split('_o')[-1])
            vert = corner in (0, 2)
            key = (inter, corner)
            extra = 0.0
            if key in seen:
                extra = -12.0
            seen[key] = True
            off = (OFF_CLOSE if vert == close_vert[inter] else OFF_FAR) + extra
            agents.append((lane_key, dest, off))
        out.append({'agents': agents, 'static': []})
    return out

def axis_of(vx, vy):
    if abs(vy) >= abs(vx) and abs(vy) > 0.2:
        return 1
    if abs(vx) > 0.2:
        return -1
    return 0

def actions_for(cell, state, kind, radius):
    n = cell.n_agents
    if kind == 'slow':
        return [0] * n
    if kind == 'fast':
        return [1] * n
    zones = cell.agent_intersection_zone()
    spacing = cell._get_spacing()
    actions = [1] * n
    by = {}
    for i, z in enumerate(zones):
        by.setdefault(int(z), []).append(i)
    for z, ids in by.items():
        cx = z * spacing
        groups = {1: [], -1: []}
        info = {}
        for i in ids:
            x, y, vx, vy = [float(v) for v in state[i]]
            if abs(x) + abs(y) + abs(vx) + abs(vy) < 0.001:
                continue
            axis = axis_of(vx, vy)
            if not axis:
                continue
            dist = abs(y) if axis == 1 else abs(x - cx)
            info[i] = axis
            if dist < 5:
                continue
            if kind == 'local' and dist > radius:
                continue
            groups[axis].append((dist / max(1.0, float(np.hypot(vx, vy))), i))
        if kind in ('vert', 'horiz'):
            for i, axis in info.items():
                actions[i] = (1 if axis == 1 else 0) if kind == 'vert' else 0 if axis == 1 else 1
            continue
        if not groups[1] or not groups[-1]:
            continue
        tv = min((t for t, _ in groups[1]))
        th = min((t for t, _ in groups[-1]))
        yield_axis = -1 if tv <= th else 1
        for axis, rows in groups.items():
            for _t, i in rows:
                actions[i] = 0 if axis == yield_axis else 1
    return actions

def run(cell, scen, kind, radius):
    state = cell.reset(scen)
    steps = 0
    while steps < 180 and (not cell.done):
        steps += 1
        state = cell.step(actions_for(cell, state, kind, radius))
        if cell.crashed:
            break
    return (100.0 * cell.n_arrived() / cell.n_agents, 100.0 * float(cell.crashed))

def report(n_int, n, radius):
    scens = build(n_int, n, 2026)
    cell = chain.ChainCell(n_int, 3 * n_int, [5, 10], connector_length=150, duration=180)
    print(f'\n=== agents {3 * n_int} n={n} radius={radius} ===', flush=True)
    for kind in ('slow', 'fast', 'closer', 'vert', 'horiz', 'local'):
        arr, cr = ([], [])
        for sc in scens:
            a, c = run(cell, sc, kind, radius)
            arr.append(a)
            cr.append(c)
        print(f'  {kind:8s} arr {np.mean(arr):6.1f} crash {np.mean(cr):6.1f}', flush=True)
    cell.close()
if __name__ == chr(95) * 2 + chr(110) + chr(97) + chr(109) + chr(101) + chr(95) * 2:
    report(2, 16, 12.0)
    report(2, 16, 8.0)
    report(4, 8, 12.0)
