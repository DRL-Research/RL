"""Analyze: scenario difficulty & route lengths with FIXED generator (multi_hop=1.0, max_hop_dist=1)."""
import sys, os
sys.path.insert(0, '.')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from run_chain_scalability import generate_chain_scenario, _approach_lanes, _outer_exits

rng = np.random.default_rng(42)

print("NEW SETTINGS: multi_hop_ratio=1.0, max_hop_dist=1")
print("(Every agent crosses exactly 1 intersection boundary)")

for n_int in [2, 3, 4, 5, 8]:
    n_agents = n_int * 3
    print(f"\n{'='*60}")
    print(f"Chain: {n_int} intersections, {n_agents} agents")
    print(f"{'='*60}")
    
    multi_hop_count = 0
    local_count = 0
    max_hop_actual = 0
    
    for ep in range(15):
        scenario = generate_chain_scenario(n_int, n_agents, rng, multi_hop_ratio=1.0, max_hop_dist=1)
        for lane_key, dest, offset in scenario["agents"]:
            src_int = int(lane_key[0].split("_")[0][1:])
            dst_int = int(dest.split("_")[0][1:])
            hop_dist = abs(dst_int - src_int)
            if hop_dist > 0:
                multi_hop_count += 1
                max_hop_actual = max(max_hop_actual, hop_dist)
            else:
                local_count += 1
    
    total = multi_hop_count + local_count
    print(f"  Multi-hop: {multi_hop_count}/{total} ({100*multi_hop_count/total:.0f}%)")
    print(f"  Local:     {local_count}/{total} ({100*local_count/total:.0f}%)")
    print(f"  Max hop distance: {max_hop_actual} intersections")
    
    spacing = 2 * 11 + 80
    approach_len = 100
    longest_route_m = approach_len + max_hop_actual * spacing + approach_len
    time_at_5ms = longest_route_m / 5.0
    print(f"  Longest route: {longest_route_m:.0f}m -> {time_at_5ms:.0f}s at 5 m/s")
    print(f"  Available: 200 steps -> {'OK' if time_at_5ms < 180 else 'TOO SHORT!'}")

print(f"\n{'='*60}")
print("CONFLICT ANALYSIS: max agents passing through any single intersection")
print(f"{'='*60}")
for n_int in [2, 3, 4, 5, 8]:
    n_agents = n_int * 3
    conflicts_per_scenario = []
    for ep in range(15):
        scenario = generate_chain_scenario(n_int, n_agents, rng, multi_hop_ratio=1.0, max_hop_dist=1)
        int_traffic = {i: 0 for i in range(n_int)}
        for lane_key, dest, offset in scenario["agents"]:
            src_int = int(lane_key[0].split("_")[0][1:])
            dst_int = int(dest.split("_")[0][1:])
            lo, hi = min(src_int, dst_int), max(src_int, dst_int)
            for i in range(lo, hi + 1):
                int_traffic[i] += 1
        max_traffic = max(int_traffic.values())
        conflicts_per_scenario.append(max_traffic)
    
    print(f"  {n_int} int ({n_agents} agents): max traffic/intersection = {np.mean(conflicts_per_scenario):.1f} avg, {max(conflicts_per_scenario)} max")
