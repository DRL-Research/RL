import json
import scipy.stats as stats
import os

directory = 'experiments_results/rerun_all_2700_roundabout/histories'
seeds = ['S42', 'S100', 'S2026']
models = ['MAPS', 'Social-Attention']
windows = [200, 300]

def count_collisions(model, seed, window):
    filename = os.path.join(directory, f"{model}_{seed}.json")
    with open(filename, 'r') as f:
        data = json.load(f)
        flags = data['collision_flags']
        flags_window = flags[-window:]
        return sum(flags_window)

for window in windows:
    maps_col_total = 0
    maps_nocol_total = 0
    sa_col_total = 0
    sa_nocol_total = 0
    
    for seed in seeds:
        maps_c = count_collisions('MAPS', seed, window)
        maps_col_total += maps_c
        maps_nocol_total += (window - maps_c)
        
        sa_c = count_collisions('Social-Attention', seed, window)
        sa_col_total += sa_c
        sa_nocol_total += (window - sa_c)
        
    table = [[maps_col_total, maps_nocol_total], [sa_col_total, sa_nocol_total]]
    res = stats.fisher_exact(table, alternative='two-sided')
    
    print(f"--- Window: Last {window} episodes per seed (Total {window*3} episodes) ---")
    print(f"MAPS Collisions: {maps_col_total}, No Collisions: {maps_nocol_total}")
    print(f"Social-Attention Collisions: {sa_col_total}, No Collisions: {sa_nocol_total}")
    print(f"Table: {table}")
    print(f"Odds ratio: {res.statistic:.4f}, p-value: {res.pvalue:.4e}")
    if res.pvalue < 0.05:
        print("Result: Statistically significant (p < 0.05)\n")
    else:
        print("Result: Not statistically significant (p >= 0.05)\n")

