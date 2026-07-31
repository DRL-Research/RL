import json
import glob
import os
import math

layouts = [
    ("Single Intersection", "rerun_all_2700"),
    ("Double Intersection", "rerun_all_2700_double"),
    ("Roundabout", "rerun_all_2700_roundabout")
]

base_dir = "/Users/gil/PycharmProjects/RL/experiments_results"
seeds = ["S42", "S100", "S2026"]

def t_cdf(t, df):
    # Approximation of t CDF for fractional df using math
    # We will just print the t and df, and we can compute exact p-values later if needed.
    pass

def independent_t_test(x, y):
    n1 = len(x)
    n2 = len(y)
    if n1 < 2 or n2 < 2: return 1.0, 1.0, 1.0
    
    mean1 = sum(x) / n1
    mean2 = sum(y) / n2
    
    var1 = sum((xi - mean1)**2 for xi in x) / (n1 - 1)
    var2 = sum((yi - mean2)**2 for yi in y) / (n2 - 1)
    
    se = math.sqrt(var1/n1 + var2/n2)
    if se == 0: return 0.0, float('inf'), 0.0
    
    t = abs(mean1 - mean2) / se
    df = (var1/n1 + var2/n2)**2 / ( (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1) )
    
    # We'll just return t and df and we can calculate p outside using scipy if installed, or an approximation.
    return t, df

for layout_name, layout_folder in layouts:
    maps_train = []
    base_train = []
    maps_eval = []
    base_eval = []
    layout_path = os.path.join(base_dir, layout_folder, "histories")
    
    # For eval baseline, the strongest baseline differs!
    # Single eval: MA-GA-DDPG
    # Double eval: Social-Attention
    # Roundabout eval: Social-Attention
    if layout_name == "Single Intersection":
        eval_base_name = "MA-GA-DDPG"
        train_base_name = "Social-Attention"
    else:
        eval_base_name = "Social-Attention"
        train_base_name = "Social-Attention"
        
    for s in seeds:
        maps_f = os.path.join(layout_path, f"MAPS_{s}.json")
        tbase_f = os.path.join(layout_path, f"{train_base_name}_{s}.json")
        ebase_f = os.path.join(layout_path, f"{eval_base_name}_{s}.json")
        
        if os.path.exists(maps_f) and os.path.exists(tbase_f):
            with open(maps_f, 'r') as fp:
                data = json.load(fp).get("collision_flags", [])
                maps_t = sum(data[:1800])
                maps_e = sum(data[-100:])
            with open(tbase_f, 'r') as fp:
                tbase_t = sum(json.load(fp).get("collision_flags", [])[:1800])
                
            maps_train.append(maps_t)
            base_train.append(tbase_t)
            
        if os.path.exists(maps_f) and os.path.exists(ebase_f):
            with open(ebase_f, 'r') as fp:
                ebase_e = sum(json.load(fp).get("collision_flags", [])[-100:])
            with open(maps_f, 'r') as fp:
                maps_e = sum(json.load(fp).get("collision_flags", [])[-100:])
            maps_eval.append(maps_e)
            base_eval.append(ebase_e)
            
    print(f"=== {layout_name} ===")
    p_train = paired_t_test(maps_train, base_train)
    p_eval = paired_t_test(maps_eval, base_eval)
    
    print(f"TRAIN: MAPS vs {train_base_name}")
    print(f"  MAPS : {maps_train}")
    print(f"  Base : {base_train}")
    print(f"  p = {p_train:.4f} ({'Significant' if p_train < 0.05 else 'Not Significant'})")
    
    print(f"EVAL : MAPS vs {eval_base_name}")
    print(f"  MAPS : {maps_eval}")
    print(f"  Base : {base_eval}")
    print(f"  p = {p_eval:.4f} ({'Significant' if p_eval < 0.05 else 'Not Significant'})\\n")

