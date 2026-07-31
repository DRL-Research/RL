import json
import glob
import os
import math

layouts = [
    ("Single Intersection", "rerun_all_2700"),
    ("Double Intersection", "rerun_all_2700_double"),
    ("Roundabout", "rerun_all_2700_roundabout")
]

algorithms = [
    ("IDM", "IDM"),
    ("COMA", "COMA"),
    ("VDN", "VDN"),
    ("IPPO", "IPPO"),
    ("VN-MA-DDPG", "VN-MA-DDPG"),
    ("Social-Attention", "Social-Attention"),
    ("MA-GA-DDPG", "MA-GA-DDPG")
]
maps_algo = ("MAPS (ours)", "MAPS")

base_dir = "/Users/gil/PycharmProjects/RL/experiments_results"

def fisher_exact_2sided(a, b, c, d):
    n = a + b + c + d
    def log_hypergeom(a1, b1, c1, d1):
        return (math.lgamma(a1+b1+1) + math.lgamma(c1+d1+1) + math.lgamma(a1+c1+1) + math.lgamma(b1+d1+1)
                - math.lgamma(a1+1) - math.lgamma(b1+1) - math.lgamma(c1+1) - math.lgamma(d1+1) - math.lgamma(n+1))
        
    p_obs = log_hypergeom(a, b, c, d)
    
    row1 = a + b
    col1 = a + c
    
    p_value = 0.0
    for i in range(max(0, row1 + col1 - n), min(row1, col1) + 1):
        a_i = i
        b_i = row1 - i
        c_i = col1 - i
        d_i = n - row1 - col1 + i
        
        p_i = log_hypergeom(a_i, b_i, c_i, d_i)
        # Add epsilon to handle float inaccuracies
        if p_i <= p_obs + 1e-9:
            p_value += math.exp(p_i)
            
    # Calculate odds ratio
    try:
        odds_ratio = (a * d) / (b * c)
    except ZeroDivisionError:
        odds_ratio = float('inf')
        
    return odds_ratio, min(1.0, p_value)

def get_pooled_eval_collisions(file_prefix, layout_folder):
    layout_path = os.path.join(base_dir, layout_folder, "histories")
    search_pattern = os.path.join(layout_path, f"{file_prefix}_*.json")
    files = glob.glob(search_pattern)
    
    if not files:
        return None, 0
        
    total_eval_collisions = 0
    total_eval_episodes = 0
    
    for f in files:
        with open(f, 'r') as fp:
            data = json.load(fp)
        c_flags = data.get("collision_flags", [])
        
        eval_c = c_flags[-100:]
        total_eval_collisions += sum(eval_c)
        total_eval_episodes += len(eval_c)
        
    return total_eval_collisions, total_eval_episodes

def main():
    print("Statistical Significance Tests (2-sided Fisher's Exact Test) on Evaluation Collisions\\n")
    
    for layout_name, layout_folder in layouts:
        print(f"=== {layout_name} ===")
        
        # Get MAPS data
        maps_coll, maps_episodes = get_pooled_eval_collisions(maps_algo[1], layout_folder)
        if maps_coll is None:
            print("MAPS data not found.\\n")
            continue
            
        # Find strongest baseline (lowest eval collisions)
        best_baseline_name = None
        best_baseline_coll = float('inf')
        best_baseline_episodes = 0
        
        for algo_name, file_prefix in algorithms:
            coll, ep = get_pooled_eval_collisions(file_prefix, layout_folder)
            if coll is not None and coll < best_baseline_coll:
                best_baseline_coll = coll
                best_baseline_episodes = ep
                best_baseline_name = algo_name
                
        if best_baseline_name is None:
            print("No baselines found.\\n")
            continue
            
        maps_success = maps_episodes - maps_coll
        baseline_success = best_baseline_episodes - best_baseline_coll
        
        odds_ratio, p_value = fisher_exact_2sided(maps_coll, maps_success, best_baseline_coll, baseline_success)
        
        print(f"MAPS (ours): {maps_coll} collisions out of {maps_episodes} eval episodes")
        print(f"Strongest Baseline ({best_baseline_name}): {best_baseline_coll} collisions out of {best_baseline_episodes} eval episodes")
        
        print(f"Contingency Table:")
        print(f"                 Collisions | No Collisions")
        print(f"  MAPS           {maps_coll:<10} | {maps_success}")
        print(f"  {best_baseline_name:<14} {best_baseline_coll:<10} | {baseline_success}")
        
        print(f"\\nFisher's Exact Test Results:")
        print(f"Odds Ratio: {odds_ratio:.4f}")
        print(f"P-value:    {p_value:.4e}")
        
        if p_value < 0.05:
            print("Conclusion: The difference IS statistically significant (p < 0.05).")
        else:
            print("Conclusion: The difference is NOT statistically significant (p >= 0.05).")
        print("\\n")

if __name__ == '__main__':
    main()
