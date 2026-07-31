import math

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
        if p_i <= p_obs + 1e-9:
            p_value += math.exp(p_i)
            
    return min(1.0, p_value)

# Data from the table
# Format: layout -> { 'train_n': 1800, 'eval_n': 100,
#                     'maps_train': X, 'maps_eval': Y,
#                     'baseline_name': Z, 'baseline_train': W, 'baseline_eval': V }

# Single Intersection: MAPS train 79, eval 1
# Strongest train baseline: Social-Attention (304)
# Strongest eval baseline: MA-GA-DDPG (6)
# Double Intersection: MAPS train 1057, eval 49
# Strongest train baseline: Social-Attention (1124)
# Strongest eval baseline: Social-Attention (60)
# Roundabout: MAPS train 783, eval 48
# Strongest train baseline: Social-Attention (970)
# Strongest eval baseline: Social-Attention (52)

scenarios = [
    {
        "layout": "Single Intersection",
        "train_n": 1800,
        "eval_n": 100,
        "maps_train": 79,
        "maps_eval": 1,
        "baseline_train_name": "Social-Attention",
        "baseline_train": 304,
        "baseline_eval_name": "MA-GA-DDPG",
        "baseline_eval": 6
    },
    {
        "layout": "Double Intersection",
        "train_n": 1800,
        "eval_n": 100,
        "maps_train": 1057,
        "maps_eval": 49,
        "baseline_train_name": "Social-Attention",
        "baseline_train": 1124,
        "baseline_eval_name": "Social-Attention",
        "baseline_eval": 60
    },
    {
        "layout": "Roundabout",
        "train_n": 1800,
        "eval_n": 100,
        "maps_train": 783,
        "maps_eval": 48,
        "baseline_train_name": "Social-Attention",
        "baseline_train": 970,
        "baseline_eval_name": "Social-Attention",
        "baseline_eval": 52
    }
]

for s in scenarios:
    print(f"=== {s['layout']} ===")
    
    # Train
    maps_t_succ = s['train_n'] - s['maps_train']
    base_t_succ = s['train_n'] - s['baseline_train']
    pval_train = fisher_exact_2sided(s['maps_train'], maps_t_succ, s['baseline_train'], base_t_succ)
    
    print(f"TRAIN (N=1800): MAPS ({s['maps_train']}) vs {s['baseline_train_name']} ({s['baseline_train']})")
    print(f"  P-value: {pval_train:.4e}")
    if pval_train < 0.05:
        print("  -> Statistically Significant")
    else:
        print("  -> NOT Significant")
        
    # Eval
    maps_e_succ = s['eval_n'] - s['maps_eval']
    base_e_succ = s['eval_n'] - s['baseline_eval']
    pval_eval = fisher_exact_2sided(s['maps_eval'], maps_e_succ, s['baseline_eval'], base_e_succ)
    
    print(f"EVAL  (N=100) : MAPS ({s['maps_eval']}) vs {s['baseline_eval_name']} ({s['baseline_eval']})")
    print(f"  P-value: {pval_eval:.4e}")
    if pval_eval < 0.05:
        print("  -> Statistically Significant")
    else:
        print("  -> NOT Significant")
    print("")

