import scipy.stats as stats

data = {
    "Single Intersection": {
        "MAPS": [3, 297],
        "MA-GA-DDPG": [18, 282]
    },
    "Double Intersection": {
        "MAPS": [147, 153],
        "Social-Attention": [180, 120]
    },
    "Roundabout": {
        "MAPS": [144, 156],
        "Social-Attention": [156, 144]
    }
}

for layout, counts in data.items():
    print(f"--- {layout} ---")
    maps_col, maps_nocol = counts["MAPS"]
    
    baseline_name = [k for k in counts.keys() if k != "MAPS"][0]
    base_col, base_nocol = counts[baseline_name]
    
    table = [[maps_col, maps_nocol], [base_col, base_nocol]]
    res = stats.fisher_exact(table, alternative='two-sided')
    print(f"MAPS vs {baseline_name}")
    print(f"Table: {table}")
    print(f"Odds ratio: {res.statistic:.4f}, p-value: {res.pvalue:.4e}")

    # Generate LaTeX table row
    latex_row = f"{layout} & MAPS vs {baseline_name} & {table[0][0]}:{table[0][1]} vs {table[1][0]}:{table[1][1]} & {res.statistic:.3f} & {res.pvalue:.4f} \\\\"
    print("LaTeX Table Row:")
    print(latex_row)
