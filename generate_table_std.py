import json
import glob
import os
import statistics
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
    ("MA-GA-DDPG", "MA-GA-DDPG"),
    ("MAPS (ours)", "MAPS")
]

base_dir = "/Users/gil/PycharmProjects/RL/experiments_results"

def format_cell(mean, std, is_att, is_best, is_second_best):
    if not is_att:
        s = f"${math.ceil(mean)} \\pm {math.ceil(std)}$"
    else:
        s = f"${mean:.1f} \\pm {std:.1f}$"
        
    if is_best:
        return f"\\textbf{{{s}}}"
    elif is_second_best:
        return f"\\underline{{{s}}}"
    return s

def main():
    print("Generating Latex Table Data with Mean \\pm Std...")
    
    # 1. Collect all data
    results = {}
    for algo_name, file_prefix in algorithms:
        results[algo_name] = {}
        for layout_name, layout_folder in layouts:
            layout_path = os.path.join(base_dir, layout_folder, "histories")
            search_pattern = os.path.join(layout_path, f"{file_prefix}_*.json")
            files = glob.glob(search_pattern)
            
            if not files:
                results[algo_name][layout_name] = None
                continue
                
            train_colls = []
            eval_colls = []
            eval_atts = []
            
            for f in files:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    
                c_flags = data.get("collision_flags", [])
                e_lengths = data.get("episode_lengths", [])
                s_flags = data.get("success_flags", [])
                
                train_c = sum(c_flags[:1800])
                train_colls.append(train_c)
                
                eval_c = sum(c_flags[-100:])
                eval_colls.append(eval_c)
                
                eval_l = e_lengths[-100:]
                eval_s = s_flags[-100:] if s_flags else [1]*len(eval_l)
                successful_eval_l = [l for l, s in zip(eval_l, eval_s) if s]
                
                if successful_eval_l:
                    eval_atts.append(sum(successful_eval_l) / len(successful_eval_l))
            
            train_mean = statistics.mean(train_colls)
            train_std = statistics.pstdev(train_colls)
            
            eval_mean = statistics.mean(eval_colls)
            eval_std = statistics.pstdev(eval_colls)
            
            att_mean = statistics.mean(eval_atts) if eval_atts else 0
            att_std = statistics.pstdev(eval_atts) if eval_atts else 0
            
            results[algo_name][layout_name] = {
                'train_mean': train_mean, 'train_std': train_std,
                'eval_mean': eval_mean, 'eval_std': eval_std,
                'att_mean': att_mean, 'att_std': att_std
            }
            
    # 2. Find best and second best for each metric in each layout
    best_vals = {}
    for layout_name, _ in layouts:
        best_vals[layout_name] = {}
        for metric in ['train_mean', 'eval_mean', 'att_mean']:
            vals = []
            for algo_name, _ in algorithms:
                res = results[algo_name].get(layout_name)
                if res is not None:
                    vals.append(res[metric])
            
            # Using a small round to avoid float precision issues when finding unique values
            unique_vals = sorted(list(set(round(v, 5) for v in vals)))
            best = unique_vals[0] if len(unique_vals) > 0 else None
            second_best = unique_vals[1] if len(unique_vals) > 1 else None
            best_vals[layout_name][metric] = (best, second_best)

    # 3. Print the table
    print("\\begin{table*}[t]")
    print("\\centering")
    print("\\small")
    print("\\setlength{\\tabcolsep}{5pt}")
    print("\\begin{tabular}{l ccc ccc ccc}")
    print("\\toprule")
    print("& \\multicolumn{3}{c}{\\textbf{Single Intersection}} & \\multicolumn{3}{c}{\\textbf{Double Intersection}} & \\multicolumn{3}{c}{\\textbf{Roundabout}} \\\\")
    print("\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}")
    print("\\textbf{Approach} & Coll.\\ (train) & Coll.\\ (eval) & $ATT$ & Coll.\\ (train) & Coll.\\ (eval) & $ATT$ & Coll.\\ (train) & Coll.\\ (eval) & $ATT$ \\\\")
    print("\\midrule")

    for algo_name, _ in algorithms:
        if algo_name == "MAPS (ours)":
            print("\\midrule")
            
        row_str = f"{algo_name}"
        
        for layout_name, _ in layouts:
            res = results[algo_name].get(layout_name)
            if res is None:
                row_str += " & N/A & N/A & N/A"
                continue
            
            bests = best_vals[layout_name]
            
            t_m = res['train_mean']
            e_m = res['eval_mean']
            a_m = res['att_mean']
            
            t_m_rounded = round(t_m, 5)
            e_m_rounded = round(e_m, 5)
            a_m_rounded = round(a_m, 5)
            
            is_best_train = (t_m_rounded == bests['train_mean'][0])
            is_second_train = (t_m_rounded == bests['train_mean'][1])
            
            is_best_eval = (e_m_rounded == bests['eval_mean'][0])
            is_second_eval = (e_m_rounded == bests['eval_mean'][1])
            
            is_best_att = (a_m_rounded == bests['att_mean'][0])
            is_second_att = (a_m_rounded == bests['att_mean'][1])
            
            cell_train = format_cell(t_m, res['train_std'], is_att=False, is_best=is_best_train, is_second_best=is_second_train)
            cell_eval = format_cell(e_m, res['eval_std'], is_att=False, is_best=is_best_eval, is_second_best=is_second_eval)
            cell_att = format_cell(a_m, res['att_std'], is_att=True, is_best=is_best_att, is_second_best=is_second_att)
            
            row_str += f" & {cell_train} & {cell_eval} & {cell_att}"
            
        row_str += " \\\\"
        print(row_str)
        
    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\caption{Collision counts and traversal efficiency across three layouts. Collisions are reported over 1,800 training episodes and 100 deterministic evaluation episodes; $ATT$ is the average number of simulation steps to traverse. Lower is better throughout. \\textbf{Bold}: best; \\underline{underlined}: second-best. MAPS is best in all nine cells, while the runner-up changes across layouts. All metrics are reported as mean $\\pm$ std across 3 seeds.}")
    print("\\label{tab:main_results_std}")
    print("\\end{table*}")

if __name__ == '__main__':
    main()
