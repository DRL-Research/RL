import json
import glob
import os

expected_results = {
    "Single Intersection": {
        "IDM": {"train": 652, "eval": 34, "att": 13.2},
        "COMA": {"train": 1156, "eval": 66, "att": 9.1},
        "VDN": {"train": 792, "eval": 50, "att": 10.7},
        "IPPO": {"train": 693, "eval": 37, "att": 15.5},
        "VN-MA-DDPG": {"train": 655, "eval": 37, "att": 9.9},
        "Social-Attention": {"train": 304, "eval": 12, "att": 17.2},
        "MA-GA-DDPG": {"train": 373, "eval": 5, "att": 7.4},
        "MAPS (ours)": {"train": 79, "eval": 1, "att": 6.8},
    },
    "Double Intersection": {
        "IDM": {"train": 1169, "eval": 69, "att": 5.1},
        "COMA": {"train": 1437, "eval": 80, "att": 5.9},
        "VDN": {"train": 1377, "eval": 75, "att": 7.3},
        "IPPO": {"train": 1359, "eval": 73, "att": 6.6},
        "VN-MA-DDPG": {"train": 1337, "eval": 86, "att": 6.1},
        "Social-Attention": {"train": 1124, "eval": 60, "att": 5.4},
        "MA-GA-DDPG": {"train": 1376, "eval": 77, "att": 6.4},
        "MAPS (ours)": {"train": 1057, "eval": 48, "att": 4.8},
    },
    "Roundabout": {
        "IDM": {"train": 1317, "eval": 76, "att": 12.7},
        "COMA": {"train": 1106, "eval": 61, "att": 16.1},
        "VDN": {"train": 1048, "eval": 57, "att": 20.9},
        "IPPO": {"train": 1104, "eval": 61, "att": 16.1},
        "VN-MA-DDPG": {"train": 988, "eval": 55, "att": 16.6},
        "Social-Attention": {"train": 970, "eval": 52, "att": 11.0},
        "MA-GA-DDPG": {"train": 1068, "eval": 62, "att": 13.2},
        "MAPS (ours)": {"train": 783, "eval": 47, "att": 9.3},
    }
}

layouts = {
    "Single Intersection": "rerun_all_2700",
    "Double Intersection": "rerun_all_2700_double",
    "Roundabout": "rerun_all_2700_roundabout"
}

algo_mapping = {
    "IDM": "IDM",
    "COMA": "COMA",
    "VDN": "VDN",
    "IPPO": "IPPO",
    "VN-MA-DDPG": "VN-MA-DDPG",
    "Social-Attention": "Social-Attention",
    "MA-GA-DDPG": "MA-GA-DDPG",
    "MAPS (ours)": "MAPS"
}

base_dir = "/Users/gil/PycharmProjects/RL/experiments_results"

def main():
    all_matched = True
    
    for layout_name, layout_folder in layouts.items():
        print(f"\\n{'='*50}")
        print(f"Validating Layout: {layout_name}")
        print(f"{'='*50}")
        
        layout_path = os.path.join(base_dir, layout_folder, "histories")
        if not os.path.exists(layout_path):
            print(f"Directory not found: {layout_path}")
            all_matched = False
            continue
            
        for algo_name, file_prefix in algo_mapping.items():
            search_pattern = os.path.join(layout_path, f"{file_prefix}_*.json")
            files = glob.glob(search_pattern)
            
            if not files:
                print(f"[{algo_name}] No files found matching {search_pattern}")
                all_matched = False
                continue
                
            train_colls = []
            eval_colls = []
            eval_atts = []
            
            for f in files:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                    
                c_flags = data.get("collision_flags", [])
                e_lengths = data.get("episode_lengths", [])
                
                # Training episodes: First 1800
                train_c = sum(c_flags[:1800])
                train_colls.append(train_c)
                
                # Eval episodes: Last 100
                eval_c = sum(c_flags[-100:])
                eval_colls.append(eval_c)
                
                # ATT is the average episode length during eval
                eval_l = e_lengths[-100:]
                if eval_l:
                    eval_atts.append(sum(eval_l) / len(eval_l))
                    
            avg_train_coll = sum(train_colls) / len(files)
            avg_eval_coll = sum(eval_colls) / len(files)
            avg_att = sum(eval_atts) / len(files) if eval_atts else 0
            
            expected = expected_results[layout_name][algo_name]
            
            # Format nicely
            calc_train = round(avg_train_coll)
            calc_eval = round(avg_eval_coll)
            calc_att = round(avg_att, 1)
            
            match_train = calc_train == expected['train']
            match_eval = calc_eval == expected['eval']
            match_att = calc_att == expected['att']
            
            matches = match_train and match_eval and match_att
            if not matches:
                all_matched = False
                
            status = "✅ PASS" if matches else "❌ FAIL"
            
            print(f"[{algo_name}] {status}")
            if not match_train:
                print(f"    Train Coll: calculated={calc_train}, expected={expected['train']}")
            if not match_eval:
                print(f"    Eval Coll : calculated={calc_eval}, expected={expected['eval']}")
            if not match_att:
                print(f"    ATT       : calculated={calc_att}, expected={expected['att']}")

    print(f"\\n{'='*50}")
    if all_matched:
        print("🎉 All results match the table successfully!")
    else:
        print("⚠️ Some results did not match the table.")

if __name__ == '__main__':
    main()
