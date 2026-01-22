import os
import csv
import glob
import numpy as np
from tabulate import tabulate
import argparse

def collect_tnt_results(base_dir):
    """Docstring"""
    """Docstring"""
    tnt_scenes = ['Barn', 'Caterpillar', 'Courthouse', 'Ignatius', 'Meetingroom', 'Truck']
    
    all_results = {
        'precision': [],
        'recall': [],
        'fscore': []
    }
    
    scene_names = []
    precision_values = []
    recall_values = []
    fscore_values = []
    
    for scene in tnt_scenes:
        result_file = os.path.join(base_dir, scene, "test", "mesh", "evaluation", "result.csv")
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    reader = csv.DictReader(f)
                    row = next(reader)
                    
                    precision = float(row['precision'])
                    recall = float(row['recall'])
                    fscore = float(row['fscore'])
                    
                    scene_names.append(scene)
                    precision_values.append(precision)
                    recall_values.append(recall)
                    fscore_values.append(fscore)
                    
                    all_results['precision'].append(precision)
                    all_results['recall'].append(recall)
                    all_results['fscore'].append(fscore)
                    
            except Exception as e:
                print(f" {scene} : {str(e)}")
        else:
            print(f": {scene}  {result_file}")
    
    if len(all_results['precision']) > 0:
        avg_results = {
            'precision': np.mean(all_results['precision']),
            'recall': np.mean(all_results['recall']),
            'fscore': np.mean(all_results['fscore'])
        }
        
        scene_names.append("Average")
        precision_values.append(avg_results['precision'])
        recall_values.append(avg_results['recall'])
        fscore_values.append(avg_results['fscore'])
        
        table_data = [
            ["precision"] + [f"{val:.6f}" for val in precision_values],
            ["recall"] + [f"{val:.6f}" for val in recall_values],
            ["fscore"] + [f"{val:.6f}" for val in fscore_values]
        ]
        
        headers = ["Metric"] + scene_names
        
        print("\nTNT:")
        print(tabulate(table_data, headers=headers, tablefmt="plain"))
        
        output_file = os.path.join(base_dir, "tnt_results_table.txt")
        with open(output_file, 'w') as f:
            f.write("TNT\n")
            f.write("=" * 80 + "\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt="plain"))
            f.write("\n")
        
        print(f"\n: {output_file}")
        
        return avg_results
    else:
        print(": ")
        return None

def main():
    parser = argparse.ArgumentParser(description='
    parser.add_argument('--input_dir', type=str, required=True,
                      help='
    parser.add_argument('--output_file', type=str, default=None,
                      help='
    
    args = parser.parse_args()
    
    if args.output_file:
        avg_results = collect_tnt_results(args.input_dir)
        if avg_results:
            default_output = os.path.join(args.input_dir, "tnt_results_table.txt")
            if os.path.exists(default_output):
                os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
                os.rename(default_output, args.output_file)
                print(f": {args.output_file}")
    else:
        collect_tnt_results(args.input_dir)

if __name__ == "__main__":
    main()
