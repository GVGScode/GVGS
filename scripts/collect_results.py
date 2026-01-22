import os
import json
import glob
import numpy as np
from tabulate import tabulate
import argparse

def collect_results(base_dir):
    """Collect all scan results and calculate average"""
    scan_dirs = glob.glob(os.path.join(base_dir, "scan*"))
    
    def extract_scan_number(scan_path):
        scan_name = os.path.basename(scan_path)
        if scan_name.startswith('scan'):
            try:
                return int(scan_name[4:])
            except ValueError:
                return 0
        return 0
    
    scan_dirs = sorted(scan_dirs, key=extract_scan_number)
    
    all_results = {
        'mean_d2s': [],
        'mean_s2d': [],
        'overall': []
    }
    
    scan_ids = []
    mean_d2s_values = []
    mean_s2d_values = []
    overall_values = []
    
    for scan_dir in scan_dirs:
        scan_id = os.path.basename(scan_dir)
        result_file = os.path.join(scan_dir, "test", "mesh", "results.json")
        
        if os.path.exists(result_file):
            try:
                with open(result_file, 'r') as f:
                    result = json.load(f)
                
                scan_ids.append(scan_id)
                mean_d2s_values.append(result['mean_d2s'])
                mean_s2d_values.append(result['mean_s2d'])
                overall_values.append(result['overall'])
                
                all_results['mean_d2s'].append(result['mean_d2s'])
                all_results['mean_s2d'].append(result['mean_s2d'])
                all_results['overall'].append(result['overall'])
                
            except Exception as e:
                print(f"Error reading results for {scan_id}: {str(e)}")
        else:
            print(f"Warning: {scan_id} result file not found")
    
    avg_results = {
        'mean_d2s': np.mean(all_results['mean_d2s']),
        'mean_s2d': np.mean(all_results['mean_s2d']),
        'overall': np.mean(all_results['overall'])
    }
    
    scan_ids.append("Average")
    mean_d2s_values.append(avg_results['mean_d2s'])
    mean_s2d_values.append(avg_results['mean_s2d'])
    overall_values.append(avg_results['overall'])
    
    table_data = [
        ["mean_d2s"] + [f"{val:.6f}" for val in mean_d2s_values],
        ["mean_s2d"] + [f"{val:.6f}" for val in mean_s2d_values],
        ["overall"] + [f"{val:.6f}" for val in overall_values]
    ]
    
    headers = ["Metric"] + scan_ids
    
    print("\nResults Statistics:")
    print(tabulate(table_data, headers=headers, tablefmt="plain"))
    
    output_file = os.path.join(base_dir, "results_table.txt")
    with open(output_file, 'w') as f:
        f.write(tabulate(table_data, headers=headers, tablefmt="plain"))
    
    print(f"\nResults saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Collect and summarize DTU evaluation results')
    parser.add_argument('--input_dir', type=str, required=True,
                      help='Input directory path containing scan* subdirectories')
    parser.add_argument('--output_file', type=str, default=None,
                      help='Output file path (optional, default saved to results_table.txt in input directory)')
    
    args = parser.parse_args()
    
    if args.output_file:
        base_dir = os.path.dirname(args.output_file)
        os.makedirs(base_dir, exist_ok=True)
        collect_results(args.input_dir)
        os.rename(
            os.path.join(args.input_dir, "results_table.txt"),
            args.output_file
        )
        print(f"Results saved to: {args.output_file}")
    else:
        collect_results(args.input_dir)

if __name__ == "__main__":
    main() 