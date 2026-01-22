#!/usr/bin/env python3
"""
"""

import os
import subprocess
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

scenes = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]

available_gpus = [7,6,5,4,3,2,1,0]
gpu_memory_threshold = 1024
max_concurrent = 4

gpu_status = {gpu_id: False for gpu_id in available_gpus}
status_lock = threading.Lock()
completed_scenes = []
failed_scenes = []

def get_gpu_memory():
    """Docstring"""
    """Docstring"""
    try:
        result = subprocess.run([
            'nvidia-smi', '--query-gpu=index,memory.used', 
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True, check=True)
        
        gpu_memory = {}
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                gpu_id, memory_used = line.split(', ')
                gpu_memory[int(gpu_id)] = int(memory_used)
        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return {}

def is_gpu_free(gpu_id):
    """Docstring"""
    """Docstring"""
    gpu_memory = get_gpu_memory()
    if gpu_id not in gpu_memory:
        return False
    
    with status_lock:
        return (gpu_memory[gpu_id] < gpu_memory_threshold and 
                not gpu_status[gpu_id])

def assign_gpu():
    """Docstring"""
    """Docstring"""
    with status_lock:
        for gpu_id in available_gpus:
            if not gpu_status[gpu_id]:
                gpu_memory = get_gpu_memory()
                if gpu_id in gpu_memory and gpu_memory[gpu_id] < gpu_memory_threshold:
                    gpu_status[gpu_id] = True
                    return gpu_id
    return None

def release_gpu(gpu_id):
    """Docstring"""
    """Docstring"""
    with status_lock:
        gpu_status[gpu_id] = False

def run_scene(scene, out_base_path, data_base_path, eval_path, out_name):
    """Docstring"""
    """Docstring"""
    gpu_id = None
    try:
        point_cloud_path = f"{out_base_path}/scan{scene}/{out_name}/point_cloud/iteration_30000/point_cloud.ply"
        if os.path.exists(point_cloud_path):
            print(f"✅ Scene {scene}: Point cloud already exists, skipping training")
            print(f"   Path: {point_cloud_path}")
            return True
        
        while gpu_id is None:
            gpu_id = assign_gpu()
            if gpu_id is None:
                print(f"Scene {scene}: Waiting for available GPU...")
                time.sleep(10)
        
        print(f"Scene {scene}: Assigned to GPU {gpu_id}")
        
        cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/scan{scene} -m {out_base_path}/scan{scene}/{out_name} --quiet -r2 --ncc_scale 0.5 '
               f'--use_dpt_loss --use_unmasked_depth '
               f'--use_quadtree_depth_alignment '
               f'--use_boolean_mask '
               f'--use_depth_direction_consistency_loss '
               )
        
        print(f"[GPU {gpu_id}] Training scene {scene}")
        if os.system(cmd) != 0:
            raise Exception(f"Training failed for scene {scene}")
        
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/scan{scene}/{out_name} --quiet --num_cluster 1 --voxel_size 0.002 --max_depth 5.0'
        print(f"[GPU {gpu_id}] Rendering scene {scene}")
        if os.system(cmd) != 0:
            raise Exception(f"Rendering failed for scene {scene}")
        
        cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python scripts/eval_dtu/evaluate_single_scene.py --input_mesh {out_base_path}/scan{scene}/{out_name}/mesh/tsdf_fusion_post.ply --scan_id {scene} --output_dir {out_base_path}/scan{scene}/{out_name}/mesh --mask_dir {data_base_path} --DTU {eval_path}"
        print(f"[GPU {gpu_id}] Evaluating scene {scene}")
        if os.system(cmd) != 0:
            raise Exception(f"Evaluation failed for scene {scene}")
        
        print(f"✅ Scene {scene} completed successfully on GPU {gpu_id}")
        return True
        
    except Exception as e:
        print(f"❌ Scene {scene} failed on GPU {gpu_id}: {e}")
        return False
    finally:
        if gpu_id is not None:
            release_gpu(gpu_id)

def monitor_gpus():
    """Docstring"""
    """Docstring"""
    while True:
        gpu_memory = get_gpu_memory()
        print(f"\n=== GPU Status ===")
        for gpu_id in available_gpus:
            if gpu_id in gpu_memory:
                with status_lock:
                    status = "ASSIGNED" if gpu_status[gpu_id] else "FREE"
                print(f"GPU {gpu_id}: {gpu_memory[gpu_id]}MB ({status})")
        print("==================\n")
        time.sleep(30)

def main():
    """Docstring"""
    """Docstring"""
    parser = argparse.ArgumentParser(description='Multi-GPU parallel training script for DTU dataset')
    parser.add_argument('--out_base_path', type=str, required=True,
                        help='Output base path (e.g., ./output_dtu)')
    parser.add_argument('--data_base_path', type=str, required=True,
                        help='DTU dataset base path (e.g., data/DTU/dtu)')
    parser.add_argument('--eval_path', type=str, required=True,
                        help='DTU evaluation dataset path (e.g., data/DTU/dtu_eval/Points/stl)')
    parser.add_argument('--out_name', type=str, default='test',
                        help='Output directory name (default: test)')
    args = parser.parse_args()
    
    out_base_path = args.out_base_path
    data_base_path = args.data_base_path
    eval_path = args.eval_path
    out_name = args.out_name
    
    print(f"Starting multi-GPU training")
    print(f"Output base path: {out_base_path}")
    print(f"Data base path: {data_base_path}")
    print(f"Eval path: {eval_path}")
    print(f"Output name: {out_name}")
    print(f"Available GPUs: {available_gpus}")
    print(f"Memory threshold: {gpu_memory_threshold}MB")
    print(f"Total scenes: {len(scenes)}")
    
    monitor_thread = threading.Thread(target=monitor_gpus, daemon=True)
    monitor_thread.start()
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_scene = {executor.submit(run_scene, scene, out_base_path, data_base_path, eval_path, out_name): scene for scene in scenes}
        
        for future in as_completed(future_to_scene):
            scene = future_to_scene[future]
            try:
                success = future.result()
                if success:
                    completed_scenes.append(scene)
                else:
                    failed_scenes.append(scene)
            except Exception as e:
                print(f"Scene {scene} failed with exception: {e}")
                failed_scenes.append(scene)
    
    print(f"\n=== Final Results ===")
    print(f"Completed: {sorted(completed_scenes)}")
    print(f"Failed: {sorted(failed_scenes)}")
    print(f"Success rate: {len(completed_scenes)}/{len(scenes)} ({len(completed_scenes)/len(scenes)*100:.1f}%)")
    
    if completed_scenes:
        print(f"\n=== Collecting Results ===")
        collect_cmd = f"python scripts/collect_results.py --input_dir {out_base_path} --output_file ./{out_base_path}.txt"
        print(f"Executing: {collect_cmd}")
        try:
            if os.system(collect_cmd) == 0:
                print("✅ Results collection completed successfully")
            else:
                print("❌ Results collection failed")
        except Exception as e:
            print(f"❌ Error during results collection: {e}")
    else:
        print("⚠️  No completed scenes to collect results from")

if __name__ == "__main__":
    main()
