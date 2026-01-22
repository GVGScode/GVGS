#!/usr/bin/env python3
"""
Multi-GPU parallel training script for MipNeRF360 dataset
Supports GPU status checking and dynamic allocation
"""

import os
import subprocess
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill']
factors = ['4', '2', '2', '4', '4', '2', '2', '4', '4']
data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda', 'cuda']
data_base_path = '/path/to/Data/mipnerf360'
out_base_path = 'output_mip360'
out_name = 'test'

available_gpus = [7,6,5,4,3]
gpu_memory_threshold = 1024
max_concurrent = 3

gpu_status = {gpu_id: False for gpu_id in available_gpus}
status_lock = threading.Lock()
completed_scenes = []
failed_scenes = []

def get_gpu_memory():
    """Get GPU memory usage"""
    """Get GPU memory usage"""
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
    """Check if GPU is free"""
    """Check if GPU is free"""
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

def run_scene(scene, scene_id):
    """Docstring"""
    """Docstring"""
    gpu_id = None
    try:
        while gpu_id is None:
            gpu_id = assign_gpu()
            if gpu_id is None:
                print(f"Scene {scene}: Waiting for available GPU...")
                time.sleep(10)
        
        print(f"Scene {scene}: Assigned to GPU {gpu_id}")
        
        
        cmd = f'rm -rf {out_base_path}/{scene}/{out_name}/*'
        print(f"[GPU {gpu_id}] {cmd}")
        os.system(cmd)

        cmd = (f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py -s {data_base_path}/{scene} -m {out_base_path}/{scene}/{out_name} --quiet -r{factors[scene_id]} --data_device {data_devices[scene_id]} --densify_abs_grad_threshold 0.0002 --eval '
               f'--use_dpt_loss --use_unmasked_depth '
               f'--use_quadtree_depth_alignment '
            #    f'--use_da3_init '
               f'--use_boolean_mask '
               f'--use_depth_direction_consistency_loss'
               )
        print(f"[GPU {gpu_id}] Training scene {scene}")
        print(f"[GPU {gpu_id}] {cmd}")
        if os.system(cmd) != 0:
            raise Exception(f"Training failed for scene {scene}")

        common_args = f"--quiet --skip_train"
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python render.py -m {out_base_path}/{scene}/{out_name} {common_args}'
        print(f"[GPU {gpu_id}] Rendering scene {scene}")
        print(f"[GPU {gpu_id}] {cmd}")
        if os.system(cmd) != 0:
            raise Exception(f"Rendering failed for scene {scene}")
        
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python metrics.py -m {out_base_path}/{scene}/{out_name}'
        print(f"[GPU {gpu_id}] Evaluating scene {scene}")
        print(f"[GPU {gpu_id}] {cmd}")
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
    print(f"Starting multi-GPU training for MipNeRF360")
    print(f"Output base path: {out_base_path}")
    print(f"Available GPUs: {available_gpus}")
    print(f"Memory threshold: {gpu_memory_threshold}MB")
    print(f"Max concurrent: {max_concurrent}")
    print(f"Total scenes: {len(scenes)}")
    
    monitor_thread = threading.Thread(target=monitor_gpus, daemon=True)
    monitor_thread.start()
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_scene = {executor.submit(run_scene, scene, id): (scene, id) for id, scene in enumerate(scenes)}
        
        for future in as_completed(future_to_scene):
            scene, scene_id = future_to_scene[future]
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

if __name__ == "__main__":
    main()
