#!/usr/bin/env python3
"""
"""

import os
import subprocess
import time
import threading
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

scenes = ['Courthouse', 'Truck', 'Caterpillar', 'Barn', 'Meetingroom', 'Ignatius']
data_devices = ['cpu', 'cuda', 'cuda', 'cuda', 'cuda', 'cpu']

scene_mono_dpt_params = {
    'Barn': {'near': 3.5, 'far': 6.0},
    'Truck': {'near': 5.0, 'far': 8.0},
    'Caterpillar': {'near': 3.5, 'far': 6.0},
    'Ignatius': {'near': 3.5, 'far': 6.0},
    'Meetingroom': {'near': 3.5, 'far': 10.0},
    'Courthouse': {'near': 5.5, 'far': 8.0},
}

def load_scene_params(scene):
    """Load scene parameters from default dictionary"""
    return scene_mono_dpt_params.get(scene, {'near': 5.0, 'far': 8.0})

# scenes = ['Courthouse']
# data_devices = ['cpu']

available_gpus = [ 2, 1, 0]
gpu_memory_threshold = 1024
max_concurrent = 3

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

def run_scene(scene, data_device, data_base_path, out_base_path, out_name, eval_path):
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
        
        scene_params = load_scene_params(scene)
        mono_dpt_near = scene_params['near']
        mono_dpt_far = scene_params['far']
        print(f"Scene {scene}: Using mono_dpt_trunc_near={mono_dpt_near}, mono_dpt_trunc_far={mono_dpt_far}")
        
        cmd = (
            f'CUDA_VISIBLE_DEVICES={gpu_id} python train.py '
            f'-s {data_base_path}/{scene} '
            f'-m {out_base_path}/{scene}/{out_name} '
            f'-r2 --ncc_scale 0.5 --data_device {data_device} '
            f'--densify_abs_grad_threshold 0.00015 '
            f'--opacity_cull_threshold 0.05 '
            f'--exposure_compensation '
            f'--use_quadtree_depth_alignment '
            f'--use_dpt_loss --use_unmasked_depth '
            f'--use_depth_percentile_filter --depth_percentile_threshold 0.25 '
            f'--mono_dpt_trunc_near {mono_dpt_near} --mono_dpt_trunc_far {mono_dpt_far}'
        )
        print("Training cmd is :",cmd)
        print(f"[GPU {gpu_id}] Training scene {scene}")
        if os.system(cmd) != 0:
            raise Exception(f"Training failed for scene {scene}")
        
        if scene == 'Meetingroom':
            voxel_size = '0.006'
            max_depth = '4.5'
        elif scene == 'Courthouse':
            voxel_size = '0.006'
            max_depth = '10.0'
        else:
            voxel_size = '0.004'
            max_depth = '3.5'
        
        cmd = f'CUDA_VISIBLE_DEVICES={gpu_id} python scripts/render_tnt.py -m {out_base_path}/{scene}/{out_name} --data_device {data_device} --num_cluster 1 --use_depth_filter --voxel_size {voxel_size} --max_depth {max_depth}'
        print(f"[GPU {gpu_id}] Rendering scene {scene}")
        if os.system(cmd) != 0:
            raise Exception(f"Rendering failed for scene {scene}")
        
        cmd = f'python scripts/tnt_eval/run.py --conf_path ./config/TNT_{scene}.yaml -m {out_base_path}/{scene} dataset_GT_path={eval_path}/{scene}'
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
    parser = argparse.ArgumentParser(description='Multi-GPU parallel training script for TNT dataset')
    parser.add_argument('--out_base_path', type=str, required=True,
                        help='Output base path (e.g., ./output_tnt)')
    parser.add_argument('--data_base_path', type=str, required=True,
                        help='Data base path (e.g., data/TNT/TrainingSet)')
    parser.add_argument('--eval_path', type=str, required=True,
                        help='Evaluation path (e.g., data/TNT/GT)')
    parser.add_argument('--out_name', type=str, default='test',
                        help='Output directory name (default: test)')
    args = parser.parse_args()
    
    out_base_path = args.out_base_path
    data_base_path = args.data_base_path
    eval_path = args.eval_path
    out_name = args.out_name
    
    print(f"Starting multi-GPU training for TNT dataset")
    print(f"Data base path: {data_base_path}")
    print(f"Output base path: {out_base_path}")
    print(f"Evaluation path: {eval_path}")
    print(f"Output name: {out_name}")
    print(f"Available GPUs: {available_gpus}")
    print(f"Memory threshold: {gpu_memory_threshold}MB")
    print(f"Max concurrent: {max_concurrent}")
    print(f"Total scenes: {len(scenes)}")
    
    monitor_thread = threading.Thread(target=monitor_gpus, daemon=True)
    monitor_thread.start()
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        future_to_scene = {
            executor.submit(run_scene, scene, data_devices[idx], data_base_path, out_base_path, out_name, eval_path): scene 
            for idx, scene in enumerate(scenes)
        }
        
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
    
    if len(completed_scenes) > 0:
        print(f"\n=== Collecting Evaluation Results ===")
        results_dir = "results"
        os.makedirs(results_dir, exist_ok=True)
        results_filename = f"{os.path.basename(os.path.normpath(out_base_path))}_results.txt"
        output_file = os.path.join(results_dir, results_filename)
        
        collect_cmd = f'python scripts/collect_tnt_results.py --input_dir {out_base_path} --output_file {output_file}'
        print(f"Running: {collect_cmd}")
        if os.system(collect_cmd) == 0:
            print(f"✅ Results collected successfully: {output_file}")
        else:
            print(f"⚠️  Warning: Failed to collect results")

if __name__ == "__main__":
    main()
