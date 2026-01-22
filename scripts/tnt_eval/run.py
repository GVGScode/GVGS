# ----------------------------------------------------------------------------
# -                   TanksAndTemples Website Toolbox                        -
# -                    http://www.tanksandtemples.org                        -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2017
# Arno Knapitsch <arno.knapitsch@gmail.com >
# Jaesik Park <syncle@gmail.com>
# Qian-Yi Zhou <Qianyi.Zhou@gmail.com>
# Vladlen Koltun <vkoltun@gmail.com>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
# ----------------------------------------------------------------------------
#
# This python script is for downloading dataset from www.tanksandtemples.org
# The dataset has a different license, please refer to
# https://tanksandtemples.org/license/

# this script requires Open3D python binding
# please follow the intructions in setup.py before running this script.
import numpy as np
import open3d as o3d
import os
import argparse
import csv
from omegaconf import OmegaConf
import torch
from config import scenes_tau_dict
from registration import (
    trajectory_alignment,
    registration_vol_ds,
    registration_unif,
    read_trajectory,
)
from help_func import auto_orient_and_center_poses
from trajectory_io import CameraPose
from evaluation import EvaluateHisto
from util import make_dir
from plot import plot_graph

def load_config(*yaml_files, cli_args=[]):
    yaml_confs = [OmegaConf.load(f) for f in yaml_files]
    cli_conf = OmegaConf.from_cli(cli_args)
    conf = OmegaConf.merge(*yaml_confs, cli_conf)
    OmegaConf.resolve(conf)
    return conf


def run_evaluation(dataset_dir, traj_path, ply_path, out_dir, view_crop=False):
    scene = os.path.basename(os.path.normpath(dataset_dir))

    if scene not in scenes_tau_dict:
        print(dataset_dir, scene)
        raise Exception("invalid dataset-dir, not in scenes_tau_dict")

    print("")
    print("===========================")
    print("Evaluating %s" % scene)
    print("===========================")

    dTau = scenes_tau_dict[scene]
    # put the crop-file, the GT file, the COLMAP SfM log file and
    # the alignment of the according scene in a folder of
    # the same scene name in the dataset_dir
    colmap_ref_logfile = os.path.join(dataset_dir, scene + "_COLMAP_SfM.log")
    alignment = os.path.join(dataset_dir, scene + "_trans.txt")
    gt_filen = os.path.join(dataset_dir, scene + ".ply")
    cropfile = os.path.join(dataset_dir, scene + ".json")
    map_file = os.path.join(dataset_dir, scene + "_mapping_reference.txt")
    if not os.path.isfile(map_file):
        map_file = None
    map_file = None  # Disable map_file as in reference

    make_dir(out_dir)

    # Load reconstruction and according GT
    print(ply_path)
    pcd = o3d.io.read_point_cloud(ply_path)
    # add center points
    import trimesh
    mesh = trimesh.load_mesh(ply_path)
    # add center points
    sampled_vertices = mesh.vertices[mesh.faces].mean(axis=1)
    # add 4 points based on the face vertices
    # face_vertices = mesh.vertices[mesh.faces]# .mean(axis=1)
    # weights = np.array([[3, 3, 3],
    #                     [4, 4, 1],
    #                     [4, 1, 4],
    #                     [1, 4, 4]],dtype=np.float32) / 9.0
    # sampled_vertices = np.sum(face_vertices.reshape(-1, 1, 3, 3) * weights.reshape(1, 4, 3, 1), axis=2).reshape(-1, 3)
    
    vertices = np.concatenate([mesh.vertices, sampled_vertices], axis=0)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    ### end add center points
    print(gt_filen)
    gt_pcd = o3d.io.read_point_cloud(gt_filen)

    gt_trans = np.loadtxt(alignment)
    
    # Support multiple trajectory formats
    traj_to_register = []
    if traj_path.endswith('.npy'):
        ld = np.load(traj_path)
        for i in range(len(ld)):
            traj_to_register.append(CameraPose(meta=None, mat=ld[i]))
    elif traj_path.endswith('.json'):  # instant-npg or sdfstudio format
        import json
        with open(traj_path, encoding='UTF-8') as f:
            meta = json.load(f)
        poses_dict = {}
        for i, frame in enumerate(meta['frames']):
            filepath = frame['file_path']
            new_i = int(filepath[13:18]) - 1
            poses_dict[new_i] = np.array(frame['transform_matrix'])
        poses = []
        for i in range(len(poses_dict)):
            poses.append(poses_dict[i])
        poses = torch.from_numpy(np.array(poses).astype(np.float32))
        poses, _ = auto_orient_and_center_poses(poses, method='up', center_poses=True)
        scale_factor = 1.0 / float(torch.max(torch.abs(poses[:, :3, 3])))
        poses[:, :3, 3] *= scale_factor
        poses = poses.numpy()
        for i in range(len(poses)):
            traj_to_register.append(CameraPose(meta=None, mat=poses[i]))
    else:
        traj_to_register = read_trajectory(traj_path)
    
    print(colmap_ref_logfile)
    gt_traj_col = read_trajectory(colmap_ref_logfile)

    trajectory_transform = trajectory_alignment(map_file, traj_to_register,
                                                gt_traj_col, gt_trans, scene)

    # Refine alignment by using the actual GT and MVS pointclouds
    vol = o3d.visualization.read_selection_polygon_volume(cropfile)
    # big pointclouds will be downlsampled to this number to speed up alignment
    dist_threshold = dTau

    # Registration refinment in 3 iterations
    r2 = registration_vol_ds(pcd, gt_pcd, trajectory_transform, vol, dTau,
                             dTau * 80, 40)
    r3 = registration_vol_ds(pcd, gt_pcd, r2.transformation, vol, dTau / 2.0,
                             dTau * 20, 40)
    r = registration_unif(pcd, gt_pcd, r3.transformation, vol, 2 * dTau, 40)
    trajectory_transform = r.transformation
    
    # Histogramms and P/R/F1
    plot_stretch = 5
    [
        precision,
        recall,
        fscore,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
    ] = EvaluateHisto(
        pcd,
        gt_pcd,
        trajectory_transform, # r.transformation,
        vol,
        dTau / 2.0,
        dTau,
        out_dir,
        plot_stretch,
        scene,
        view_crop
    )
    eva = [precision, recall, fscore]
    print("==============================")
    print("evaluation result : %s" % scene)
    print("==============================")
    print("distance tau : %.3f" % dTau)
    print("precision : %.4f" % eva[0])
    print("recall : %.4f" % eva[1])
    print("f-score : %.4f" % eva[2])
    print("==============================")

    # Plotting
    plot_graph(
        scene,
        fscore,
        dist_threshold,
        edges_source,
        cum_source,
        edges_target,
        cum_target,
        plot_stretch,
        out_dir,
    )
    
    out_file = os.path.join(out_dir, 'result.csv')
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        name = ['precision','recall','fscore']
        writer.writerow(name)
        writer.writerow(eva)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', default='./config/base.yaml',
                        help='path to config yaml file')
    parser.add_argument(
        "--view-crop",
        type=int,
        default=0,
        help="whether view the crop pointcloud after aligned",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="path to a dataset/scene directory containing X.json, X.ply, ...",
    )
    parser.add_argument(
        "--traj-path",
        type=str,
        default=None,
        help=
        "path to trajectory file. See `convert_to_logfile.py` to create this file.",
    )
    parser.add_argument(
        "--ply-path",
        type=str,
        default=None,
        help="path to reconstruction ply file",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="",
        help=
        "output directory, default: an evaluation directory is created in the directory of the ply file",
    )
    parser.add_argument(
        "-m",
        "--model-path",
        type=str,
        default=None,
        help="model path (alternative to conf_path)",
    )
    args, extras = parser.parse_known_args()
    
    # Support both config file mode and direct argument mode
    if args.conf_path and os.path.exists(args.conf_path):
        config = load_config(args.conf_path, cli_args=extras)
        model_path = config.get('load_model_path', None) or args.model_path
        
        # If -m parameter is provided, override model_path and update mesh path
        if args.model_path:
            model_path = args.model_path
            # Try to find mesh in test/mesh/tsdf_fusion_post.ply first
            test_mesh_path = os.path.join(model_path, "test/mesh/tsdf_fusion_post.ply")
            if os.path.exists(test_mesh_path):
                config.mesh_ply_path = test_mesh_path
                config.eval_output_path = os.path.join(model_path, "test/mesh/evaluation")
            else:
                # Try to find mesh in config.yaml if exists
                if os.path.exists(f"{model_path}/config.yaml"):
                    model_config = load_config(f"{model_path}/config.yaml", cli_args=extras)
                    mesh_path_from_config = model_config.get('mesh_ply_path', None)
                    if mesh_path_from_config and os.path.exists(mesh_path_from_config):
                        config.mesh_ply_path = mesh_path_from_config
                        config.eval_output_path = model_config.get('eval_output_path', config.eval_output_path)
                    else:
                        # If config.yaml exists but mesh path is invalid, try to find mesh in common locations
                        possible_mesh_paths = [
                            os.path.join(model_path, "test/mesh/tsdf_fusion.ply"),
                            os.path.join(model_path, "mesh/tsdf_fusion_post.ply"),
                            os.path.join(model_path, "mesh/tsdf_fusion.ply"),
                        ]
                        found = False
                        for possible_path in possible_mesh_paths:
                            if os.path.exists(possible_path):
                                config.mesh_ply_path = possible_path
                                config.eval_output_path = os.path.join(os.path.dirname(possible_path), "evaluation")
                                found = True
                                break
                        if not found:
                            # Last resort: use config from original config file but warn
                            print(f"Warning: Could not find mesh file in {model_path}, using config file path")
                            print(f"  This may fail if the config path doesn't exist")
                else:
                    # No config.yaml, try common mesh locations
                    possible_mesh_paths = [
                        os.path.join(model_path, "test/mesh/tsdf_fusion.ply"),
                        os.path.join(model_path, "mesh/tsdf_fusion_post.ply"),
                        os.path.join(model_path, "mesh/tsdf_fusion.ply"),
                    ]
                    found = False
                    for possible_path in possible_mesh_paths:
                        if os.path.exists(possible_path):
                            config.mesh_ply_path = possible_path
                            config.eval_output_path = os.path.join(os.path.dirname(possible_path), "evaluation")
                            found = True
                            break
                    if not found:
                        print(f"Error: Could not find mesh file in {model_path}")
                        print(f"  Tried: {possible_mesh_paths}")
                        raise FileNotFoundError(f"Mesh file not found in {model_path}")
        elif model_path and os.path.exists(f"{model_path}/config.yaml"):
            config = load_config(f"{model_path}/config.yaml", cli_args=extras)
        
        args.view_crop = False  # (args.view_crop > 0)
        
        if config.get('eval_output_path', '').strip() == "":
            config.eval_output_path = os.path.join(
                os.path.dirname(config.mesh_ply_path), "evaluation")
        
        print("version of o3d:", o3d.__version__)
        run_evaluation(
            dataset_dir=config.dataset_GT_path,
            traj_path=config.traj_path,
            ply_path=config.mesh_ply_path,
            out_dir=config.eval_output_path,
            view_crop=args.view_crop
        )
    elif args.model_path:
        # Support -m parameter directly without conf_path
        model_path = args.model_path
        scene_name = os.path.basename(os.path.normpath(model_path))
        
        # Try to find mesh in test/mesh/tsdf_fusion_post.ply first
        test_mesh_path = os.path.join(model_path, "test/mesh/tsdf_fusion_post.ply")
        if os.path.exists(test_mesh_path):
            mesh_ply_path = test_mesh_path
            eval_output_path = os.path.join(model_path, "test/mesh/evaluation")
            dataset_GT_path = f"/path/to/Datasets/TNT_GOF/GT/{scene_name}"
            traj_path = f"/path/to/Datasets/TNT_GOF/TrainingSet/{scene_name}/{scene_name}_COLMAP_SfM.log"
        elif os.path.exists(f"{model_path}/config.yaml"):
            # Load from config.yaml if exists
            config = load_config(f"{model_path}/config.yaml", cli_args=extras)
            mesh_ply_path = config.get('mesh_ply_path', test_mesh_path)
            eval_output_path = config.get('eval_output_path', os.path.join(model_path, "test/mesh/evaluation"))
            dataset_GT_path = config.get('dataset_GT_path', f"/path/to/Datasets/TNT_GOF/GT/{scene_name}")
            traj_path = config.get('traj_path', f"/path/to/Datasets/TNT_GOF/TrainingSet/{scene_name}/{scene_name}_COLMAP_SfM.log")
        else:
            # Default path
            mesh_ply_path = test_mesh_path
            eval_output_path = os.path.join(model_path, "test/mesh/evaluation")
            dataset_GT_path = f"/path/to/Datasets/TNT_GOF/GT/{scene_name}"
            traj_path = f"/path/to/Datasets/TNT_GOF/TrainingSet/{scene_name}/{scene_name}_COLMAP_SfM.log"
        
        if eval_output_path.strip() == "":
            eval_output_path = os.path.join(os.path.dirname(mesh_ply_path), "evaluation")
        
        print("version of o3d:", o3d.__version__)
        run_evaluation(
            dataset_dir=dataset_GT_path,
            traj_path=traj_path,
            ply_path=mesh_ply_path,
            out_dir=eval_output_path,
            view_crop=False
        )
    else:
        # Fallback to direct argument mode
        if not all([args.dataset_dir, args.traj_path, args.ply_path]):
            parser.error("Either --conf_path or --dataset-dir, --traj-path, and --ply-path must be provided")
        
        if args.out_dir.strip() == "":
            args.out_dir = os.path.join(os.path.dirname(args.ply_path),
                                        "evaluation")

        run_evaluation(
            dataset_dir=args.dataset_dir,
            traj_path=args.traj_path,
            ply_path=args.ply_path,
            out_dir=args.out_dir,
            view_crop=(args.view_crop > 0)
        )
