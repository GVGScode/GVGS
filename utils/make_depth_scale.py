import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scene'))
from colmap_loader import read_extrinsics_text, read_intrinsics_text, read_extrinsics_binary, read_intrinsics_binary, qvec2rotmat

def get_scales(key, cameras, images, points3d_ordered, args):
    """
    Calculate depth scale parameters for a single image
    Args:
        key: Image ID
        cameras: Camera intrinsics
        images: Camera extrinsics
        points3d_ordered: 3D point cloud
        args: Command line arguments
    Returns:
        dict: Dictionary containing scale and offset
    """
    image_meta = images[key]
    cam_intrinsic = cameras[image_meta.camera_id]

    pts_idx = image_meta.point3D_ids

    mask = pts_idx >= 0
    mask *= pts_idx < len(points3d_ordered)

    pts_idx = pts_idx[mask]
    valid_xys = image_meta.xys[mask]

    if len(pts_idx) > 0:
        pts = points3d_ordered[pts_idx]
    else:
        pts = np.array([0, 0, 0])

    R = qvec2rotmat(image_meta.qvec)
    pts = np.dot(pts, R.T) + image_meta.tvec

    invcolmapdepth = 1. / pts[..., 2] 
    n_remove = len(image_meta.name.split('.')[-1]) + 1
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{image_meta.name[:-n_remove]}.png", cv2.IMREAD_UNCHANGED)
    
    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]

    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.height

    maps = (valid_xys * s).astype(np.float32)
    valid = (
        (maps[..., 0] >= 0) * 
        (maps[..., 1] >= 0) * 
        (maps[..., 0] < cam_intrinsic.width * s) * 
        (maps[..., 1] < cam_intrinsic.height * s) * 
        (invcolmapdepth > 0))
    
    if valid.sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        maps = maps[valid, :]
        invcolmapdepth = invcolmapdepth[valid]
        invmonodepth = cv2.remap(invmonodepthmap, maps[..., 0], maps[..., 1], interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)[..., 0]
        
        ## Median / dev
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))

        t_mono = np.median(invmonodepth)
        s_mono = np.mean(np.abs(invmonodepth - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale = 0
        offset = 0
    return {"image_name": image_meta.name[:-n_remove], "scale": scale, "offset": offset}

def load_colmap_data(base_dir):
    """
    Load COLMAP data
    Args:
        base_dir: COLMAP dataset root directory
    Returns:
        tuple: (cameras, images, points3d_ordered)
    """
    cameras_extrinsic_file = os.path.join(base_dir, "sparse/0", "images.bin")
    cameras_intrinsic_file = os.path.join(base_dir, "sparse/0", "cameras.bin")
    points3D_file = os.path.join(base_dir, "sparse/0", "points3D.bin")
    
    try:
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
        points3D = read_points3D_binary(points3D_file)
    except:
        cameras_extrinsic_file = os.path.join(base_dir, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(base_dir, "sparse/0", "cameras.txt")
        points3D_file = os.path.join(base_dir, "sparse/0", "points3D.txt")
        
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        points3D = read_points3D_text(points3D_file)
    
    points3d_ordered = np.array([points3D[key].xyz for key in points3D])
    
    return cam_intrinsics, cam_extrinsics, points3d_ordered

def read_points3D_binary(path_to_model_file):
    """
    Read binary format 3D point file
    """
    import struct
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for point_line_index in range(num_points):
            binary_point_line_properties = struct.unpack("<dddddBBBd", fid.read(43))
            point_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = np.array(binary_point_line_properties[7])
            track_length = struct.unpack("<Q", fid.read(8))[0]
            track_elems = struct.unpack("<" + "ii" * track_length, fid.read(8 * track_length))
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point_id] = Point3D(id=point_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs)
    return points3D

def read_points3D_text(path):
    """
    Read text format 3D point file
    """
    points3D = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                point_id = int(elems[0])
                xyz = np.array(tuple(map(float, elems[1:4])))
                rgb = np.array(tuple(map(int, elems[4:7])))
                error = float(elems[7])
                track_length = int(elems[8])
                track_elems = tuple(map(int, elems[9:9 + 2 * track_length]))
                image_ids = np.array(tuple(map(int, track_elems[0::2])))
                point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
                points3D[point_id] = Point3D(id=point_id, xyz=xyz, rgb=rgb, error=error, image_ids=image_ids, point2D_idxs=point2D_idxs)
    return points3D

from collections import namedtuple
Point3D = namedtuple("Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, required=True, help='Path to COLMAP dataset')
    parser.add_argument('--depths_dir', type=str, required=True, help='Path to generated depth maps')
    args = parser.parse_args()
    
    print(f"Loading COLMAP data from {args.base_dir}")
    cam_intrinsics, cam_extrinsics, points3d_ordered = load_colmap_data(args.base_dir)
    
    print(f"Computing depth scales for {len(cam_extrinsics)} images...")
    depth_param_list = Parallel(n_jobs=-1, backend="threading")(
        delayed(get_scales)(key, cam_intrinsics, cam_extrinsics, points3d_ordered, args) for key in cam_extrinsics
    )

    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param != None
    }

    output_dir = os.path.join(args.base_dir, "sparse", "0")
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, "depth_params.json")
    with open(output_file, "w") as f:
        json.dump(depth_params, f, indent=2)
    
    print(f"Generated depth_params.json with {len(depth_params)} images")
    print(f"Saved to: {output_file}")
    
    scales = [depth_params[key]["scale"] for key in depth_params if depth_params[key]["scale"] > 0]
    if scales:
        print(f"Scale statistics: min={min(scales):.4f}, max={max(scales):.4f}, median={np.median(scales):.4f}")
    else:
        print("No valid scales found!")
