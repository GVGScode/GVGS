#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
COLMAP Sparse Model Reader
Read cameras and images from COLMAP sparse reconstruction
"""

import numpy as np
import struct
from pathlib import Path
import collections


# Camera model types
CAMERA_MODELS = {
    0: "SIMPLE_PINHOLE",
    1: "PINHOLE",
    2: "SIMPLE_RADIAL",
    3: "RADIAL",
    4: "OPENCV",
    5: "OPENCV_FISHEYE",
    6: "FULL_OPENCV",
    7: "FOV",
    8: "SIMPLE_RADIAL_FISHEYE",
    9: "RADIAL_FISHEYE",
    10: "THIN_PRISM_FISHEYE"
}

Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
Image = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file."""
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_text(path):
    """Read cameras.txt"""
    cameras = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                camera_id = int(elems[0])
                model = elems[1]
                width = int(elems[2])
                height = int(elems[3])
                params = np.array(tuple(map(float, elems[4:])))
                cameras[camera_id] = Camera(id=camera_id, model=model,
                                           width=width, height=height,
                                           params=params)
    return cameras


def read_cameras_binary(path):
    """Read cameras.bin"""
    cameras = {}
    with open(path, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            model_name = CAMERA_MODELS[model_id]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = (len(camera_properties) - 4)
            params = read_next_bytes(fid, 8*num_params,
                                    "d"*num_params)
            cameras[camera_id] = Camera(id=camera_id, model=model_name,
                                       width=width, height=height,
                                       params=np.array(params))
    return cameras


def read_images_text(path):
    """Read images.txt"""
    images = {}
    with open(path, "r") as fid:
        while True:
            line = fid.readline()
            if not line:
                break
            line = line.strip()
            if len(line) > 0 and line[0] != "#":
                elems = line.split()
                image_id = int(elems[0])
                qvec = np.array(tuple(map(float, elems[1:5])))
                tvec = np.array(tuple(map(float, elems[5:8])))
                camera_id = int(elems[8])
                image_name = elems[9]
                elems = fid.readline().split()
                xys = np.column_stack([tuple(map(float, elems[0::3])),
                                      tuple(map(float, elems[1::3]))])
                point3D_ids = np.array(tuple(map(int, elems[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
    return images


def read_images_binary(path):
    """Read images.bin"""
    images = {}
    with open(path, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                          format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                       format_char_sequence="ddq"*num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                  tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            images[image_id] = Image(
                id=image_id, qvec=qvec, tvec=tvec,
                camera_id=camera_id, name=image_name,
                xys=xys, point3D_ids=point3D_ids)
    return images


def qvec2rotmat(qvec):
    """Convert quaternion to rotation matrix"""
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])


def read_colmap_sparse(sparse_path):
    """
    Read COLMAP sparse model
    
    Args:
        sparse_path: Path to sparse directory
        
    Returns:
        cameras: Dictionary of Camera objects
        images: Dictionary of Image objects
    """
    sparse_path = Path(sparse_path)
    
    # Try binary format first, then text format
    cameras_bin = sparse_path / "cameras.bin"
    cameras_txt = sparse_path / "cameras.txt"
    images_bin = sparse_path / "images.bin"
    images_txt = sparse_path / "images.txt"
    
    if cameras_bin.exists():
        print(f"Reading cameras from {cameras_bin}")
        cameras = read_cameras_binary(cameras_bin)
    elif cameras_txt.exists():
        print(f"Reading cameras from {cameras_txt}")
        cameras = read_cameras_text(cameras_txt)
    else:
        raise FileNotFoundError(f"No cameras file found in {sparse_path}")
    
    if images_bin.exists():
        print(f"Reading images from {images_bin}")
        images = read_images_binary(images_bin)
    elif images_txt.exists():
        print(f"Reading images from {images_txt}")
        images = read_images_text(images_txt)
    else:
        raise FileNotFoundError(f"No images file found in {sparse_path}")
    
    print(f"Loaded {len(cameras)} cameras and {len(images)} images")
    return cameras, images


def get_camera_params(camera, image):
    """
    Get camera intrinsic and extrinsic parameters
    
    Args:
        camera: Camera object
        image: Image object
        
    Returns:
        K: 3x3 intrinsic matrix
        R: 3x3 rotation matrix (world to camera)
        t: 3x1 translation vector (world to camera)
        camera_center: 3x1 camera center in world coordinates
    """
    # Intrinsic matrix
    if camera.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE"]:
        fx = fy = camera.params[0]
        cx = camera.params[1]
        cy = camera.params[2]
    elif camera.model in ["PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "RADIAL", "RADIAL_FISHEYE"]:
        fx = camera.params[0]
        fy = camera.params[1]
        cx = camera.params[2]
        cy = camera.params[3]
    else:
        raise NotImplementedError(f"Camera model {camera.model} not supported")
    
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    # Extrinsic parameters
    R = qvec2rotmat(image.qvec)
    t = image.tvec.reshape(3, 1)
    
    # Camera center in world coordinates
    camera_center = -R.T @ t
    
    return K, R, t, camera_center

