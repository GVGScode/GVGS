#
# This file is adapted from a publicly available 3D Gaussian Splatting implementation.
# All identifying information has been removed to preserve double-blind review.
#
# The original license applies to this code.
#

from scene.cameras import Camera
import numpy as np
from utils.graphics_utils import fov2focal
import sys
import cv2
import torch
import os

WARNED = False

def load_depth_map(image_path, resolution, use_dpt_loss=False, use_unmasked_depth=False, depth_params=None, da3mono=False):
    """
    Load depth map, reference CityGS-X implementation
    Args:
        image_path: Image path
        resolution: Resolution
        use_dpt_loss: Whether to use DPT loss
        use_unmasked_depth: Whether to use unmasked depth (directly use depth maps from train/depths directory)
        depth_params: Depth parameters, containing scale and offset
        da3mono: Whether to use DA3 depth (depths_da3_2 directory)
    """
    if not use_dpt_loss:
        return None, None, None
    if "MatrixCity" in image_path:
        depth_path = image_path.replace("images/", "depth/", 1)
        mask_path = image_path.replace('images', 'mask')
        mask_path = mask_path.replace('jpg', 'png')
    else:
        if da3mono:
            depth_path = image_path.replace('images', 'train/depths_da3_2')
        else:
            depth_path = image_path.replace('images', 'train/depths')
        mask_path = image_path.replace('images', 'train/mask')
        mask_path = mask_path.replace('jpg', 'png')
    
    depth_path = depth_path.replace('jpg', 'png')
    
    depth_reliable = None
    invdepthmap = None
    depth_mask = None
    
    if os.path.exists(depth_path) and use_dpt_loss:
        invdepthmap = cv2.imread(depth_path, -1).astype(np.float32) / float(2**16)
        invdepthmap = cv2.resize(invdepthmap, resolution)
        depth_reliable = True
        
        invdepthmap[invdepthmap < 0] = 0
        
        if invdepthmap.ndim != 2:
            invdepthmap = invdepthmap[..., 0]
        
        if depth_params is not None and "scale" in depth_params and "offset" in depth_params:
            if depth_params["scale"] > 0:
                invdepthmap = invdepthmap * depth_params["scale"] + depth_params["offset"]
                invdepthmap[invdepthmap < 0] = 0
                print(f"Applied depth scaling: scale={depth_params['scale']:.4f}, offset={depth_params['offset']:.4f}")
        
        invdepthmap = torch.from_numpy(invdepthmap[None])
        
        if not use_unmasked_depth and os.path.exists(mask_path) and use_dpt_loss:
            mask_color = cv2.imread(mask_path, -1).astype(np.float32)
            mask_color = cv2.resize(mask_color, resolution)
            mask = np.any(mask_color != [0, 0, 0], axis=-1)
            mask = torch.from_numpy(mask)
            invdepthmap[mask.unsqueeze(0) == 0] = 0
            depth_mask = mask
        else:
            depth_mask = None
    else:
        depth_reliable = None
        invdepthmap = None
        depth_mask = None
    
    return depth_reliable, invdepthmap, depth_mask

def loadCam(args, id, cam_info, resolution_scale, opt=None):
    orig_w, orig_h = cam_info.width, cam_info.height
    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global_down = orig_w / 1600
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    print(f"scale {float(global_down) * float(resolution_scale)}")
                    WARNED = True
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    sys.stdout.write('\r')
    sys.stdout.write("load camera {}".format(id))
    sys.stdout.flush()

    use_dpt_loss = getattr(opt, 'use_dpt_loss', False) if opt else getattr(args, 'use_dpt_loss', False)
    use_unmasked_depth = getattr(opt, 'use_unmasked_depth', False) if opt else getattr(args, 'use_unmasked_depth', False)
    da3mono = getattr(args, 'da3mono', False)
    
    depth_reliable, invdepthmap, depth_mask = load_depth_map(
        cam_info.image_path, resolution, 
        use_dpt_loss,
        use_unmasked_depth,
        cam_info.depth_params,
        da3mono
    )

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image_width=resolution[0], image_height=resolution[1],
                  image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=cam_info.global_id, 
                  preload_img=args.preload_img, 
                  ncc_scale=args.ncc_scale,
                  data_device=args.data_device,
                  depth_reliable=depth_reliable,
                  invdepthmap=invdepthmap,
                  depth_mask=depth_mask,
                  depth_params=cam_info.depth_params)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, opt=None):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale, opt))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
