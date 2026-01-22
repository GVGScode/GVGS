#
# This file is adapted from a publicly available 3D Gaussian Splatting implementation.
# All identifying information has been removed to preserve double-blind review.
#
# The original license applies to this code.
#


import os
from datetime import datetime
import torch
import random
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim, lncc, get_img_grad_weight
from utils.general_utils import get_expon_lr_func
from utils.graphics_utils import patch_offsets, patch_warp
from gaussian_renderer import render, network_gui, render_with_view_influence_filter, render_with_mask, render_with_color_mask
import sys, time
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import cv2
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, erode
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.app_model import AppModel
from scene.cameras import Camera
import torchvision

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
import time
import torch.nn.functional as F

def setup_seed(seed):
     """Set all random seeds for reproducibility"""
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
     
     # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
     # torch.use_deterministic_algorithms(True)
RANDOM_SEED = 22
setup_seed(RANDOM_SEED)

def quadtree_depth_alignment(mono_invdepth, gaussian_invdepth, mask, split_level, 
                            min_valid_pixels=10, min_depth_range=1e-3, geo_mask=None,
                            filter_outliers=True, outlier_threshold_method='otsu'):
    """
    Align monocular depth and Gaussian depth using quadtree-based statistical method
    
    Args:
        mono_invdepth: Monocular inverse depth map (H, W)
        gaussian_invdepth: Gaussian rendered inverse depth map (H, W)
        mask: Valid region mask (H, W)
        split_level: Split level (0 for full image, 1 for 4 blocks, 2 for 16 blocks, ...)
        min_valid_pixels: Minimum valid pixels per block, default 10
        min_depth_range: Minimum depth range threshold, default 1e-3
        geo_mask: Geometric consistency mask (H, W), optional. If provided, only use geometrically consistent pixels for calibration
        filter_outliers: Whether to filter background regions with large differences, default True
        outlier_threshold_method: Difference threshold method, options:
            'otsu' (adaptive threshold, recommended), 'percentile_90', 'percentile_95', 'percentile_99', 
            'mean_2std', 'mean_3std'
    
    Returns:
        corrected_mono_invdepth: Calibrated monocular inverse depth map (H, W)
    """
    H, W = mono_invdepth.shape
    device = mono_invdepth.device
    
    with torch.no_grad():
        corrected_mono_invdepth = mono_invdepth.clone()
    
    num_splits = 2 ** split_level
    
    block_h = H // num_splits
    block_w = W // num_splits
    
    for i in range(num_splits):
        for j in range(num_splits):
            h_start = i * block_h
            h_end = (i + 1) * block_h if i < num_splits - 1 else H
            w_start = j * block_w
            w_end = (j + 1) * block_w if j < num_splits - 1 else W
            
            block_mono = mono_invdepth[h_start:h_end, w_start:w_end]
            block_gaussian = gaussian_invdepth[h_start:h_end, w_start:w_end]
            block_mask = mask[h_start:h_end, w_start:w_end]
            
            if geo_mask is not None:
                block_geo_mask = geo_mask[h_start:h_end, w_start:w_end]
                block_mask = block_mask & block_geo_mask
            
            valid_count = block_mask.sum()
            if valid_count > min_valid_pixels:
                valid_mono = block_mono[block_mask]
                valid_gaussian = block_gaussian[block_mask]
                
                if (valid_gaussian.max() - valid_gaussian.min()) > min_depth_range:
                    if filter_outliers and valid_mono.numel() > min_valid_pixels:
                        diff = torch.abs(valid_mono - valid_gaussian)
                        
                        if outlier_threshold_method == 'otsu':
                            with torch.no_grad():
                                diff_min = diff.min().item()
                                diff_max = diff.max().item()
                            
                            if diff_max > diff_min + 1e-6:
                                diff_cpu = diff.detach().cpu().numpy()
                                diff_normalized = ((diff_cpu - diff_min) / (diff_max - diff_min) * 255.0).astype(np.uint8)
                                del diff_cpu
                                
                                otsu_thresh_uint8, _ = cv2.threshold(diff_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                                del diff_normalized
                                
                                threshold_value = diff_min + (otsu_thresh_uint8 / 255.0) * (diff_max - diff_min)
                                threshold = torch.tensor(threshold_value, device=diff.device)
                            else:
                                threshold = diff.mean() + 2 * diff.std()
                        elif outlier_threshold_method == 'percentile_90':
                            threshold = torch.quantile(diff, 0.90)
                        elif outlier_threshold_method == 'percentile_95':
                            threshold = torch.quantile(diff, 0.95)
                        elif outlier_threshold_method == 'percentile_99':
                            threshold = torch.quantile(diff, 0.99)
                        elif outlier_threshold_method == 'mean_2std':
                            threshold = diff.mean() + 2 * diff.std()
                        elif outlier_threshold_method == 'mean_3std':
                            threshold = diff.mean() + 3 * diff.std()
                        else:
                            threshold = torch.quantile(diff, 0.90)
                        
                        diff_mask = diff <= threshold
                        
                        if diff_mask.sum() > min_valid_pixels:
                            valid_mono = valid_mono[diff_mask]
                            valid_gaussian = valid_gaussian[diff_mask]
                    
                    t_gaussian = torch.median(valid_gaussian)
                    s_gaussian = torch.mean(torch.abs(valid_gaussian - t_gaussian))
                    
                    t_mono = torch.median(valid_mono)
                    s_mono = torch.mean(torch.abs(valid_mono - t_mono))
                    
                    if s_mono > 1e-6:
                        scale = s_gaussian / s_mono
                        offset = t_gaussian - t_mono * scale
                        
                        corrected_block = block_mono * scale + offset
                        corrected_mono_invdepth[h_start:h_end, w_start:w_end] = corrected_block
                        del corrected_block
                    
                    del valid_mono, valid_gaussian
                    if filter_outliers and 'diff' in locals():
                        del diff, diff_mask
    
    return corrected_mono_invdepth

def draw_quadtree_grid(image, split_level, color=(0, 255, 0), thickness=2):
    """
    Draw quadtree grid lines on image
    
    Args:
        image: Input image (H, W, 3) numpy array
        split_level: Split level
        color: Grid line color (B, G, R)
        thickness: Grid line thickness
    
    Returns:
        Image with grid lines
    """
    H, W = image.shape[:2]
    result = image.copy()
    
    num_splits = 2 ** split_level
    
    if num_splits <= 1:
        return result
    
    for i in range(1, num_splits):
        y = i * H // num_splits
        cv2.line(result, (0, y), (W, y), color, thickness)
    
    for j in range(1, num_splits):
        x = j * W // num_splits
        cv2.line(result, (x, 0), (x, H), color, thickness)
    
    return result

def gen_virtul_cam(cam, trans_noise=1.0, deg_noise=15.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = cam.R.transpose()
    Rt[:3, 3] = cam.T
    Rt[3, 3] = 1.0
    C2W = np.linalg.inv(Rt)

    translation_perturbation = np.random.uniform(-trans_noise, trans_noise, 3)
    rotation_perturbation = np.random.uniform(-deg_noise, deg_noise, 3)
    rx, ry, rz = np.deg2rad(rotation_perturbation)
    Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])
    R_perturbation = Rz @ Ry @ Rx

    C2W[:3, :3] = C2W[:3, :3] @ R_perturbation
    C2W[:3, 3] = C2W[:3, 3] + translation_perturbation
    Rt = np.linalg.inv(C2W)
    virtul_cam = Camera(100000, Rt[:3, :3].transpose(), Rt[:3, 3], cam.FoVx, cam.FoVy,
                        cam.image_width, cam.image_height,
                        cam.image_path, cam.image_name, 100000,
                        trans=np.array([0.0, 0.0, 0.0]), scale=1.0, 
                        preload_img=False, data_device = "cuda")
    return virtul_cam

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, load_iteration=None, load_model_path=None):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    # backup main code
    cmd = f'cp ./train.py {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./arguments {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./gaussian_renderer {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./scene {dataset.model_path}/'
    os.system(cmd)
    cmd = f'cp -rf ./utils {dataset.model_path}/'
    os.system(cmd)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, load_model_path=load_model_path, opt=opt)
    gaussians.training_setup(opt)

    app_model = AppModel()
    app_model.train()
    app_model.cuda()
    
    cache_device = torch.device(dataset.data_device)
    quadtree_cache = {}
    current_cached_split_level = -1
    quadtree_cache_hits = 0
    quadtree_cache_misses = 0
    
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        app_model.load_weights(scene.model_path)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_single_view_for_log = 0.0
    ema_multi_view_geo_for_log = 0.0
    ema_multi_view_pho_for_log = 0.0
    ema_dpt_loss_for_log = 0.0
    ema_opacity_loss_for_log = 0.0
    normal_loss, geo_loss, ncc_loss, dpt_loss, opacity_loss = None, None, None, None, None
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    debug_path = os.path.join(scene.model_path, "debug")
    os.makedirs(debug_path, exist_ok=True)

    for iteration in range(first_iter, opt.iterations + 1):

        iter_start.record()
        gaussians.update_learning_rate(iteration)
        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        gt_image, gt_image_gray = viewpoint_cam.get_image()
        if iteration > 1000 and opt.exposure_compensation:
            gaussians.use_app = True
        
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, app_model=app_model,
                            return_plane=iteration>opt.single_view_weight_from_iter, return_depth_normal=iteration>opt.single_view_weight_from_iter)
        image, viewspace_point_tensor, visibility_filter, radii = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        
        
        # Loss
        ssim_loss = (1.0 - ssim(image, gt_image))
        if 'app_image' in render_pkg and ssim_loss < 0.5:
            app_image = render_pkg['app_image']
            Ll1 = l1_loss(app_image, gt_image)
        else:
            Ll1 = l1_loss(image, gt_image)
        image_loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
        loss = image_loss.clone()
        
        # scale loss
        if visibility_filter.sum() > 0:
            scale = gaussians.get_scaling[visibility_filter]
            sorted_scale, _ = torch.sort(scale, dim=-1)
            min_scale_loss = sorted_scale[...,0]
            loss += opt.scale_loss_weight * min_scale_loss.mean()
        # single-view loss
        if iteration > opt.single_view_weight_from_iter:
            weight = opt.single_view_weight
            normal = render_pkg["rendered_normal"]
            depth_normal = render_pkg["depth_normal"]

            image_weight = (1.0 - get_img_grad_weight(gt_image))
            image_weight = (image_weight).clamp(0,1).detach() ** 2
            
            if opt.use_boolean_mask:
                abs_diff = (depth_normal - normal).abs().sum(0)
                threshold = torch.quantile(abs_diff, 0.7)

                mask = abs_diff < threshold
                if mask.sum() > 0:
                    if not opt.wo_image_weight:
                        selected_pixels = (image_weight * abs_diff)[mask]
                        normal_loss = weight * selected_pixels.mean()
                    else:
                        selected_pixels = abs_diff[mask]
                        normal_loss = weight * selected_pixels.mean()
                else:
                    normal_loss = weight * torch.tensor(0.0, device=abs_diff.device)
                loss += normal_loss
            else:
                if not opt.wo_image_weight:
                    normal_loss = weight * (image_weight * (((depth_normal - normal)).abs().sum(0))).mean()
                else:
                    normal_loss = weight * (((depth_normal - normal)).abs().sum(0)).mean()
                loss += (normal_loss)

        if opt.use_gs_opacity_loss and iteration > opt.gs_opacity_loss_from_iter:
            gs_opacity = gaussians.get_opacity  # [N_gaussians, 1]
            gs_opacity_target = torch.ones_like(gs_opacity)
            gs_opacity_loss = opt.gs_opacity_loss_weight * torch.abs(gs_opacity - gs_opacity_target).mean()
            loss += gs_opacity_loss
        else:
            gs_opacity_loss = None

        if opt.use_opacity_loss and iteration > opt.opacity_loss_from_iter:
            opacity = render_pkg['opacity']  # [H, W]
            opacity_target = torch.ones_like(opacity)
            opacity_loss_base = torch.abs(opacity - opacity_target).mean()
            
            opacity_weight_func = get_expon_lr_func(
                opt.opacity_loss_weight_init, 
                opt.opacity_loss_weight_final, 
                max_steps=opt.opacity_loss_end_iter
            )
            current_opacity_weight = opacity_weight_func(iteration)
            opacity_loss = current_opacity_weight * opacity_loss_base
            loss += opacity_loss
        else:
            opacity_loss = None

        invDepth = None
        mono_invdepth = None
        corrected_mono_invdepth = None
        dpt_loss = None

        # multi-view loss

        geo_consistency_mask = None
        if iteration > opt.multi_view_weight_from_iter:
            nearest_cam = None if len(viewpoint_cam.nearest_id) == 0 else scene.getTrainCameras()[random.sample(viewpoint_cam.nearest_id,1)[0]]
            # print(f"Current viewpoint image name: {viewpoint_cam.image_name}; Nearest viewpoint image name: {nearest_cam.image_name if nearest_cam is not None else 'None'}")
            use_virtul_cam = False
            if opt.use_virtul_cam and (np.random.random() < opt.virtul_cam_prob or nearest_cam is None):
                nearest_cam = gen_virtul_cam(viewpoint_cam, trans_noise=dataset.multi_view_max_dis, deg_noise=dataset.multi_view_max_angle)
                use_virtul_cam = True
            if nearest_cam is not None:
                patch_size = opt.multi_view_patch_size
                sample_num = opt.multi_view_sample_num
                progress = min(iteration / opt.multi_view_pixel_noise_th_end_iter, 1.0)
                pixel_noise_th = opt.multi_view_pixel_noise_th_init + progress * (
                    opt.multi_view_pixel_noise_th_final - opt.multi_view_pixel_noise_th_init
                )
                total_patch_size = (patch_size * 2 + 1) ** 2
                ncc_weight = opt.multi_view_ncc_weight
                geo_weight = opt.multi_view_geo_weight
                ## compute geometry consistency mask and loss
                H, W = render_pkg['plane_depth'].squeeze().shape
                ix, iy = torch.meshgrid(
                    torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([ix, iy], dim=-1).float().to(render_pkg['plane_depth'].device)

                nearest_render_pkg = render(nearest_cam, gaussians, pipe, bg, app_model=app_model,
                                            return_plane=True, return_depth_normal=False)

                nearest_infl = nearest_render_pkg['infl']
                nearest_infl_mask = nearest_infl > 0.01
                # print("Selected {} Gaussians from all {} Gaussians".format(nearest_infl_mask.sum(), nearest_infl.shape[0]))
                
                accT_masked_pkg = render(
                    viewpoint_cam,
                    gaussians,
                    pipe,
                    bg,
                    app_model=app_model,
                    gaussian_mask=nearest_infl_mask,
                    return_plane=True,
                    return_depth_normal=False
                )
                accT_masked = accT_masked_pkg.get('accT_masked', None)
                
                accT_masked = 1.0 - accT_masked



                pts = gaussians.get_points_from_depth(viewpoint_cam, render_pkg['plane_depth'])
                pts_in_nearest_cam = pts @ nearest_cam.world_view_transform[:3,:3] + nearest_cam.world_view_transform[3,:3]
                map_z, d_mask = gaussians.get_points_depth_in_depth_map(nearest_cam, nearest_render_pkg['plane_depth'], pts_in_nearest_cam)
                
                pts_in_nearest_cam = pts_in_nearest_cam / (pts_in_nearest_cam[:,2:3])
                pts_in_nearest_cam = pts_in_nearest_cam * map_z.squeeze()[...,None]
                R = torch.tensor(nearest_cam.R).float().cuda()
                T = torch.tensor(nearest_cam.T).float().cuda()
                pts_ = (pts_in_nearest_cam-T)@R.transpose(-1,-2)
                pts_in_view_cam = pts_ @ viewpoint_cam.world_view_transform[:3,:3] + viewpoint_cam.world_view_transform[3,:3]
                pts_projections = torch.stack(
                            [pts_in_view_cam[:,0] * viewpoint_cam.Fx / pts_in_view_cam[:,2] + viewpoint_cam.Cx,
                            pts_in_view_cam[:,1] * viewpoint_cam.Fy / pts_in_view_cam[:,2] + viewpoint_cam.Cy], -1).float()
                pixel_noise = torch.norm(pts_projections - pixels.reshape(*pts_projections.shape), dim=-1)
                
                                
                if accT_masked is not None:
                    accT_masked_filtered = accT_masked.clone()
                    accT_masked_filtered[~d_mask.reshape(accT_masked.shape)] = 0
                    accT_masked = accT_masked_filtered
                
                if not opt.wo_use_geo_occ_aware:

                    d_mask_for_accT = d_mask & (pixel_noise < 4 * pixel_noise_th) # 10 is a hyperparameter
                    accT_weights = accT_masked.reshape(-1).clone()
                    accT_weights[~d_mask_for_accT] = 0

                    d_mask = d_mask & (pixel_noise < pixel_noise_th)
                    
                    geo_consistency_mask = d_mask.reshape(H, W).clone()
                    
                    weights = (1.0 / torch.exp(pixel_noise)).detach()
                    weights[~d_mask] = 0
                    
                    # weights = weights + 0.5 * accT_weights
                    # weights_centered = (weights - 0.75) * 4  
                    # weights = torch.sigmoid(weights_centered)  
                    # beta = 4.0
                else:
                    d_mask = d_mask
                    weights = torch.ones_like(pixel_noise)
                    weights[~d_mask] = 0
                    geo_consistency_mask = d_mask.reshape(H, W).clone()
                weights_copy = weights.clone()
                weights_original = weights.clone()
                pixels_original = pixels.clone()
                d_mask_original = d_mask.clone()


                if d_mask.sum() > 0:
                    geo_loss = geo_weight * ((weights * pixel_noise)[d_mask]).mean()
                    loss += geo_loss
                    if use_virtul_cam is False:
                        with torch.no_grad():
                            ## sample mask
                            d_mask = d_mask.reshape(-1)
                            valid_indices = torch.arange(d_mask.shape[0], device=d_mask.device)[d_mask]
                            if d_mask.sum() > sample_num:
                                index = np.random.choice(d_mask.sum().cpu().numpy(), sample_num, replace = False)
                                valid_indices = valid_indices[index]

                            weights = weights.reshape(-1)[valid_indices]
                            ## sample ref frame patch
                            pixels = pixels.reshape(-1,2)[valid_indices]
                            offsets = patch_offsets(patch_size, pixels.device)
                            ori_pixels_patch = pixels.reshape(-1, 1, 2) / viewpoint_cam.ncc_scale + offsets.float()
                            
                            H_homo, W_homo = gt_image_gray.squeeze().shape
                            pixels_patch = ori_pixels_patch.clone()
                            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W_homo - 1) - 1.0
                            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H_homo - 1) - 1.0
                            ref_gray_val = F.grid_sample(gt_image_gray.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
                            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

                            ref_to_neareast_r = nearest_cam.world_view_transform[:3,:3].transpose(-1,-2) @ viewpoint_cam.world_view_transform[:3,:3]
                            ref_to_neareast_t = -ref_to_neareast_r @ viewpoint_cam.world_view_transform[3,:3] + nearest_cam.world_view_transform[3,:3]

                        ## compute Homography
                        ref_local_n = render_pkg["rendered_normal"].permute(1,2,0)
                        ref_local_n = ref_local_n.reshape(-1,3)[valid_indices]

                        ref_local_d = render_pkg['rendered_distance'].squeeze()
                        # rays_d = viewpoint_cam.get_rays()
                        # rendered_normal2 = render_pkg["rendered_normal"].permute(1,2,0).reshape(-1,3)
                        # ref_local_d = render_pkg['plane_depth'].view(-1) * ((rendered_normal2 * rays_d.reshape(-1,3)).sum(-1).abs())
                        # ref_local_d = ref_local_d.reshape(*render_pkg['plane_depth'].shape)

                        ref_local_d = ref_local_d.reshape(-1)[valid_indices]
                        H_ref_to_neareast = ref_to_neareast_r[None] - \
                            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
                        H_ref_to_neareast = torch.matmul(nearest_cam.get_k(nearest_cam.ncc_scale)[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
                        H_ref_to_neareast = H_ref_to_neareast @ viewpoint_cam.get_inv_k(viewpoint_cam.ncc_scale)
                        
                        ## compute neareast frame patch
                        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
                        grid[:, :, 0] = 2 * grid[:, :, 0] / (W_homo - 1) - 1.0
                        grid[:, :, 1] = 2 * grid[:, :, 1] / (H_homo - 1) - 1.0
                        _, nearest_image_gray = nearest_cam.get_image()
                        sampled_gray_val = F.grid_sample(nearest_image_gray[None], grid.reshape(1, -1, 1, 2), align_corners=True)
                        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)
                        
                        ## compute loss
                        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
                        mask = ncc_mask.reshape(-1)
                        ncc = ncc.reshape(-1) * weights
                        ncc = ncc[mask].squeeze()

                        if mask.sum() > 0:
                            ncc_loss = ncc_weight * ncc.mean()
                            loss += ncc_loss
                    
                    if opt.save_debug_images:
                        render_with_nearest_mask_pkg = render_with_mask(viewpoint_cam, gaussians, pipe, bg, app_model=app_model, mask=nearest_infl_mask, return_plane=True, return_depth_normal=False)
                        render_with_color_mask_pkg = render_with_color_mask(viewpoint_cam, gaussians, pipe, bg, app_model=app_model, mask=nearest_infl_mask, return_plane=True, return_depth_normal=False)


                if opt.save_debug_images and iteration % 1000 == 0 and 'weights_original' in locals():
                    gt_img_show = ((gt_image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    if 'app_image' in render_pkg:
                        img_show = ((render_pkg['app_image']).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    else:
                        img_show = ((image).permute(1,2,0).clamp(0,1)[:,:,[2,1,0]]*255).detach().cpu().numpy().astype(np.uint8)
                    normal_show = (((normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    depth_normal_show = (((depth_normal+1.0)*0.5).permute(1,2,0).clamp(0,1)*255).detach().cpu().numpy().astype(np.uint8)
                    d_mask_show = (weights_original.float()*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    d_mask_show_color = cv2.applyColorMap(d_mask_show, cv2.COLORMAP_JET)
                    depth = render_pkg['plane_depth'].squeeze().detach().cpu().numpy()
                    depth_i = (depth - depth.min()) / (depth.max() - depth.min() + 1e-20)
                    depth_i = (depth_i * 255).clip(0, 255).astype(np.uint8)
                    depth_color = cv2.applyColorMap(depth_i, cv2.COLORMAP_JET)
                    distance = render_pkg['rendered_distance'].squeeze().detach().cpu().numpy()
                    distance_i = (distance - distance.min()) / (distance.max() - distance.min() + 1e-20)
                    distance_i = (distance_i * 255).clip(0, 255).astype(np.uint8)
                    distance_color = cv2.applyColorMap(distance_i, cv2.COLORMAP_JET)
                    image_weight = image_weight.detach().cpu().numpy()
                    image_weight = (image_weight * 255).clip(0, 255).astype(np.uint8)
                    image_weight_color = cv2.applyColorMap(image_weight, cv2.COLORMAP_JET)
                    
                    opacity = render_pkg['opacity'].detach().cpu().numpy()
                    opacity_i = (opacity - opacity.min()) / (opacity.max() - opacity.min() + 1e-20)
                    opacity_i = (opacity_i * 255).clip(0, 255).astype(np.uint8)
                    opacity_color = cv2.applyColorMap(opacity_i, cv2.COLORMAP_HOT)

                    if 'accT_masked_pkg' in locals() and 'accT_masked' in accT_masked_pkg:
                        print("accT_masked_pkg exists")
                        accT_masked = accT_masked_pkg['accT_masked'].detach().cpu().numpy()
                        accT_masked_i = (accT_masked - accT_masked.min()) / (accT_masked.max() - accT_masked.min() + 1e-20)
                        accT_masked_i = (accT_masked_i * 255).clip(0, 255).astype(np.uint8)
                        accT_masked_color = cv2.applyColorMap(accT_masked_i, cv2.COLORMAP_HOT)
                    else:
                        accT_masked_color = np.zeros_like(opacity_color)
                    
                    accT_shared_vis = np.zeros_like(opacity_color)
                    geo_consistent_vis = np.zeros_like(opacity_color)
                    diff_vis = np.zeros_like(opacity_color)
                    
                    row0 = np.concatenate([gt_img_show, img_show, normal_show, distance_color], axis=1)
                    row1 = np.concatenate([d_mask_show_color, depth_color, depth_normal_show, image_weight_color], axis=1)
                    row2 = np.concatenate([opacity_color, accT_masked_color, accT_shared_vis, geo_consistent_vis], axis=1)
                    row3 = np.concatenate([diff_vis, np.zeros_like(opacity_color), np.zeros_like(opacity_color), np.zeros_like(opacity_color)], axis=1)
                    image_to_show = np.concatenate([row0, row1, row2, row3], axis=0)
                    
                    def tensor_to_cv2img(tensor):
                        # tensor: [C,H,W], 0~1
                        img = tensor.detach().cpu().clamp(0,1)
                        if img.shape[0] == 1:
                            img = img.repeat(3,1,1)
                        img = (img * 255).permute(1,2,0).numpy().astype(np.uint8)
                        img = img[...,[2,1,0]]  # RGB to BGR for cv2
                        return img

                    render_raw_img = tensor_to_cv2img(render_pkg["render"])
                    nearest_render_img = tensor_to_cv2img(nearest_render_pkg["render"])
                    
                    if opt.save_debug_images:
                        render_with_nearest_mask_img = tensor_to_cv2img(render_with_nearest_mask_pkg["render"])
                        render_with_color_mask_img = tensor_to_cv2img(render_with_color_mask_pkg["render"])
                        color_mask_img_vis = np.zeros((H, W, 3), dtype=np.uint8)
                        d_mask_img_vis = np.zeros((H, W, 3), dtype=np.uint8)
                    
                    if opt.save_debug_images:
                        all_renders = np.concatenate([
                            render_raw_img, 
                            render_with_nearest_mask_img, 
                            nearest_render_img, 
                            render_with_color_mask_img
                        ], axis=1)
                    else:
                        all_renders = np.concatenate([
                            render_raw_img, 
                            nearest_render_img
                        ], axis=1)
                    
                    if opt.save_debug_images:
                        single_img_width = all_renders.shape[1] // 4
                    else:
                        single_img_width = all_renders.shape[1] // 2
                    
                    mask_height = all_renders.shape[0]
                    
                    if color_mask_img_vis.shape[:2] != (mask_height, single_img_width):
                        color_mask_img_vis = cv2.resize(color_mask_img_vis, (single_img_width, mask_height))
                    if d_mask_img_vis.shape[:2] != (mask_height, single_img_width):
                        d_mask_img_vis = cv2.resize(d_mask_img_vis, (single_img_width, mask_height))
                    
                    if opt.save_debug_images:
                        blank_img1 = np.zeros((mask_height, single_img_width, 3), dtype=np.uint8)
                        blank_img2 = np.zeros((mask_height, single_img_width, 3), dtype=np.uint8)
                        depth_vis_img1 = np.zeros((mask_height, single_img_width, 3), dtype=np.uint8)
                        depth_vis_img2 = np.zeros((mask_height, single_img_width, 3), dtype=np.uint8)
                    else:
                        blank_img1 = np.zeros((mask_height, single_img_width, 3), dtype=np.uint8)
                        depth_vis_img1 = np.zeros((mask_height, single_img_width, 3), dtype=np.uint8)
                        depth_vis_img2 = np.zeros((mask_height, single_img_width, 3), dtype=np.uint8)
                    
                    if invDepth is not None and mono_invdepth is not None:
                        invDepth_vis = invDepth.squeeze().detach().cpu().numpy()
                        if invDepth_vis.size > 0:
                            invDepth_norm = (invDepth_vis - invDepth_vis.min()) / (invDepth_vis.max() - invDepth_vis.min() + 1e-8)
                            invDepth_norm = (invDepth_norm * 255).astype(np.uint8)
                            invDepth_color = cv2.applyColorMap(invDepth_norm, cv2.COLORMAP_JET)
                            if invDepth_color.shape[:2] != (mask_height, single_img_width):
                                invDepth_color = cv2.resize(invDepth_color, (single_img_width, mask_height))
                            depth_vis_img1 = invDepth_color
                        
                        if corrected_mono_invdepth is not None:
                            corrected_mono_vis = corrected_mono_invdepth.squeeze().detach().cpu().numpy()
                            if corrected_mono_vis.size > 0:
                                corrected_mono_norm = (corrected_mono_vis - corrected_mono_vis.min()) / (corrected_mono_vis.max() - corrected_mono_vis.min() + 1e-8)
                                corrected_mono_norm = (corrected_mono_norm * 255).astype(np.uint8)
                                corrected_mono_color = cv2.applyColorMap(corrected_mono_norm, cv2.COLORMAP_JET)
                                if corrected_mono_color.shape[:2] != (mask_height, single_img_width):
                                    corrected_mono_color = cv2.resize(corrected_mono_color, (single_img_width, mask_height))
                                depth_vis_img2 = corrected_mono_color
                        else:
                            mono_invdepth_vis = mono_invdepth.squeeze().detach().cpu().numpy()
                            if mono_invdepth_vis.size > 0:
                                mono_invdepth_norm = (mono_invdepth_vis - mono_invdepth_vis.min()) / (mono_invdepth_vis.max() - mono_invdepth_vis.min() + 1e-8)
                                mono_invdepth_norm = (mono_invdepth_norm * 255).astype(np.uint8)
                                mono_invdepth_color = cv2.applyColorMap(mono_invdepth_norm, cv2.COLORMAP_JET)
                                if mono_invdepth_color.shape[:2] != (mask_height, single_img_width):
                                    mono_invdepth_color = cv2.resize(mono_invdepth_color, (single_img_width, mask_height))
                                depth_vis_img2 = mono_invdepth_color
                    
                    opacity_mask_img = opacity_color.copy()
                    if opacity_mask_img.shape[:2] != (mask_height, single_img_width):
                        opacity_mask_img = cv2.resize(opacity_mask_img, (single_img_width, mask_height))
                    
                    mask_renders = np.concatenate([
                        color_mask_img_vis,
                        d_mask_img_vis,
                        depth_vis_img1,  # invDepth
                        depth_vis_img2   # mono_invdepth
                    ], axis=1)
                    final_image = np.concatenate([image_to_show, all_renders, mask_renders], axis=0)
                    cv2.imwrite(os.path.join(debug_path, "%05d"%iteration + "_" + viewpoint_cam.image_name + ".jpg"), final_image)

        if iteration > opt.dpt_loss_from_iter and opt.use_dpt_loss:
            depth = render_pkg["plane_depth"]
            depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.dpt_end_iter)
            if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
                depth_clamped = torch.clamp(depth, min=1e-6)
                invDepth = 1 / depth_clamped
                
                mono_invdepth = viewpoint_cam.invdepthmap.cuda()
                depth_squeezed = depth.squeeze()  # (H, W)
                mono_invdepth_squeezed = mono_invdepth.squeeze()  # (H, W)
                invDepth_squeezed = invDepth.squeeze()  # (H, W)
                
                mask = (mono_invdepth_squeezed != 0) & (depth_squeezed > 1e-10)  # (H, W)
                
                if opt.use_depth_percentile_filter and mask.sum() > 0:
                    valid_invdepths = invDepth_squeezed[mask]
                    invdepth_threshold = torch.quantile(valid_invdepths, opt.depth_percentile_threshold)
                    
                    if opt.mono_dpt_trunc_near > 0.001 and opt.mono_dpt_trunc_far < float('inf'):
                        invdepth_near = 1.0 / opt.mono_dpt_trunc_far
                        invdepth_far = 1.0 / opt.mono_dpt_trunc_near
                        
                        invdepth_threshold = torch.clamp(invdepth_threshold, min=invdepth_near, max=invdepth_far)
                        
                        invdepth_mask = invDepth_squeezed > invdepth_threshold
                        mask = mask & invdepth_mask
                        
                    else:
                        invdepth_mask = invDepth_squeezed > invdepth_threshold
                        mask = mask & invdepth_mask
                    
                if mask.sum() > 0:
                    if opt.use_quadtree_depth_alignment:
                            if iteration < opt.quadtree_level0_end_iter:
                                split_level = 0
                            elif iteration < opt.quadtree_level1_end_iter:
                                split_level = 1
                            elif iteration < opt.quadtree_level2_end_iter:
                                split_level = 2
                            else:
                                split_level = opt.quadtree_max_level
                            
                            if split_level != current_cached_split_level:
                                if iteration > 0:
                                    num_blocks = (2 ** split_level) ** 2
                                    cache_size = len(quadtree_cache)
                                    total_accesses = quadtree_cache_hits + quadtree_cache_misses
                                    hit_rate = (quadtree_cache_hits / total_accesses * 100) if total_accesses > 0 else 0
                                    
                                    print(f"\n[ {iteration}] :  {current_cached_split_level} -> {split_level}")
                                    print(f"  : {num_blocks} ({2**split_level}×{2**split_level})")
                                    print(f"  Cache: {cache_size}，")
                                    print(f"  Cache: {quadtree_cache_hits}, {quadtree_cache_misses}, {hit_rate:.1f}%")
                                    
                                    if split_level == 0:
                                        print(f"  : Level 0  ({opt.dpt_loss_from_iter}-{opt.quadtree_level0_end_iter})")
                                    elif split_level == 1:
                                        print(f"  : Level 1  ({opt.quadtree_level0_end_iter}-{opt.quadtree_level1_end_iter})")
                                    elif split_level == 2:
                                        print(f"  : Level 2  ({opt.quadtree_level1_end_iter}-{opt.quadtree_level2_end_iter})")
                                    else:
                                        print(f"  : Level {split_level} ()")
                                    
                                    print(f"  : max_level={opt.quadtree_max_level}, min_valid_pixels={opt.quadtree_min_valid_pixels}")
                                    
                                    if geo_consistency_mask is not None:
                                        geo_mask_ratio = geo_consistency_mask.float().mean().item()
                                        print(f"  mask: {geo_mask_ratio*100:.2f}%")
                                    else:
                                        print(f"  mask（multi-view）")
                                
                                quadtree_cache.clear()
                                current_cached_split_level = split_level
                                quadtree_cache_hits = 0
                                quadtree_cache_misses = 0
                                
                                quadtree_save_dir = os.path.join(scene.model_path, f"quadtree_depth/level_{split_level}_iter_{iteration}")
                                os.makedirs(quadtree_save_dir, exist_ok=True)
                                
                                if iteration > 0:
                                    print(f"  : {quadtree_save_dir}")
                                    print(f"  (.npy)(.png)")
                            
                            cache_key = viewpoint_cam.image_name
                            
                            if cache_key in quadtree_cache:
                                cached_value = quadtree_cache[cache_key]
                                if cache_device.type == 'cpu':
                                    corrected_mono_invdepth = cached_value.cuda()
                                else:
                                    corrected_mono_invdepth = cached_value
                                quadtree_cache_hits += 1
                            else:
                                corrected_mono_invdepth = quadtree_depth_alignment(
                                    mono_invdepth_squeezed,
                                    invDepth_squeezed,
                                    mask,
                                    split_level,
                                    min_valid_pixels=opt.quadtree_min_valid_pixels,
                                    min_depth_range=opt.quadtree_min_depth_range,
                                    geo_mask=geo_consistency_mask,
                                    filter_outliers=opt.quadtree_filter_outliers,
                                    outlier_threshold_method=opt.quadtree_outlier_threshold
                                )
                                

                                with torch.no_grad():
                                    if cache_device.type == 'cpu':
                                        quadtree_cache[cache_key] = corrected_mono_invdepth.detach().cpu().clone()
                                    else:
                                        quadtree_cache[cache_key] = corrected_mono_invdepth.detach().clone()
                                quadtree_cache_misses += 1
                                
                                if 'quadtree_save_dir' in locals():
                                    save_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
                                    
                                    np.save(
                                        os.path.join(quadtree_save_dir, f"{save_name}_corrected.npy"),
                                        corrected_mono_invdepth.detach().cpu().numpy()
                                    )
                                    
                                    with torch.no_grad():
                                        def norm_depth_for_save(depth_tensor, mask_tensor):
                                            """Normalize depth map for saving, process on CPU"""
                                            depth = depth_tensor.detach().cpu().float()
                                            if mask_tensor.sum() > 0:
                                                mask_cpu = mask_tensor.cpu()
                                                valid_depth = depth[mask_cpu]
                                                if valid_depth.numel() > 0:
                                                    vmin, vmax = valid_depth.min().item(), valid_depth.max().item()
                                                    if vmax > vmin:
                                                        depth = (depth - vmin) / (vmax - vmin + 1e-7)
                                            depth = torch.clamp(depth * 255, 0, 255).numpy().astype(np.uint8)
                                            return depth
                                        
                                        mono_vis = norm_depth_for_save(mono_invdepth_squeezed, mask)
                                        corrected_vis = norm_depth_for_save(corrected_mono_invdepth, mask)
                                        render_vis = norm_depth_for_save(invDepth_squeezed, mask)
                                    
                                    mono_color = cv2.applyColorMap(mono_vis, cv2.COLORMAP_JET)
                                    corrected_color = cv2.applyColorMap(corrected_vis, cv2.COLORMAP_JET)
                                    render_color = cv2.applyColorMap(render_vis, cv2.COLORMAP_JET)
                                    
                                    corrected_color = draw_quadtree_grid(corrected_color, split_level, color=(255, 255, 255), thickness=1)
                                    
                                    compare_img = cv2.hconcat([mono_color, corrected_color, render_color])
                                    
                                    font = cv2.FONT_HERSHEY_SIMPLEX
                                    h, w = mono_vis.shape
                                    cv2.putText(compare_img, 'Original Mono', (10, 30), font, 1, (255, 255, 255), 2)
                                    cv2.putText(compare_img, f'Corrected (Lv{split_level})', (w+10, 30), font, 1, (255, 255, 255), 2)
                                    cv2.putText(compare_img, 'Render', (2*w+10, 30), font, 1, (255, 255, 255), 2)
                                    
                                    cv2.imwrite(
                                        os.path.join(quadtree_save_dir, f"{save_name}_compare.png"),
                                        compare_img
                                    )
                                    
                                    del mono_vis, corrected_vis, render_vis, mono_color, corrected_color, render_color, compare_img
                                
                                torch.cuda.empty_cache()
                                
                                if len(quadtree_cache) % 10 == 0:
                                    total = quadtree_cache_hits + quadtree_cache_misses
                                    rate = (quadtree_cache_hits / total * 100) if total > 0 else 0
                                    level_names = {0: "Full Image", 1: "Quarter", 2: "Sixteenth"}
                                    level_name = level_names.get(split_level, f"Level{split_level}")
                                    print(f"  [Cache] Cache {len(quadtree_cache)}  (: {rate:.1f}%, {level_name}, : quadtree_depth/level_{split_level}_iter_{iteration}/)")
                    else:
                        corrected_mono_invdepth = mono_invdepth_squeezed
                    
                    smoothness_loss = torch.tensor(0.0, device='cuda')
                    
                    if iteration % 1000 == 0:
                        save_dir = os.path.join(scene.model_path, "dpt_vis")
                        os.makedirs(save_dir, exist_ok=True)

                        def norm_for_vis(t, min_=None, max_=None):
                            t = t.clone()
                            t = t.float()
                            if t.numel() == 0:
                                return t
                            t[~mask] = float('nan')
                            valid = t[mask]
                            if min_ is None or max_ is None:
                                if valid.numel() == 0:
                                    min__, max__ = 0, 1
                                else:
                                    valid_no_nan = valid[~torch.isnan(valid)]
                                    if valid_no_nan.numel() == 0:
                                        min__, max__ = 0, 1
                                    else:
                                        min__ = valid_no_nan.min()
                                        max__ = valid_no_nan.max()
                                    if max__ == min__:
                                        max__ = min__ + 1
                            else:
                                min__ = min_
                                max__ = max_
                            t = (t - min__) / (max__ - min__ + 1e-7)
                            t = torch.nan_to_num(t, nan=0.0)
                            return t
                        
                        inv_min = invDepth_squeezed.min()
                        inv_max = invDepth_squeezed.max()
                        invDepth_vis = norm_for_vis(invDepth_squeezed, inv_min, inv_max).cpu()
                        mono_invdepth_vis = norm_for_vis(mono_invdepth_squeezed, inv_min, inv_max).cpu()
                        corrected_vis = norm_for_vis(corrected_mono_invdepth, inv_min, inv_max).cpu()
                        
                        diff = torch.abs(corrected_mono_invdepth - mono_invdepth_squeezed)
                        if mask.sum() > 0:
                            max_val = diff[mask].max()
                            if max_val == 0:
                                max_val = 1e-6
                            diff_vis = (diff / max_val * 255.0).cpu()
                        else:
                            diff_vis = (diff * 0).cpu()
                        
                        diff_render_mono = torch.abs(invDepth_vis - mono_invdepth_vis)
                        diff_render_corrected = torch.abs(invDepth_vis - corrected_vis)
                        
                        arrs_row1 = []
                        for idx, img in enumerate([invDepth_vis, mono_invdepth_vis, corrected_vis, diff_vis]):
                            arr = (img.detach().cpu().numpy() * 255.0).astype(np.uint8)
                            color = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
                            if opt.use_quadtree_depth_alignment and idx in [2, 3]:
                                if iteration < opt.quadtree_level0_end_iter:
                                    vis_split_level = 0
                                elif iteration < opt.quadtree_level1_end_iter:
                                    vis_split_level = 1
                                elif iteration < opt.quadtree_level2_end_iter:
                                    vis_split_level = 2
                                else:
                                    vis_split_level = opt.quadtree_max_level
                                color = draw_quadtree_grid(color, vis_split_level, color=(255, 255, 255), thickness=1)
                            arrs_row1.append(color)
                        
                        max_diff = max(diff_render_mono.max().item(), diff_render_corrected.max().item())
                        if max_diff < 1e-6:
                            max_diff = 1.0
                        arrs_row2 = []
                        for img in [diff_render_mono, diff_render_corrected]:
                            arr = (img.detach().cpu().numpy() / max_diff * 255.0).clip(0, 255).astype(np.uint8)
                            color = cv2.applyColorMap(arr, cv2.COLORMAP_JET)
                            arrs_row2.append(color)
                        blank = np.zeros_like(arrs_row2[0])
                        arrs_row2.append(blank)
                        arrs_row2.append(blank)
                        
                        mask_vis = mask.detach().cpu().float().numpy() * 255.0
                        mask_vis = mask_vis.astype(np.uint8)
                        mask_color = cv2.applyColorMap(mask_vis, cv2.COLORMAP_HOT)
                        arrs_row3 = [mask_color]
                        blank = np.zeros_like(arrs_row3[0])
                        arrs_row3.extend([blank, blank, blank])
                        
                        concat_row1 = cv2.hconcat(arrs_row1)
                        concat_row2 = cv2.hconcat(arrs_row2)
                        concat_row3 = cv2.hconcat(arrs_row3)
                        concat_image = cv2.vconcat([concat_row1, concat_row2, concat_row3])
                        
                        fn = os.path.join(save_dir, f"iter{iteration}_{os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]}_compare.png")
                        cv2.imwrite(fn, concat_image)
                    
                    inv_det = invDepth_squeezed.detach()
                    corr_det = corrected_mono_invdepth.detach()
                    
                    loss_gauss = torch.abs(invDepth_squeezed[mask] - corr_det[mask]).mean()
                    Ll1depth = depth_l1_weight(iteration) * opt.depth_l1_loss_scale * (loss_gauss)
                    
                    if opt.use_depth_direction_consistency_loss:
                        H, W = invDepth_squeezed.shape
                        
                        mask_expanded = mask.clone()
                        
                        diff_gauss_up = torch.zeros_like(invDepth_squeezed)
                        diff_mono_up = torch.zeros_like(corrected_mono_invdepth)
                        mask_up = torch.zeros_like(mask, dtype=torch.bool)
                        if H > 1:
                            diff_gauss_up[1:, :] = invDepth_squeezed[1:, :] - invDepth_squeezed[:-1, :]
                            diff_mono_up[1:, :] = corrected_mono_invdepth[1:, :] - corrected_mono_invdepth[:-1, :]
                            mask_up[1:, :] = mask[1:, :] & mask[:-1, :]
                        
                        diff_gauss_down = torch.zeros_like(invDepth_squeezed)
                        diff_mono_down = torch.zeros_like(corrected_mono_invdepth)
                        mask_down = torch.zeros_like(mask, dtype=torch.bool)
                        if H > 1:
                            diff_gauss_down[:-1, :] = invDepth_squeezed[:-1, :] - invDepth_squeezed[1:, :]
                            diff_mono_down[:-1, :] = corrected_mono_invdepth[:-1, :] - corrected_mono_invdepth[1:, :]
                            mask_down[:-1, :] = mask[:-1, :] & mask[1:, :]
                        
                        diff_gauss_left = torch.zeros_like(invDepth_squeezed)
                        diff_mono_left = torch.zeros_like(corrected_mono_invdepth)
                        mask_left = torch.zeros_like(mask, dtype=torch.bool)
                        if W > 1:
                            diff_gauss_left[:, 1:] = invDepth_squeezed[:, 1:] - invDepth_squeezed[:, :-1]
                            diff_mono_left[:, 1:] = corrected_mono_invdepth[:, 1:] - corrected_mono_invdepth[:, :-1]
                            mask_left[:, 1:] = mask[:, 1:] & mask[:, :-1]
                        
                        diff_gauss_right = torch.zeros_like(invDepth_squeezed)
                        diff_mono_right = torch.zeros_like(corrected_mono_invdepth)
                        mask_right = torch.zeros_like(mask, dtype=torch.bool)
                        if W > 1:
                            diff_gauss_right[:, :-1] = invDepth_squeezed[:, :-1] - invDepth_squeezed[:, 1:]
                            diff_mono_right[:, :-1] = corrected_mono_invdepth[:, :-1] - corrected_mono_invdepth[:, 1:]
                            mask_right[:, :-1] = mask[:, :-1] & mask[:, 1:]
                        
                        def compute_direction_consistency_loss(diff_gauss, diff_mono, direction_mask):
                            """Compute direction consistency loss"""
                            if direction_mask.sum() == 0:
                                return torch.tensor(0.0, device=diff_gauss.device)
                            
                            valid_diff_gauss = diff_gauss[direction_mask]
                            valid_diff_mono = diff_mono[direction_mask]
                            
                            eps = 1e-8
                            sign_gauss = valid_diff_gauss / (torch.abs(valid_diff_gauss) + eps)
                            sign_mono = valid_diff_mono / (torch.abs(valid_diff_mono) + eps)
                            
                            sign_product = sign_gauss * sign_mono
                            inconsistent_mask = sign_product < 0
                            
                            if inconsistent_mask.sum() > 0:
                                inconsistent_loss = torch.abs(valid_diff_gauss[inconsistent_mask] - valid_diff_mono[inconsistent_mask]).mean()
                                return inconsistent_loss
                            else:
                                return torch.tensor(0.0, device=diff_gauss.device)
                        
                        loss_direction_up = compute_direction_consistency_loss(diff_gauss_up, diff_mono_up, mask_up)
                        loss_direction_down = compute_direction_consistency_loss(diff_gauss_down, diff_mono_down, mask_down)
                        loss_direction_left = compute_direction_consistency_loss(diff_gauss_left, diff_mono_left, mask_left)
                        loss_direction_right = compute_direction_consistency_loss(diff_gauss_right, diff_mono_right, mask_right)
                        
                        loss_direction_consistency = (loss_direction_up + loss_direction_down + loss_direction_left + loss_direction_right) / 4.0
                        
                        if iteration % 1000 == 0:
                            direction_vis_dir = os.path.join(scene.model_path, "direction_consistency_vis")
                            os.makedirs(direction_vis_dir, exist_ok=True)
                            
                            def norm_diff_for_vis(diff_tensor, direction_mask):
                                """Normalize difference map for visualization"""
                                diff_vis = diff_tensor.clone().detach().cpu().float()
                                diff_vis[~direction_mask.cpu()] = float('nan')
                                valid_diff = diff_vis[direction_mask.cpu()]
                                if valid_diff.numel() > 0:
                                    valid_no_nan = valid_diff[~torch.isnan(valid_diff)]
                                    if valid_no_nan.numel() > 0:
                                        vmin, vmax = valid_no_nan.min().item(), valid_no_nan.max().item()
                                        if vmax > vmin:
                                            diff_vis = (diff_vis - vmin) / (vmax - vmin + 1e-7)
                                        else:
                                            diff_vis = diff_vis * 0.0
                                diff_vis = torch.nan_to_num(diff_vis, nan=0.0)
                                diff_vis = (diff_vis * 127.5 + 127.5).clamp(0, 255)
                                return diff_vis.numpy().astype(np.uint8)
                            
                            def compute_sign_consistency_map(diff_gauss, diff_mono, direction_mask):
                                """Compute sign consistency map: consistent=255 (white), inconsistent=0 (black)"""
                                eps = 1e-8
                                sign_gauss = diff_gauss / (torch.abs(diff_gauss) + eps)
                                sign_mono = diff_mono / (torch.abs(diff_mono) + eps)
                                sign_product = sign_gauss * sign_mono
                                
                                consistency_map = torch.zeros_like(direction_mask, dtype=torch.float32)
                                consistent_mask = (sign_product >= 0) & direction_mask
                                consistency_map[consistent_mask] = 255.0
                                inconsistent_mask = (sign_product < 0) & direction_mask
                                consistency_map[inconsistent_mask] = 128.0
                                
                                return consistency_map.detach().cpu().numpy().astype(np.uint8)
                            
                            diff_gauss_up_vis = norm_diff_for_vis(diff_gauss_up, mask_up)
                            diff_mono_up_vis = norm_diff_for_vis(diff_mono_up, mask_up)
                            diff_gauss_down_vis = norm_diff_for_vis(diff_gauss_down, mask_down)
                            diff_mono_down_vis = norm_diff_for_vis(diff_mono_down, mask_down)
                            diff_gauss_left_vis = norm_diff_for_vis(diff_gauss_left, mask_left)
                            diff_mono_left_vis = norm_diff_for_vis(diff_mono_left, mask_left)
                            diff_gauss_right_vis = norm_diff_for_vis(diff_gauss_right, mask_right)
                            diff_mono_right_vis = norm_diff_for_vis(diff_mono_right, mask_right)
                            
                            consistency_up = compute_sign_consistency_map(diff_gauss_up, diff_mono_up, mask_up)
                            consistency_down = compute_sign_consistency_map(diff_gauss_down, diff_mono_down, mask_down)
                            consistency_left = compute_sign_consistency_map(diff_gauss_left, diff_mono_left, mask_left)
                            consistency_right = compute_sign_consistency_map(diff_gauss_right, diff_mono_right, mask_right)
                            
                            def compute_diff_diff(diff_gauss, diff_mono, direction_mask):
                                """Compute difference between two differences"""
                                diff_diff = torch.abs(diff_gauss - diff_mono)
                                diff_diff_vis = diff_diff.clone().detach().cpu().float()
                                diff_diff_vis[~direction_mask.cpu()] = float('nan')
                                valid_diff = diff_diff_vis[direction_mask.cpu()]
                                if valid_diff.numel() > 0:
                                    valid_no_nan = valid_diff[~torch.isnan(valid_diff)]
                                    if valid_no_nan.numel() > 0:
                                        vmax = valid_no_nan.max().item()
                                        if vmax > 1e-6:
                                            diff_diff_vis = (diff_diff_vis / vmax * 255.0).clamp(0, 255)
                                        else:
                                            diff_diff_vis = diff_diff_vis * 0.0
                                diff_diff_vis = torch.nan_to_num(diff_diff_vis, nan=0.0)
                                return diff_diff_vis.numpy().astype(np.uint8)
                            
                            diff_diff_up = compute_diff_diff(diff_gauss_up, diff_mono_up, mask_up)
                            diff_diff_down = compute_diff_diff(diff_gauss_down, diff_mono_down, mask_down)
                            diff_diff_left = compute_diff_diff(diff_gauss_left, diff_mono_left, mask_left)
                            diff_diff_right = compute_diff_diff(diff_gauss_right, diff_mono_right, mask_right)
                            
                            def apply_colormap_to_diff(arr):
                                """Apply colormap to difference map"""
                                return cv2.applyColorMap(arr, cv2.COLORMAP_JET)
                            
                            def consistency_to_rgb(consistency_map):
                                """Convert consistency map to RGB: white=consistent, red=inconsistent, black=invalid"""
                                h, w = consistency_map.shape
                                rgb = np.zeros((h, w, 3), dtype=np.uint8)
                                consistent_mask = consistency_map == 255
                                rgb[consistent_mask] = [255, 255, 255]
                                inconsistent_mask = consistency_map == 128
                                rgb[inconsistent_mask] = [0, 0, 255]
                                return rgb
                            
                            row1_up = []
                            row1_up.append(apply_colormap_to_diff(diff_gauss_up_vis))
                            row1_up.append(apply_colormap_to_diff(diff_mono_up_vis))
                            row1_up.append(consistency_to_rgb(consistency_up))
                            row1_up.append(apply_colormap_to_diff(diff_diff_up))
                            
                            row2_down = []
                            row2_down.append(apply_colormap_to_diff(diff_gauss_down_vis))
                            row2_down.append(apply_colormap_to_diff(diff_mono_down_vis))
                            row2_down.append(consistency_to_rgb(consistency_down))
                            row2_down.append(apply_colormap_to_diff(diff_diff_down))
                            
                            row3_left = []
                            row3_left.append(apply_colormap_to_diff(diff_gauss_left_vis))
                            row3_left.append(apply_colormap_to_diff(diff_mono_left_vis))
                            row3_left.append(consistency_to_rgb(consistency_left))
                            row3_left.append(apply_colormap_to_diff(diff_diff_left))
                            
                            row4_right = []
                            row4_right.append(apply_colormap_to_diff(diff_gauss_right_vis))
                            row4_right.append(apply_colormap_to_diff(diff_mono_right_vis))
                            row4_right.append(consistency_to_rgb(consistency_right))
                            row4_right.append(apply_colormap_to_diff(diff_diff_right))
                            
                            concat_row1 = cv2.hconcat(row1_up)
                            concat_row2 = cv2.hconcat(row2_down)
                            concat_row3 = cv2.hconcat(row3_left)
                            concat_row4 = cv2.hconcat(row4_right)
                            direction_vis_image = cv2.vconcat([concat_row1, concat_row2, concat_row3, concat_row4])
                            
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            h, w = diff_gauss_up_vis.shape
                            labels = ['Gaussian Diff', 'Mono Diff', 'Consistency', 'Difference']
                            for i, label in enumerate(labels):
                                cv2.putText(direction_vis_image, label, (i*w + 10, 30), font, 0.8, (255, 255, 255), 2)
                            
                            direction_labels = ['Up', 'Down', 'Left', 'Right']
                            for i, label in enumerate(direction_labels):
                                cv2.putText(direction_vis_image, label, (10, i*h + 50), font, 0.8, (255, 255, 255), 2)
                            
                            save_name = os.path.splitext(os.path.basename(viewpoint_cam.image_name))[0]
                            fn = os.path.join(direction_vis_dir, f"iter{iteration}_{save_name}_direction_consistency.png")
                            cv2.imwrite(fn, direction_vis_image)
                            
                            del diff_gauss_up_vis, diff_mono_up_vis, diff_gauss_down_vis, diff_mono_down_vis
                            del diff_gauss_left_vis, diff_mono_left_vis, diff_gauss_right_vis, diff_mono_right_vis
                            del consistency_up, consistency_down, consistency_left, consistency_right
                            del diff_diff_up, diff_diff_down, diff_diff_left, diff_diff_right
                    
                        direction_weight = opt.depth_direction_consistency_weight
                        
                        Ll1depth_direction = depth_l1_weight(iteration) * direction_weight * loss_direction_consistency
                        
                        if not torch.isnan(Ll1depth_direction) and not torch.isinf(Ll1depth_direction):
                            loss += Ll1depth_direction
                    
                    if torch.isnan(Ll1depth) or torch.isinf(Ll1depth):
                        print(f'Ll1depth loss is NaN or inf, {viewpoint_cam.image_name}')
                        dpt_loss = None
                    else:
                        loss += Ll1depth
                        dpt_loss = Ll1depth
                else:
                    print(f'Warning: No valid depth pixels for DPT loss, {viewpoint_cam.image_name}')
                    dpt_loss = None

        loss.backward()
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * image_loss.item() + 0.6 * ema_loss_for_log
            ema_single_view_for_log = 0.4 * normal_loss.item() if normal_loss is not None else 0.0 + 0.6 * ema_single_view_for_log
            ema_multi_view_geo_for_log = 0.4 * geo_loss.item() if geo_loss is not None else 0.0 + 0.6 * ema_multi_view_geo_for_log
            ema_multi_view_pho_for_log = 0.4 * ncc_loss.item() if ncc_loss is not None else 0.0 + 0.6 * ema_multi_view_pho_for_log
            if dpt_loss is not None and not torch.isinf(dpt_loss) and not torch.isnan(dpt_loss):
                ema_dpt_loss_for_log = 0.4 * dpt_loss.item() + 0.6 * ema_dpt_loss_for_log
            else:
                ema_dpt_loss_for_log = 0.6 * ema_dpt_loss_for_log
            if opacity_loss is not None:
                ema_opacity_loss_for_log = 0.4 * opacity_loss.item() + 0.6 * ema_opacity_loss_for_log
            else:
                ema_opacity_loss_for_log = 0.6 * ema_opacity_loss_for_log
            if iteration % 10 == 0:
                loss_dict = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "Single": f"{ema_single_view_for_log:.{5}f}",
                    "Geo": f"{ema_multi_view_geo_for_log:.{5}f}",
                    "Pho": f"{ema_multi_view_pho_for_log:.{5}f}",
                    "DPT": f"{ema_dpt_loss_for_log:.{5}f}",
                    "Opacity": f"{ema_opacity_loss_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict)
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), app_model)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                torch.cuda.empty_cache()
                    
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                mask = (render_pkg["out_observe"] > 0) & visibility_filter
                if mask.device != gaussians.max_radii2D.device:
                    mask = mask.to(gaussians.max_radii2D.device)
                if radii.device != gaussians.max_radii2D.device:
                    radii = radii.to(gaussians.max_radii2D.device)
                gaussians.max_radii2D[mask] = torch.max(gaussians.max_radii2D[mask], radii[mask])
                viewspace_point_tensor_abs = render_pkg["viewspace_points_abs"]
                gaussians.add_densification_stats(viewspace_point_tensor, viewspace_point_tensor_abs, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.densify_abs_grad_threshold, 
                                                opt.opacity_cull_threshold, scene.cameras_extent, size_threshold)
            
            # multi-view observe trim
            if opt.use_multi_view_trim and iteration % 1000 == 0 and iteration < opt.densify_until_iter:
                observe_the = 2
                observe_cnt = torch.zeros_like(gaussians.get_opacity)
                for view in scene.getTrainCameras():
                    render_pkg_tmp = render(view, gaussians, pipe, bg, app_model=app_model, return_plane=False, return_depth_normal=False)
                    out_observe = render_pkg_tmp["out_observe"]
                    observe_cnt[out_observe > 0] += 1
                prune_mask = (observe_cnt < observe_the).squeeze()
                if prune_mask.sum() > 0:
                    gaussians.prune_points(prune_mask)

            # reset_opacity
            if iteration < opt.densify_until_iter:
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                app_model.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                app_model.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                with torch.no_grad():
                    checkpoint_data = gaussians.capture()
                torch.save((checkpoint_data, iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                del checkpoint_data
                app_model.save_weights(scene.model_path, iteration)
                torch.cuda.empty_cache()
    
    app_model.save_weights(scene.model_path, opt.iterations)
    torch.cuda.empty_cache()

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, app_model):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    out = renderFunc(viewpoint, scene.gaussians, *renderArgs, app_model=app_model)
                    image = out["render"]
                    if 'app_image' in out:
                        image = out['app_image']
                    image = torch.clamp(image, 0.0, 1.0)
                    gt_image, _ = viewpoint.get_image()
                    gt_image = torch.clamp(gt_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    torch.set_num_threads(8)
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6008)
    parser.add_argument('--debug_from', type=int, default=-100)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--load_iteration", type=int, default=None, help="Initialize training from model at specified iteration (e.g., 30000)")
    parser.add_argument("--load_model_path", type=str, default=None, help="Load model from specified path, if None then load from model_path")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    setup_seed(args.random_seed)
    print(f"\n{'='*60}")
    print(f"Random seed set to: {args.random_seed}")
    print(f"{'='*60}\n")
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, load_iteration=args.load_iteration, load_model_path=args.load_model_path)

    # All done
    print("\nTraining complete.")
