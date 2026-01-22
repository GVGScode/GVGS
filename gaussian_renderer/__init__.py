#
# This file is adapted from a publicly available 3D Gaussian Splatting implementation.
# All identifying information has been removed to preserve double-blind review.
#
# The original license applies to this code.
#

import torch
import math
from diff_plane_rasterization import GaussianRasterizationSettings as PlaneGaussianRasterizationSettings
from diff_plane_rasterization import GaussianRasterizer as PlaneGaussianRasterizer
from scene.gaussian_model import GaussianModel
from scene.app_model import AppModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image

def render_normal(viewpoint_cam, depth, offset=None, normal=None, scale=1):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf(scale=scale)
    st = max(int(scale/2)-1,0)
    if offset is not None:
        offset = offset[st::scale,st::scale]
    normal_ref = normal_from_depth_image(depth[st::scale,st::scale], 
                                            intrinsic_matrix.to(depth.device), 
                                            extrinsic_matrix.to(depth.device), offset)

    normal_ref = normal_ref.permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, 
           app_model: AppModel=None, return_plane = True, return_depth_normal = True, get_infl = True, gaussian_mask = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    screenspace_points_abs = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
        screenspace_points_abs.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    means3D = pc.get_xyz
    means2D = screenspace_points
    means2D_abs = screenspace_points_abs
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    return_dict = None
    if gaussian_mask is None:
        gaussian_mask = torch.Tensor([])
    
    raster_settings = PlaneGaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            render_geo=return_plane,
            debug=pipe.debug,
            get_infl=get_infl,
            gaussian_mask=gaussian_mask
        )

    rasterizer = PlaneGaussianRasterizer(raster_settings=raster_settings)

    if not return_plane:
        rendered_image, infl, radii, out_observe, _, _, opacity_map, accT_masked = rasterizer(
            means3D = means3D,
            means2D = means2D,
            means2D_abs = means2D_abs,
            shs = shs,
            colors_precomp = colors_precomp,
            opacities = opacity,
            scales = scales,
            rotations = rotations,
            cov3D_precomp = cov3D_precomp)
        
        return_dict =  {"render": rendered_image,
                        "viewspace_points": screenspace_points,
                        "viewspace_points_abs": screenspace_points_abs,
                        "visibility_filter" : radii > 0,
                        "radii": radii,
                        "out_observe": out_observe,
                        "infl": infl,
                        "opacity": opacity_map,
                        "accT_masked": accT_masked}
        if app_model is not None and pc.use_app:
            appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
            app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
            return_dict.update({"app_image": app_image})
        return return_dict

    global_normal = pc.get_normal(viewpoint_camera)
    local_normal = global_normal @ viewpoint_camera.world_view_transform[:3,:3]
    pts_in_cam = means3D @ viewpoint_camera.world_view_transform[:3,:3] + viewpoint_camera.world_view_transform[3,:3]
    depth_z = pts_in_cam[:, 2]
    local_distance = (local_normal * pts_in_cam).sum(-1).abs()
    input_all_map = torch.zeros((means3D.shape[0], 5)).cuda().float()
    input_all_map[:, :3] = local_normal
    input_all_map[:, 3] = 1.0
    input_all_map[:, 4] = local_distance

    rendered_image, infl, radii, out_observe, out_all_map, plane_depth, opacity_map, accT_masked = rasterizer(
        means3D = means3D,
        means2D = means2D,
        means2D_abs = means2D_abs,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        all_map = input_all_map,
        cov3D_precomp = cov3D_precomp)

    rendered_normal = out_all_map[0:3]
    rendered_alpha = out_all_map[3:4, ]
    rendered_distance = out_all_map[4:5, ]
    
    return_dict =  {"render": rendered_image,
                    "viewspace_points": screenspace_points,
                    "viewspace_points_abs": screenspace_points_abs,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "out_observe": out_observe,
                    "infl": infl,
                    "rendered_normal": rendered_normal,
                    "plane_depth": plane_depth,
                    "rendered_distance": rendered_distance,
                    "opacity": opacity_map,
                    "accT_masked": accT_masked
                    }
    
    if app_model is not None and pc.use_app:
        appear_ab = app_model.appear_ab[torch.tensor(viewpoint_camera.uid).cuda()]
        app_image = torch.exp(appear_ab[0]) * rendered_image + appear_ab[1]
        return_dict.update({"app_image": app_image})   

    if return_depth_normal:
        depth_normal = render_normal(viewpoint_camera, plane_depth.squeeze()) * (rendered_alpha).detach()
        return_dict.update({"depth_normal": depth_normal})
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return return_dict


def render_with_view_influence_filter(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
                                influence_score_threshold = 1.0, image_shape=None, scaling_modifier = 1.0, 
                                override_color = None, render_depth = False, get_infl = True):
    """
    influence
    ，influence
    
    Args:
        viewpoint_camera: 
        pc: 
        pipe: 
        bg_color: 
        influence_score_threshold: ，
        image_shape: 
        scaling_modifier: 
        override_color: 
        render_depth: 
        get_infl: 
    
    Returns:
    """
    
    # if not hasattr(pc, 'influence') or len(pc.influence) == 0:
    #     print("Warning: No influence information available, rendering all points")
    #     return render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, 
    #                  override_color, return_plane=True, return_depth_normal=True)
    
    print("First pass: determining visible Gaussians...")
    full_result = render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, 
                        override_color, return_plane=True, return_depth_normal=True)
    
    visibility_filter = full_result["visibility_filter"]
    visible_indices = torch.where(visibility_filter)[0]
    
    if len(visible_indices) == 0:
        print("Warning: No visible Gaussians found")
        return full_result
    
    visible_influence = full_result["infl"][visible_indices]
    if visible_influence.dim() > 1:
        visible_influence = visible_influence.squeeze()
    
    selected_visible_indices = visible_indices[visible_influence > influence_score_threshold]
    print(f"Selected {len(selected_visible_indices)} visible Gaussians out of {len(visible_indices)} total, threshold: {influence_score_threshold}")
    
    total_points = pc.get_xyz.shape[0]
    mask = torch.zeros(total_points, dtype=torch.bool, device="cuda")
    mask[selected_visible_indices] = True
    
    class FilteredGaussianModel:
        def __init__(self, original_pc, mask):
            self.original_pc = original_pc
            self.mask = mask
            
        @property
        def get_xyz(self):
            return self.original_pc.get_xyz[self.mask]
            
        @property
        def get_opacity(self):
            return self.original_pc.get_opacity[self.mask]
            
        @property
        def get_features(self):
            return self.original_pc.get_features[self.mask]
            
        @property
        def get_scaling(self):
            return self.original_pc.get_scaling[self.mask]
            
        @property
        def get_rotation(self):
            return self.original_pc.get_rotation[self.mask]
            
        def get_covariance(self, scaling_modifier):
            return self.original_pc.get_covariance(scaling_modifier)[self.mask]
            
        def get_normal(self, viewpoint_camera):
            return self.original_pc.get_normal(viewpoint_camera)[self.mask]
            
        @property
        def active_sh_degree(self):
            return self.original_pc.active_sh_degree
            
        @property
        def max_sh_degree(self):
            return self.original_pc.max_sh_degree
    
    print("Second pass: rendering with selected Gaussians...")
    filtered_pc = FilteredGaussianModel(pc, mask)
    
    result = render(viewpoint_camera, filtered_pc, pipe, bg_color, scaling_modifier, 
                   override_color, return_plane=True, return_depth_normal=True)
    
    original_visibility_filter = torch.zeros(total_points, dtype=torch.bool, device="cuda")
    original_visibility_filter[mask] = result["visibility_filter"]
    
    original_radii = torch.zeros(total_points, device="cuda")
    original_radii[mask] = result["radii"].float()

    original_influence = torch.zeros(total_points, device="cuda")
    if "infl" in result:
        original_influence[mask] = result["infl"]
    
    result["visibility_filter"] = original_visibility_filter
    result["radii"] = original_radii
    result["infl"] = original_influence
    
    print(f"Rendering complete: {len(selected_visible_indices)} Gaussians rendered")
    
    return result


def render_with_influence_filter(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, 
                                influence_ratio = 1.0, image_shape=None, scaling_modifier = 1.0, 
                                override_color = None, render_depth = False, get_infl = True):
    """
    influence
    ，influence
    
    Args:
        viewpoint_camera: 
        pc: 
        pipe: 
        bg_color: 
        influence_ratio:  (0.0-1.0)，1.0
        image_shape: 
        scaling_modifier: 
        override_color: 
        render_depth: 
        get_infl: 
    
    Returns:
    """
    
    if influence_ratio >= 1.0:
        return render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, 
                     override_color, return_plane=True, return_depth_normal=True)
    
    if not hasattr(pc, 'influence') or len(pc.influence) == 0:
        print("Warning: No influence information available, rendering all points")
        return render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, 
                     override_color, return_plane=True, return_depth_normal=True)
    
    print("First pass: determining visible Gaussians...")
    full_result = render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, 
                        override_color, return_plane=True, return_depth_normal=True)
    
    visibility_filter = full_result["visibility_filter"]
    visible_indices = torch.where(visibility_filter)[0]
    
    if len(visible_indices) == 0:
        print("Warning: No visible Gaussians found")
        return full_result
    
    print(f"Found {len(visible_indices)} visible Gaussians out of {pc.get_xyz.shape[0]} total")
    
    visible_influence = pc.influence[visible_indices]
    if visible_influence.dim() > 1:
        visible_influence = visible_influence.squeeze()
    
    num_visible_points = len(visible_indices)
    num_points_to_keep = max(1, int(num_visible_points * influence_ratio))
    
    print(f"Selecting {num_points_to_keep} out of {num_visible_points} visible points (ratio: {influence_ratio:.2f})")
    
    _, top_indices = torch.topk(visible_influence, num_points_to_keep, largest=True)
    selected_visible_indices = visible_indices[top_indices]
    
    total_points = pc.get_xyz.shape[0]
    mask = torch.zeros(total_points, dtype=torch.bool, device="cuda")
    mask[selected_visible_indices] = True
    
    class FilteredGaussianModel:
        def __init__(self, original_pc, mask):
            self.original_pc = original_pc
            self.mask = mask
            
        @property
        def get_xyz(self):
            return self.original_pc.get_xyz[self.mask]
            
        @property
        def get_opacity(self):
            return self.original_pc.get_opacity[self.mask]
            
        @property
        def get_features(self):
            return self.original_pc.get_features[self.mask]
            
        @property
        def get_scaling(self):
            return self.original_pc.get_scaling[self.mask]
            
        @property
        def get_rotation(self):
            return self.original_pc.get_rotation[self.mask]
            
        def get_covariance(self, scaling_modifier):
            return self.original_pc.get_covariance(scaling_modifier)[self.mask]
            
        def get_normal(self, viewpoint_camera):
            return self.original_pc.get_normal(viewpoint_camera)[self.mask]
            
        @property
        def active_sh_degree(self):
            return self.original_pc.active_sh_degree
            
        @property
        def max_sh_degree(self):
            return self.original_pc.max_sh_degree
    
    print("Second pass: rendering with selected Gaussians...")
    filtered_pc = FilteredGaussianModel(pc, mask)
    
    result = render(viewpoint_camera, filtered_pc, pipe, bg_color, scaling_modifier, 
                   override_color, return_plane=True, return_depth_normal=True)
    
    original_visibility_filter = torch.zeros(total_points, dtype=torch.bool, device="cuda")
    original_visibility_filter[mask] = result["visibility_filter"]
    
    original_radii = torch.zeros(total_points, device="cuda")
    original_radii[mask] = result["radii"].float()

    original_influence = torch.zeros(total_points, device="cuda")
    if "infl" in result:
        original_influence[mask] = result["infl"]
    
    result["visibility_filter"] = original_visibility_filter
    result["radii"] = original_radii
    result["infl"] = original_influence
    
    print(f"Rendering complete: {len(selected_visible_indices)} Gaussians rendered")
    
    return result



def render_with_mask(viewpoint_camera, pc:GaussianModel, pipe, bg_color, scaling_modifier=1.0, override_color=None, mask=None, app_model:AppModel=None, return_plane=False, return_depth_normal=False):
    """
    Render only the Gaussians specified by the mask.

    Args:
        viewpoint_camera: The camera viewpoint.
        pc: The Gaussian point cloud model.
        pipe: The rendering pipeline.
        bg_color: Background color.
        scaling_modifier: Scaling modifier for Gaussians.
        override_color: Optional color override.
        mask: Boolean mask indicating which Gaussians to render.
        return_plane: Whether to return the rendered plane.
        return_depth_normal: Whether to return depth and normal maps.

    Returns:
        The rendering result dictionary.
    """
    if mask is not None:
        # Apply mask to the point cloud
        class MaskedGaussianModel:
            def __init__(self, original_pc, mask):
                self.original_pc = original_pc
                self.mask = mask

            @property
            def get_xyz(self):
                return self.original_pc.get_xyz[self.mask]

            @property
            def get_scaling(self):
                return self.original_pc.get_scaling[self.mask]

            @property
            def get_rotation(self):
                return self.original_pc.get_rotation[self.mask]

            def get_covariance(self, scaling_modifier):
                return self.original_pc.get_covariance(scaling_modifier)[self.mask]

            def get_normal(self, viewpoint_camera):
                return self.original_pc.get_normal(viewpoint_camera)[self.mask]

            @property
            def active_sh_degree(self):
                return self.original_pc.active_sh_degree

            @property
            def max_sh_degree(self):
                return self.original_pc.max_sh_degree

            @property
            def get_features(self):
                return self.original_pc.get_features[self.mask]

            @property
            def get_opacity(self):
                return self.original_pc.get_opacity[self.mask]

        filtered_pc = MaskedGaussianModel(pc, mask)
    else:
        filtered_pc = pc

    result = render(
        viewpoint_camera, filtered_pc, pipe, bg_color, scaling_modifier,
        override_color, return_plane=return_plane, return_depth_normal=return_depth_normal
    )
    return result


def render_with_color_mask(viewpoint_camera, pc: GaussianModel, pipe, bg_color, scaling_modifier=1.0, mask=None, app_model: AppModel=None, return_plane=False, return_depth_normal=False):
    """
    Render with color masking: mask，mask。
    
    Args:
        viewpoint_camera: The camera viewpoint.
        pc: The Gaussian point cloud model.
        pipe: The rendering pipeline.
        bg_color: Background color.
        scaling_modifier: Scaling modifier for Gaussians.
        mask: Boolean mask indicating which Gaussians to render as white (True=white, False=black).
        return_plane: Whether to return the rendered plane.
        return_depth_normal: Whether to return depth and normal maps.
        
    Returns:
        The rendering result dictionary with color-masked rendering.
    """
    if mask is None:
        mask = torch.ones(pc.get_xyz.shape[0], dtype=torch.bool, device="cuda")
    
    white_color = torch.ones((pc.get_xyz.shape[0], 3), device="cuda")
    black_color = torch.zeros((pc.get_xyz.shape[0], 3), device="cuda")
    
    override_color = torch.where(mask.unsqueeze(1), white_color, black_color)
    
    result = render(
        viewpoint_camera, pc, pipe, bg_color, scaling_modifier,
        override_color=override_color, return_plane=return_plane, return_depth_normal=return_depth_normal
    )
    
    return result
