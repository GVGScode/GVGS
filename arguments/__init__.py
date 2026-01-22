#
# This file is adapted from a publicly available 3D Gaussian Splatting implementation.
# All identifying information has been removed to preserve double-blind review.
#
# The original license applies to this code.
#

from argparse import ArgumentParser, Namespace
import sys
import os

def _detect_dataset_type():
    """
    Detect dataset type from command line arguments
    Returns 'tnt' or 'dtu'
    """
    source_path = None
    model_path = None
    
    for i, arg in enumerate(sys.argv[1:], start=1):
        if arg in ['-s', '--source_path'] and i + 1 < len(sys.argv):
            source_path = sys.argv[i + 1]
        elif arg in ['-m', '--model_path'] and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]
    
    path_to_check = source_path or model_path
    if path_to_check:
        path_lower = path_to_check.lower()
        if 'scan' in path_lower:
            return 'dtu'
        mip360_scenes = ['bicycle', 'bonsai', 'counter', 'flowers', 'garden', 'kitchen', 'room', 'stump', 'treehill', 'mipnerf360', 'mip360']
        if any(scene in path_lower for scene in mip360_scenes):
            return 'dtu'
        elif any(scene in path_lower for scene in ['barn', 'truck', 'caterpillar', 'ignatius', 'meetingroom', 'courthouse', 'tnt']):
            return 'tnt'
    
    return 'tnt'

_dataset_type = _detect_dataset_type()

if _dataset_type == 'dtu':
    from arguments import __init_dtu__ as dtu_args
    ModelParams = dtu_args.ModelParams
    PipelineParams = dtu_args.PipelineParams
    OptimizationParams = dtu_args.OptimizationParams
    get_combined_args = dtu_args.get_combined_args
    GroupParams = dtu_args.GroupParams
    ParamGroup = dtu_args.ParamGroup
    print(f"[Config] Detected DTU dataset, using parameters from arguments/__init_dtu__.py")
else:
    print(f"[Config] Detected TNT dataset, using parameters from arguments/__init__.py")
    
    class GroupParams:
        pass

    class ParamGroup:
        def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
            group = parser.add_argument_group(name)
            for key, value in vars(self).items():
                shorthand = False
                if key.startswith("_"):
                    shorthand = True
                    key = key[1:]
                t = type(value)
                value = value if not fill_none else None 
                if shorthand:
                    if t == bool:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                    else:
                        group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
                else:
                    if t == bool:
                        group.add_argument("--" + key, default=value, action="store_true")
                    else:
                        group.add_argument("--" + key, default=value, type=t)

        def extract(self, args):
            group = GroupParams()
            for arg in vars(args).items():
                if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                    setattr(group, arg[0], arg[1])
            return group

    class ModelParams(ParamGroup): 
        def __init__(self, parser, sentinel=False):
            self.sh_degree = 3
            self._source_path = ""
            self._model_path = ""
            self._images = "images"
            self._resolution = -1
            self._white_background = False
            self.data_device = "cuda"
            self.eval = False
            self.preload_img = True
            self.ncc_scale = 1.0
            self.multi_view_num = 8
            self.multi_view_max_angle = 30
            self.multi_view_min_dis = 0.01
            self.multi_view_max_dis = 1.5
            self.use_mvsformer_init = False
            self.use_da3_init = False
            self.da3mono = False
            super().__init__(parser, "Loading Parameters", sentinel)

        def extract(self, args):
            g = super().extract(args)
            g.source_path = os.path.abspath(g.source_path)
            return g

    class PipelineParams(ParamGroup):
        def __init__(self, parser):
            self.convert_SHs_python = False
            self.compute_cov3D_python = False
            self.debug = False
            self.random_seed = 22
            super().__init__(parser, "Pipeline Parameters")

    class OptimizationParams(ParamGroup):
        def __init__(self, parser):
            self.iterations = 60_000
            self.position_lr_init = 0.00016
            self.position_lr_final = 0.0000016
            self.position_lr_delay_mult = 0.01
            self.position_lr_max_steps = 60_000
            self.feature_lr = 0.0025
            self.opacity_lr = 0.05
            self.scaling_lr = 0.005
            self.rotation_lr = 0.001
            self.percent_dense = 0.001
            self.lambda_dssim = 0.2
            self.densification_interval = 200
            self.opacity_reset_interval = 6000
            self.densify_from_iter = 400
            self.densify_until_iter = 48_000
            self.densify_grad_threshold = 0.0002
            self.scale_loss_weight = 100.0
            
            self.wo_image_weight = False
            self.use_boolean_mask = False
            self.single_view_weight = 0.015
            self.single_view_weight_from_iter = 25000

            self.use_virtul_cam = False
            self.virtul_cam_prob = 0.5
            self.use_multi_view_trim = True
            self.multi_view_ncc_weight = 0.15
            self.multi_view_geo_weight = 0.03
            self.multi_view_weight_from_iter = 25000
            self.multi_view_patch_size = 3
            self.multi_view_sample_num = 102400
            # Pixel noise threshold
            self.multi_view_pixel_noise_th_init = 1.0
            self.multi_view_pixel_noise_th_final = 1.0
            self.multi_view_pixel_noise_th_end_iter = 60000
            self.wo_use_geo_occ_aware = False
            
            self.use_consistency_loss = False
            self.save_debug_images = True
            
            # DPT loss parameters
            self.dpt_loss_from_iter = 25000
            self.dpt_end_iter = 55000
            self.depth_l1_weight_init = 4.0
            self.depth_l1_weight_final = 2.0
            self.depth_l1_loss_scale = 0.01
            self.use_dpt_loss = False
            self.use_unmasked_depth = False
            self.use_depth_correction_network = False
            self.mono_dpt_trunc_near = 0.0
            self.mono_dpt_trunc_far = float('inf')
            self.use_depth_percentile_filter = False
            self.depth_percentile_threshold = 0.3
            
            # Depth direction consistency loss parameters
            self.use_depth_direction_consistency_loss = False
            self.depth_direction_consistency_weight = 0.05
            
            # Quadtree depth alignment parameters
            self.use_quadtree_depth_alignment = False
            self.quadtree_level0_end_iter = 20000
            self.quadtree_level1_end_iter = 30000
            self.quadtree_level2_end_iter = 40000
            self.quadtree_max_level = 3
            self.quadtree_min_valid_pixels = 10
            self.quadtree_min_depth_range = 1e-3
            self.quadtree_filter_outliers = False
            self.quadtree_outlier_threshold = 'otsu'

            # Opacity loss parameters
            self.use_opacity_loss = False
            self.opacity_loss_weight_init = 0.1
            self.opacity_loss_weight_final = 1.0
            self.opacity_loss_from_iter = 14000
            self.opacity_loss_end_iter = 60000

            # GS Opacity loss parameters
            self.use_gs_opacity_loss = False
            self.gs_opacity_loss_weight = 0.001
            self.gs_opacity_loss_from_iter = 20000

            # accT geometric consistency loss parameters
            self.use_accT_geo_consistency_loss = False
            self.accT_geo_consistency_weight = 0.5

            self.opacity_cull_threshold = 0.005
            self.densify_abs_grad_threshold = 0.0008
            self.abs_split_radii2D_threshold = 20
            self.max_abs_split_points = 50_000
            self.max_all_points = 6000_000
            self.exposure_compensation = False
            self.random_background = False
            super().__init__(parser, "Optimization Parameters")

    def get_combined_args(parser : ArgumentParser):
        cmdlne_string = sys.argv[1:]
        cfgfile_string = "Namespace()"
        args_cmdline = parser.parse_args(cmdlne_string)

        try:
            cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
            print("Looking for config file in", cfgfilepath)
            with open(cfgfilepath) as cfg_file:
                print("Config file found: {}".format(cfgfilepath))
                cfgfile_string = cfg_file.read()
        except TypeError:
            print("Config file not found at")
            pass
        args_cfgfile = eval(cfgfile_string)

        merged_dict = vars(args_cfgfile).copy()
        for k,v in vars(args_cmdline).items():
            if v != None:
                merged_dict[k] = v
        return Namespace(**merged_dict)
