import shutil
from termcolor import colored
import ipdb
import yaml
import os
import random
import hydra
import logging
from omegaconf import OmegaConf

import numpy as np
import torch

from src.slam import SLAM
from src.datasets import get_dataset

"""
Run the SLAM system on a given dataset or on image folder.
You can configure the system using .yaml configs. See docs for reference ...
"""

# A logger for this file
log = logging.getLogger(__name__)


def sys_print(msg: str) -> None:
    log.info(colored(msg, "white", "on_grey", attrs=["bold"]))


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def backup_source_code(backup_directory):
    ignore_hidden = shutil.ignore_patterns(
        ".",
        "..",
        ".git*",
        "*pycache*",
        "*build",
        "*ext",
        "*thirdparty",
        "*.fuse*",
        "*_drive_*",
        "*pretrained*",
        "*output*",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.mp4",
        "*.gif",
        "*media*",
        "*.so",
        "*.pyc",
        "*.Python",
        "*.eggs*",
        "*.DS_Store*",
        "*.idea*",
        "*.pth",
        "*__pycache__*",
        "*.ply",
        "*exps*",
    )

    if os.path.exists(backup_directory):
        shutil.rmtree(backup_directory)

    shutil.copytree(".", backup_directory, ignore=ignore_hidden)
    os.system("chmod -R g+w {}".format(backup_directory))


def get_in_the_wild_heuristics(ht: int, wd: int, strategy: str = "generic") -> torch.Tensor:
    """We do not have camera intrinsics on in-the-wild data. In order for this to converge, we
    need a good initialize guess. There are two strategies to do this: i) generc ii) Teeds from DeepV2D
    """
    if strategy == "generic":
        fx = fy = (wd + ht) / 2
        cx, cy = wd / 2, ht / 2
    else:
        fx = fy = wd * 1.2
        cx, cy = wd / 2, ht / 2
    return fx, fy, cx, cy


def estimate_intrinsics_with_dm_calib(cfg, output_folder: str):
    """Estimate camera intrinsics using DM-Calib from the first frame of the input sequence.

    This function is completely independent from the opt_intr feature in BA.
    It estimates intrinsics BEFORE the SLAM system starts, using a diffusion-based model.

    Args:
        cfg: Hydra configuration object

    Returns:
        Updated cfg with estimated intrinsics
    """
    if not cfg.get("use_dm_calib", False):
        return cfg

    sys_print("=" * 50)
    sys_print("Using DM-Calib to estimate camera intrinsics")
    sys_print("=" * 50)

    try:
        from src.utils.dm_calib import DMCalibEstimator
    except ImportError as e:
        sys_print(f"Warning: Could not import DM-Calib module: {e}")
        sys_print("Falling back to config intrinsics")
        return cfg

    # Get DM-Calib parameters from config
    dm_calib_model_path = cfg.get("dm_calib_model_path", "./pretrained/DM-Calib")
    dm_calib_domain = cfg.get("dm_calib_domain", "object")
    dm_calib_processing_res = cfg.get("dm_calib_processing_res", 768)
    dm_calib_denoise_steps = cfg.get("dm_calib_denoise_steps", 20)
    dm_calib_ensemble_size = cfg.get("dm_calib_ensemble_size", 3)
    device = cfg.get("device", "cuda:0")

    # Initialize estimator
    estimator = DMCalibEstimator(
        pretrained_model_path=dm_calib_model_path,
        device=device,
        processing_res=dm_calib_processing_res,
        denoise_steps=dm_calib_denoise_steps,
        ensemble_size=dm_calib_ensemble_size,
    )

    # Save intermediate camera representation for debugging (controlled by slam.yaml).
    save_camera_img_dir = None
    if cfg.get("save_evaluation_intermediate", False):
        save_camera_img_dir = os.path.join(output_folder, "evaluation", "dmcalib", "pre_fov_adaptation", "camera_images")
        os.makedirs(save_camera_img_dir, exist_ok=True)

    # Get input folder and find first image
    input_folder = cfg.data.input_folder

    # Determine dataset type and find first image
    dataset_name = cfg.data.dataset.lower()

    # Find first image based on dataset type
    import glob as glob_lib
    image_extensions = ['.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG']

    first_image = None
    if dataset_name == 'replica':
        color_paths = sorted(glob_lib.glob(os.path.join(input_folder, "color/*.png")))
        if color_paths:
            first_image = color_paths[0]
    elif dataset_name == 'scannet':
        color_paths = sorted(glob_lib.glob(os.path.join(input_folder, "color/*.jpg")))
        if color_paths:
            first_image = color_paths[0]
    elif dataset_name == 'tumrgbd':
        # TUM-RGBD has rgb.txt
        import pandas as pd
        rgb_txt = os.path.join(input_folder, "rgb.txt")
        if os.path.exists(rgb_txt):
            df = pd.read_csv(rgb_txt, delimiter=' ', skiprows=1, usecols=[1])
            first_image_path = df.iloc[0, 0]
            first_image = os.path.join(input_folder, first_image_path)
    elif dataset_name in ['tartanair', 'kitti', 'eth3d', 'euroc', 'sintel', 'davis']:
        # Try common patterns
        for ext in image_extensions:
            patterns = [
                os.path.join(input_folder, f"*{ext}"),
                os.path.join(input_folder, "image*", f"*{ext}"),
                os.path.join(input_folder, "images", f"*{ext}"),
            ]
            for pattern in patterns:
                images = sorted(glob_lib.glob(pattern))
                if images:
                    first_image = images[0]
                    break
            if first_image:
                break
    else:
        # Generic folder - try to find any image
        for ext in image_extensions:
            images = sorted(glob_lib.glob(os.path.join(input_folder, f"*{ext}")))
            if images:
                first_image = images[0]
                break
        # Also check subdirectories
        if not first_image:
            for ext in image_extensions:
                images = sorted(glob_lib.glob(os.path.join(input_folder, "**", f"*{ext}")))
                if images:
                    first_image = images[0]
                    break

    if first_image is None or not os.path.exists(first_image):
        sys_print(f"Warning: Could not find first image in {input_folder}")
        sys_print("Falling back to config intrinsics")
        return cfg

    sys_print(f"Using first image: {first_image}")

    # Estimate intrinsics
    try:
        intrinsics, K_matrix = estimator.estimate_intrinsics(
            first_image,
            domain=dm_calib_domain,
            save_camera_img_dir=save_camera_img_dir,
        )

        # Update config with estimated intrinsics
        old_fx = cfg.data.cam.fx
        old_fy = cfg.data.cam.fy
        old_cx = cfg.data.cam.cx
        old_cy = cfg.data.cam.cy

        cfg.data.cam.fx = intrinsics['fx']
        cfg.data.cam.fy = intrinsics['fy']
        cfg.data.cam.cx = intrinsics['cx']
        cfg.data.cam.cy = intrinsics['cy']

        sys_print("=" * 50)
        sys_print("DM-Calib Intrinsics Estimation Results:")
        sys_print(f"  Old intrinsics: fx={old_fx}, fy={old_fy}, cx={old_cx}, cy={old_cy}")
        sys_print(f"  New intrinsics: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
                  f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
        sys_print(f"  K matrix:\n{K_matrix}")
        sys_print("=" * 50)
        sys_print("Note: This is completely independent from opt_intr (BA optimization)")

    except Exception as e:
        sys_print(f"Warning: DM-Calib estimation failed: {e}")
        sys_print("Falling back to config intrinsics")

    # Clean up DM-Calib model to free GPU memory before starting SLAM
    if 'estimator' in locals() and estimator is not None:
        try:
            estimator.cleanup()
            sys_print("DM-Calib model cleaned up, GPU memory freed")
        except Exception as cleanup_error:
            sys_print(f"Warning: Failed to cleanup DM-Calib model: {cleanup_error}")

    return cfg


@hydra.main(version_base=None, config_path="./configs/", config_name="slam")
def run_slam(cfg):

    output_folder = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    log.info(OmegaConf.to_yaml(cfg))
    # Save the cfg to yaml file
    with open(os.path.join(output_folder, "config.yaml"), "w") as f:
        yaml.dump(OmegaConf.to_container(cfg), f, default_flow_style=False)

    setup_seed(43)
    torch.multiprocessing.set_start_method("spawn")
    # Save state for reproducibility
    backup_source_code(os.path.join(output_folder, "code"))

    sys_print(f"\n\n** Running {cfg.data.input_folder} in {cfg.mode} mode!!! **\n\n")

    # Estimate intrinsics using DM-Calib BEFORE starting SLAM (completely independent from opt_intr)
    cfg = estimate_intrinsics_with_dm_calib(cfg, output_folder)

    if cfg.data.cam.fx is None or cfg.data.cam.fy is None:
        sys_print("Using generic intrinsics for in-the-wild data")
        cfg.data.cam.fx, cfg.data.cam.fy, cfg.data.cam.cx, cfg.data.cam.cy = get_in_the_wild_heuristics(
            ht=cfg.data.cam.H, wd=cfg.data.cam.W
        )
    dataset = get_dataset(cfg, device=cfg.device)
    slam = SLAM(cfg, dataset=dataset, output_folder=output_folder)

    sys_print(f"Running on {len(dataset)} frames")
    slam.run(dataset)
    sys_print("Done!")


if __name__ == "__main__":
    run_slam()
