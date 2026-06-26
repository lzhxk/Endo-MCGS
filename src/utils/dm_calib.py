"""
DM-Calib Intrinsic Estimation Module for EDroidPGSR-V2-

This module integrates DM-Calib's camera intrinsic parameter estimation
into the EDroidPGSR-V2- SLAM system.
"""

import os
import re
import logging
import numpy as np
import torch
from PIL import Image
from typing import Tuple, Optional, List
from tqdm import tqdm
import torchvision
from skimage.measure import ransac, LineModelND
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def spherical_zbuffer_to_euclidean(spherical_tensor):
    """Convert spherical zbuffer coordinates to Euclidean coordinates."""
    theta = spherical_tensor[..., 0]
    phi = spherical_tensor[..., 1]
    z = spherical_tensor[..., 2]

    x = z * np.tan(theta)
    y = z / np.tan(phi) / np.cos(theta)

    euclidean_tensor = np.stack((x, y, z), axis=-1)
    return euclidean_tensor


def _decode_theta_phi_over_pi(
    pred_theta_img: torch.Tensor,
    pred_phi_img: torch.Tensor,
    *,
    theta_min: float,
    theta_max: float,
    phi_min: float,
    phi_max: float,
    decode_mode: str,
):
    """Decode predicted theta/pi and phi/pi from network output images."""
    if decode_mode == "direct_01":
        theta_over_pi = pred_theta_img * (theta_max - theta_min) + theta_min
        phi_over_pi = pred_phi_img * (phi_max - phi_min) + phi_min
        return theta_over_pi, phi_over_pi

    if decode_mode == "generate_rays_-11":
        theta_norm = pred_theta_img * 2.0 - 1.0
        phi_norm = pred_phi_img * 2.0 - 1.0
        theta_over_pi = (theta_norm / 2.0 + 0.5) * (theta_max - theta_min) + theta_min
        phi_over_pi = (phi_norm / 2.0 + 0.5) * (phi_max - phi_min) + phi_min
        return theta_over_pi, phi_over_pi

    raise ValueError(f"Unknown decode_mode: {decode_mode}")


def _fit_line_ransac(x: np.ndarray, y: np.ndarray):
    """Fit y = a*x + b robustly using RANSAC."""
    data = np.column_stack([x, y])
    model_robust, inliers = ransac(
        data, LineModelND, min_samples=2, residual_threshold=1, max_trials=1000
    )
    b = model_robust.predict_y([0])[0]
    a = (model_robust.params[1][1] / model_robust.params[1][0])
    y_hat = a * x + b
    med = float(np.median(np.abs(y_hat - y)))
    return a, b, med


def calculate_intrinsic(
    pred_image,
    pad=None,
    mask=None,
    *,
    decode_mode: str = "auto",
    phi_min: float = 0.0,
    phi_max: float = 1.0,
    theta_min: float = -0.59,
    theta_max: float = 0.59,
):
    """Calculate camera intrinsic parameters from predicted camera image.

    Args:
        pred_image: Predicted camera image tensor from DM-Calib
        pad: Padding values (pad_left, pad_right, pad_top, pad_bottom)
        mask: Optional mask for valid pixels
        decode_mode: Decoding mode for ray generation
        phi_min: Minimum phi angular range (default: 0.0)
        phi_max: Maximum phi angular range (default: 1.0)
        theta_min: Minimum theta angular range (default: -0.59)
        theta_max: Maximum theta angular range (default: 0.59)
    """

    _, h, w = pred_image.shape
    ori_image = pred_image.clone()
    if pad is not None:
        pad_left, pad_right, pad_top, pad_bottom = pad
        ori_image = ori_image[:, pad_top:h-pad_bottom, pad_left:w-pad_right]
        _, h, w = ori_image.shape

    def _solve_for_mode(mode: str):
        theta_over_pi, phi_over_pi = _decode_theta_phi_over_pi(
            ori_image[0],
            ori_image[1],
            theta_min=theta_min,
            theta_max=theta_max,
            phi_min=phi_min,
            phi_max=phi_max,
            decode_mode=mode,
        )

        if mask is not None:
            theta_np = theta_over_pi[mask > 0.5].reshape(-1).numpy()
            phi_np = phi_over_pi[mask > 0.5].reshape(-1).numpy()
            u = np.tile(np.arange(0, w) + 0.5, h).reshape(h, w)[mask > 0.5]
            v = ((np.arange(0, h) + 0.5).repeat(w).reshape(h, w))[mask > 0.5]
        else:
            theta_np = theta_over_pi.reshape(-1).numpy()
            phi_np = phi_over_pi.reshape(-1).numpy()
            u = np.tile(np.arange(0, w) + 0.5, h)
            v = (np.arange(0, h) + 0.5).repeat(w)

        x_u = np.tan(theta_np * np.pi)
        a_u, b_u, med_u = _fit_line_ransac(x_u, u)
        fx = a_u
        cx = b_u

        x_v = 1.0 / np.tan(phi_np * np.pi) / np.cos(theta_np * np.pi)
        a_v, b_v, med_v = _fit_line_ransac(x_v, v)
        fy = a_v
        cy = b_v

        score = med_u + med_v
        return [fx, fy, cx, cy], score

    if decode_mode == "auto":
        k1, s1 = _solve_for_mode("generate_rays_-11")
        k2, s2 = _solve_for_mode("direct_01")
        return k1 if s1 <= s2 else k2

    k, _ = _solve_for_mode(decode_mode)
    return k


def preprocess_pad(rgb, target_shape):
    """Preprocess image with padding."""
    _, h, w = rgb.shape
    ht, wt = target_shape[0], target_shape[1]
    
    # Calculate padding
    pad_left = max(0, (wt - w) // 2)
    pad_right = max(0, wt - w - pad_left)
    pad_top = max(0, (ht - h) // 2)
    pad_bottom = max(0, ht - h - pad_top)
    
    # Pad the image
    rgb_padded = F.pad(rgb, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)
    
    return rgb_padded, pad_left, pad_right, pad_top, pad_bottom


class DMCalibEstimator:
    """
    DM-Calib Camera Intrinsic Estimator.
    
    This class handles loading the DM-Calib model and estimating camera intrinsics
    from a single RGB image.
    """
    
    def __init__(
        self,
        pretrained_model_path: str = "juneyoung9/DM-Calib",
        device: str = "cuda:0",
        processing_res: int = 768,
        denoise_steps: int = 20,
        ensemble_size: int = 3,
        seed: int = 666,
    ):
        """
        Initialize DM-Calib estimator.
        
        Args:
            pretrained_model_path: Path to pretrained DM-Calib model
            device: Device to run inference on
            processing_res: Processing resolution for inference
            denoise_steps: Number of diffusion denoising steps
            ensemble_size: Number of predictions to ensemble
            seed: Random seed for reproducibility
        """
        self.pretrained_model_path = pretrained_model_path
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")
        self.processing_res = processing_res
        self.denoise_steps = denoise_steps
        self.ensemble_size = ensemble_size
        self.seed = seed
        
        self.pipeline = None
        self.totensor = torchvision.transforms.ToTensor()

    @property
    def _pipeline(self):
        """Lazy-load and expose the pipeline."""
        self._ensure_pipeline_loaded()
        return self.pipeline

    def _ensure_pipeline_loaded(self):
        """Lazy loading of the DM-Calib pipeline."""
        if self.pipeline is not None:
            return

        try:
            from diffusers import DDIMScheduler, AutoencoderKL, UNet2DConditionModel
            from src.utils.pipeline.pipeline_sd21_scale_vae import StableDiffusion21
            from transformers import CLIPTextModel, CLIPTokenizer
            from modelscope import snapshot_download
        except ImportError as e:
            logger.error(f"Missing required package: {e}")
            logger.info("Please install DM-Calib dependencies: pip install diffusers transformers modelscope")
            raise

        logger.info(f"Loading DM-Calib model from {self.pretrained_model_path}")

        # Load Stable Diffusion components
        stable_diffusion_repo_path = snapshot_download("AI-ModelScope/stable-diffusion-2-1")
        
        text_encoder = CLIPTextModel.from_pretrained(
            stable_diffusion_repo_path, subfolder="text_encoder"
        )
        scheduler = DDIMScheduler.from_pretrained(
            stable_diffusion_repo_path, subfolder="scheduler"
        )
        tokenizer = CLIPTokenizer.from_pretrained(
            stable_diffusion_repo_path, subfolder="tokenizer"
        )
        
        # Load intrinsic estimation model
        vae_cam = AutoencoderKL.from_pretrained(stable_diffusion_repo_path, subfolder="vae")
        unet_cam = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_path, subfolder="calib"
        )
        
        self.pipeline = StableDiffusion21(
            vae=vae_cam,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            unet=unet_cam,
            scheduler=scheduler,
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        
        try:
            self.pipeline.enable_xformers_memory_efficient_attention()
        except:
            pass  # Run without xformers
            
        self.pipeline = self.pipeline.to(self.device)
        logger.info("DM-Calib model loaded successfully")

    def cleanup(self):
        """Release DM-Calib model from GPU memory."""
        if self.pipeline is not None:
            logger.info("Releasing DM-Calib model from GPU memory")
            # Move pipeline to CPU and delete
            self.pipeline = self.pipeline.to('cpu')
            del self.pipeline
            self.pipeline = None

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info("DM-Calib model cleanup completed")

    def estimate_intrinsics(
        self,
        image_path: str,
        domain: str = "object",
        phi_min: float = 0.0,
        phi_max: float = 1.0,
        theta_min: float = -0.59,
        theta_max: float = 0.59,
        save_camera_img_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate camera intrinsics from an image.

        Args:
            image_path: Path to input RGB image
            domain: Domain type ('indoor', 'outdoor', 'object')
            phi_min: Minimum phi angular range (default: 0.0)
            phi_max: Maximum phi angular range (default: 1.0)
            theta_min: Minimum theta angular range (default: -0.59)
            theta_max: Maximum theta angular range (default: 0.59)

        Returns:
            Tuple of (intrinsics_dict, K_matrix)
            - intrinsics_dict: {'fx', 'fy', 'cx', 'cy'}
            - K_matrix: 3x3 camera intrinsic matrix
        """
        self._ensure_pipeline_loaded()
        
        # Read input image
        input_image = Image.open(image_path).convert('RGB')
        w_ori, h_ori = input_image.size
        
        # Resize for processing
        max_dim = max(w_ori, h_ori)
        if max_dim > self.processing_res:
            scale = self.processing_res / max_dim
            new_w = int(w_ori * scale)
            new_h = int(h_ori * scale)
            input_image = input_image.resize((new_w, new_h), Image.LANCZOS)
        
        img = self.totensor(input_image)
        c, h, w = img.shape
        
        # Preprocess with padding
        img_pad, pad_left, pad_right, pad_top, pad_bottom = preprocess_pad(
            img, (self.processing_res, self.processing_res)
        )
        
        # Run inference
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        
        camera_img = self.pipeline(
            image=img_pad.repeat(self.ensemble_size, 1, 1, 1),
            height=self.processing_res,
            width=self.processing_res,
            num_inference_steps=self.denoise_steps,
            guidance_scale=1,
            generator=generator,
        ).images
        
        # Average ensemble predictions
        camera_img = torch.stack(
            [self.totensor(camera_img[i]) for i in range(self.ensemble_size)]
        ).mean(0, keepdim=True)

        # Optionally save the intermediate "camera image" representation for debugging/visualization.
        if save_camera_img_dir is not None:
            os.makedirs(save_camera_img_dir, exist_ok=True)
            stem = os.path.splitext(os.path.basename(image_path))[0]
            m = re.search(r"(\d+)", stem)
            if m:
                digits = m.group(1)
                frame_id = int(digits)
                pad_width = len(digits)
                cam = (
                    camera_img[0]
                    .detach()
                    .to(torch.float32)
                    .clamp(0.0, 1.0)
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                cam_u8 = (cam * 255.0).astype(np.uint8)
                out_path = os.path.join(save_camera_img_dir, f"camera_img_{str(frame_id).zfill(pad_width)}.png")
                Image.fromarray(cam_u8).save(out_path)
        
        # Calculate intrinsics
        intrin_pred = calculate_intrinsic(
            camera_img[0],
            (pad_left, pad_right, pad_top, pad_bottom),
            mask=None,
            phi_min=phi_min,
            phi_max=phi_max,
            theta_min=theta_min,
            theta_max=theta_max,
        )
        
        # Build K matrix at resized resolution
        K = np.eye(3)
        K[0, 0] = intrin_pred[0]
        K[1, 1] = intrin_pred[1]
        K[0, 2] = intrin_pred[2]
        K[1, 2] = intrin_pred[3]
        
        logger.info(f"Camera intrinsic (resized image): {K}")
        
        # Scale K back to original image resolution
        scale_x = w_ori / w
        scale_y = h_ori / h
        K_ori = K.copy()
        K_ori[0, 0] *= scale_x
        K_ori[1, 1] *= scale_y
        K_ori[0, 2] *= scale_x
        K_ori[1, 2] *= scale_y
        
        logger.info(f"Camera intrinsic (original image size): {K_ori}")
        
        intrinsics_dict = {
            'fx': float(K_ori[0, 0]),
            'fy': float(K_ori[1, 1]),
            'cx': float(K_ori[0, 2]),
            'cy': float(K_ori[1, 2]),
        }
        
        return intrinsics_dict, K_ori
    
    def estimate_from_folder(
        self,
        input_folder: str,
        output_path: Optional[str] = None,
        domain: str = "object",
        image_extensions: List[str] = None,
    ) -> List[dict]:
        """
        Estimate intrinsics for all images in a folder.
        
        Args:
            input_folder: Path to folder containing images
            output_path: Optional path to save intrinsics results
            domain: Domain type ('indoor', 'outdoor', 'object')
            image_extensions: List of image extensions to process
            
        Returns:
            List of intrinsics dictionaries for each image
        """
        if image_extensions is None:
            image_extensions = ['.jpg', '.jpeg', '.png', '.PNG', '.JPG', '.JPEG']
        
        # Find all images
        image_files = []
        for ext in image_extensions:
            image_files.extend([f for f in os.listdir(input_folder) if f.endswith(ext)])
        image_files = sorted(image_files)
        
        if len(image_files) == 0:
            logger.warning(f"No images found in {input_folder}")
            return []
            
        logger.info(f"Found {len(image_files)} images, estimating intrinsics...")
        
        results = []
        
        for img_file in tqdm(image_files, desc="Estimating intrinsics"):
            img_path = os.path.join(input_folder, img_file)
            intrinsics, K = self.estimate_intrinsics(img_path, domain=domain)
            results.append({
                'filename': img_file,
                'intrinsics': intrinsics,
                'K_matrix': K.tolist(),
            })
        
        # Save results if output path provided
        if output_path is not None:
            import json
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Results saved to {output_path}")
            
        return results


def estimate_and_update_config(
    config_path: str,
    input_folder: str,
    pretrained_model_path: str = "juneyoung9/DM-Calib",
    device: str = "cuda:0",
    domain: str = "object",
    output_config_path: Optional[str] = None,
) -> dict:
    """
    Estimate camera intrinsics using DM-Calib and update the configuration.
    
    Args:
        config_path: Path to current YAML config file
        input_folder: Path to folder containing images
        pretrained_model_path: Path to DM-Calib pretrained model
        device: Device to run inference on
        domain: Domain type
        output_config_path: Path to save updated config
        
    Returns:
        Updated configuration dictionary
    """
    import yaml
    from omegaconf import OmegaConf, DictConfig
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize estimator
    estimator = DMCalibEstimator(
        pretrained_model_path=pretrained_model_path,
        device=device,
    )
    
    # Get first image from folder
    image_files = [f for f in os.listdir(input_folder) 
                  if f.endswith(('.jpg', '.jpeg', '.png', '.PNG'))]
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {input_folder}")
    
    # Use first image for intrinsic estimation
    first_image = os.path.join(input_folder, image_files[0])
    intrinsics, K = estimator.estimate_intrinsics(first_image, domain=domain)
    
    # Update config
    if 'cam' not in config:
        config['cam'] = {}
    
    config['cam']['fx'] = intrinsics['fx']
    config['cam']['fy'] = intrinsics['fy']
    config['cam']['cx'] = intrinsics['cx']
    config['cam']['cy'] = intrinsics['cy']
    
    # Save updated config
    if output_config_path is None:
        output_config_path = config_path
    
    with open(output_config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    logger.info(f"Updated config saved to {output_config_path}")
    logger.info(f"New intrinsics: fx={intrinsics['fx']:.2f}, fy={intrinsics['fy']:.2f}, "
                f"cx={intrinsics['cx']:.2f}, cy={intrinsics['cy']:.2f}")
    
    return config


if __name__ == "__main__":
    # Test the estimator
    import argparse
    
    parser = argparse.ArgumentParser(description="DM-Calib Intrinsic Estimation")
    parser.add_argument("--input_dir", type=str, required=True, help="Input image directory")
    parser.add_argument("--output", type=str, default=None, help="Output JSON file")
    parser.add_argument("--model_path", type=str, default="juneyoung9/DM-Calib", help="Model path")
    parser.add_argument("--domain", type=str, default="object", choices=["indoor", "outdoor", "object"])
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    
    estimator = DMCalibEstimator(
        pretrained_model_path=args.model_path,
        device=args.device,
    )
    
    results = estimator.estimate_from_folder(
        args.input_dir,
        output_path=args.output,
        domain=args.domain,
    )
    
    print(f"Estimated intrinsics for {len(results)} images")
