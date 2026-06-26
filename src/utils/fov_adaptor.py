"""
FoV Adaptation Module for EDroidPGSR

This module implements a lightweight Field-of-View (FoV) adaptation strategy
guided by multi-view motion cues. It uses optical flow and camera translation
to compute a scale indicator (rho) that distinguishes between near-field and
far-field configurations, then adaptively expands the angular range of the
camera image representation for DM-Calib intrinsic estimation.

Reference:
    Based on the FoV adaptation strategy described in the user's method:
    - rho measures pixel displacement per unit camera motion (see _compute_rho).
    - rho_norm clips rho to [rho_min, rho_max] range
    - theta_range = theta_base + alpha * rho_norm
    - phi_range = phi_base + beta * rho_norm
    - phi_max = 0.5 + phi_range/2, phi_min = 0.5 - phi_range/2 (phi in [0,1])
    - theta_max = 0 + theta_range/2, theta_min = 0 - theta_range/2 (theta in [-1,1])
"""

import logging
import os
import re
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from lietorch import SE3

logger = logging.getLogger(__name__)


class FoVAdaptor:
    """
    FoV Adaptation module that uses optical flow and camera translation
    to compute a scale indicator (rho) for DM-Calib intrinsic estimation.

    This class is designed to be integrated into the SLAM pipeline after
    the warmup phase completes. It computes rho from flow/translation data
    stored in the DepthVideo, then re-runs DM-Calib with adapted FoV.
    """

    def __init__(
        self,
        video,
        cfg,
        device: str = "cuda:0",
        rho_min: float = 10.0,
        rho_max: float = 200.0,
        alpha: float = 0.3,
        beta: float = 0.3,
        rho_base: float = 50.0,
        min_keyframes: int = 4,
        evaluation_root: Optional[str] = None,
        flow_vis_max_pairs: Optional[int] = None,
        rho_use_deepest_patch: bool = True,
        rho_patch_size: int = 7,
        rho_patch_min_valid_frac: float = 0.5,
    ):
        """
        Initialize the FoVAdaptor.

        Args:
            video: DepthVideo instance containing flow_norm_sum, flow_pixel_count,
                   translation_norm, and intrinsics
            cfg: Configuration object with DM-Calib parameters
            device: Device to run inference on
            rho_min: Minimum rho value for normalization (far-field baseline)
            rho_max: Maximum rho value for normalization (near-field ceiling)
            alpha: Scaling factor for theta range adaptation
            beta: Scaling factor for phi range adaptation
            rho_base: Base rho value to fall back to if computation fails
            min_keyframes: Below this many valid pairs we log a warning but still
                compute rho when at least one pair is available
            rho_use_deepest_patch: If True, mean flow is taken on the ps×ps region with
                largest mean metric depth (farthest patch), using sensor disparity
                disps_sens (= 1/Z) on the reference keyframe; rho = mean_i(flow_i/t_i).
            rho_patch_size: Sliding window side length on the flow/disps_sens grid.
            rho_patch_min_valid_frac: Fraction of valid depth/static pixels required
                inside a patch to compete for "deepest".
        """
        self.video = video
        self.cfg = cfg
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device("cpu")

        self.rho_use_deepest_patch = rho_use_deepest_patch
        self.rho_patch_size = int(rho_patch_size)
        self.rho_patch_min_valid_frac = float(rho_patch_min_valid_frac)

        self.rho_min = rho_min
        self.rho_max = rho_max
        self.alpha = alpha
        self.beta = beta
        self.rho_base = rho_base
        self.min_keyframes = min_keyframes
        self.flow_vis_max_pairs = flow_vis_max_pairs if flow_vis_max_pairs is not None else min_keyframes
        self.flow_vis_step = int(cfg.get("fov_flow_vis_step", 8))

        self.evaluation_root = evaluation_root
        if self.evaluation_root is not None:
            self.flow_arrow_out_dir = os.path.join(self.evaluation_root, "fov_adaptation", "flow_arrows")
            self.flow_rgb_out_dir = os.path.join(self.evaluation_root, "fov_adaptation", "flow_rgb")
            self.dmcalib_camera_out_dir = os.path.join(self.evaluation_root, "dmcalib", "fov_adaptation", "camera_images")
            os.makedirs(self.flow_arrow_out_dir, exist_ok=True)
            os.makedirs(self.flow_rgb_out_dir, exist_ok=True)
            os.makedirs(self.dmcalib_camera_out_dir, exist_ok=True)

        # DM-Calib parameters from config
        self.dm_calib_model_path = cfg.get("dm_calib_model_path", "./pretrained/DM-Calib")
        self.dm_calib_domain = cfg.get("dm_calib_domain", "object")
        self.dm_calib_processing_res = cfg.get("dm_calib_processing_res", 768)
        self.dm_calib_denoise_steps = cfg.get("dm_calib_denoise_steps", 20)
        self.dm_calib_ensemble_size = cfg.get("dm_calib_ensemble_size", 3)
        self.seed = cfg.get("dm_calib_seed", 666)

        # State
        self.estimator = None
        self.adaptation_done = False
        self.rho_computed = None
        self.rho_norm_computed = None
        self.theta_range_adapted = None
        self.phi_range_adapted = None

        # Base angular ranges (from dm_calib.py lines 76-79)
        self.theta_base = 0.35 * 2  # theta_max - theta_min = 0.59 - (-0.59) = 1.18
        self.phi_base = 0.6  # phi_max - phi_min = 1.0 - 0.0 = 1.0

        logger.info(
            f"[FoVAdaptor] Initialized with rho_min={rho_min}, rho_max={rho_max}, "
            f"alpha={alpha}, beta={beta}, rho_base={rho_base}, min_keyframes={min_keyframes}, "
            f"deepest_patch={rho_use_deepest_patch}, patch={rho_patch_size}"
        )

    @staticmethod
    def _mean_flow_in_deepest_metric_depth_patch(
        flow_hw2: np.ndarray,
        disp_sens_hw: np.ndarray,
        valid_static_hw,
        patch_size: int,
        disp_eps: float = 1e-6,
        min_valid_frac: float = 0.5,
    ):
        """
        Pick a patch with maximum mean metric depth Z (farthest region).

        ``disp_sens`` matches DepthVideo.disps_sens: stored as 1/Z for Z>0.

        Returns:
            Mean ||flow|| over that patch, or None if no usable patch/grid.
        """
        if flow_hw2.ndim != 3 or flow_hw2.shape[-1] != 2:
            return None
        H, W, _ = flow_hw2.shape
        if disp_sens_hw.shape != (H, W):
            return None

        valid_d = (disp_sens_hw > disp_eps) & np.isfinite(disp_sens_hw)
        if valid_static_hw is not None and valid_static_hw.shape == (H, W):
            valid = valid_d & valid_static_hw.astype(bool)
        else:
            valid = valid_d

        mag = np.sqrt(flow_hw2[..., 0] ** 2 + flow_hw2[..., 1] ** 2).astype(np.float64)
        ps = int(patch_size)
        if ps < 1:
            return None

        if H < ps or W < ps:
            if valid.sum() < 1:
                return None
            return float(mag[valid].mean())

        min_valid = max(1, int(np.ceil(min_valid_frac * ps * ps)))
        best_mean_z = -1.0
        best_yx = (0, 0)
        found = False

        for y in range(H - ps + 1):
            for x in range(W - ps + 1):
                vm = valid[y : y + ps, x : x + ps]
                if int(vm.sum()) < min_valid:
                    continue
                dwin = disp_sens_hw[y : y + ps, x : x + ps]
                zvals = (1.0 / dwin[vm]).astype(np.float64)
                mean_z = float(zvals.mean())
                if mean_z > best_mean_z:
                    best_mean_z = mean_z
                    best_yx = (y, x)
                    found = True

        if not found:
            return None

        y, x = best_yx
        return float(mag[y : y + ps, x : x + ps].mean())

    def _mean_flow_for_pair_index(self, i: int, fns: float, fpc: float) -> float:
        """Mean flow for keyframe pair i: either deepest patch or full-grid fallback."""
        if not self.rho_use_deepest_patch or self.rho_patch_size < 1:
            return fns / max(fpc, 1e-6)

        prev_idx = i - 1
        if prev_idx < 0:
            return fns / max(fpc, 1e-6)

        flow_t = self.video.flow_2d[i].float().cpu()
        flow_hw2 = flow_t.permute(1, 2, 0).numpy()
        disp_hw = self.video.disps_sens[prev_idx].float().cpu().numpy()

        s = int(self.video.scale_factor)
        start = int(s // 2 - 1)
        sm = self.video.static_masks[prev_idx][start::s, start::s].cpu().numpy()

        patch_mean = self._mean_flow_in_deepest_metric_depth_patch(
            flow_hw2,
            disp_hw,
            sm,
            self.rho_patch_size,
            min_valid_frac=self.rho_patch_min_valid_frac,
        )
        if patch_mean is None:
            return fns / max(fpc, 1e-6)
        return patch_mean

    def _compute_rho(self) -> Tuple[float, float, int, List[int]]:
        """
        Compute the scale indicator rho from stored flow/translation data.

        Legacy mode (rho_use_deepest_patch=False):
            rho = (sum_i sum_p ||f_{i,p}|| / sum_i N_i) / mean_i(||t_i||)
            i.e. one global mean flow divided by mean translation (not identical to
            mean_i(E[||f||]_i / ||t_i||) when ||t_i|| varies).

        Deepest-patch mode (default):
            On the flow/disps_sens grid, take the ps x ps window with largest mean
            metric depth on the *reference* keyframe (i-1), then
            rho = mean_i ( mean(||f||)_patch_i / ||t_i|| ).
        """
        with self.video.get_lock():
            counter = self.video.counter.value

        if counter < 2:
            logger.warning("[FoVAdaptor] Not enough frames for rho computation")
            return self.rho_base, 0.5, 0, []

        flow_sums = []
        flow_counts = []
        trans_norms = []
        valid_indices: List[int] = []

        # Debug: log first few frames' data
        debug_frames = min(5, counter)
        debug_info = []
        for i in range(debug_frames):
            fns = self.video.flow_norm_sum[i].item()
            fpc = self.video.flow_pixel_count[i].item()
            tn = self.video.translation_norm[i].item()
            debug_info.append(f"frame[{i}]: fpc={fpc}, fns={fns:.2f}, tn={tn:.6f}")

        for i in range(counter):
            fns = self.video.flow_norm_sum[i].item()
            fpc = self.video.flow_pixel_count[i].item()
            tn = self.video.translation_norm[i].item()

            # Fallback: compute translation from poses if translation_norm is zero
            if tn < 1e-6 and i >= 1:
                with self.video.get_lock():
                    pose_curr = SE3(self.video.poses[i])
                    pose_prev = SE3(self.video.poses[i - 1])
                dP = pose_curr * pose_prev.inv()
                tn = dP.translation().norm().item()

            # Need both flow data and valid translation
            if fpc > 0 and fns > 0 and tn > 1e-6:
                flow_sums.append(fns)
                flow_counts.append(fpc)
                trans_norms.append(tn)
                valid_indices.append(i)

        if len(flow_sums) < 1:
            logger.warning(
                f"[FoVAdaptor] No valid frame pairs (counter={counter}). "
                f"First frames data: {'; '.join(debug_info)}"
            )
            return self.rho_base, 0.5, 0, valid_indices

        if len(flow_sums) < self.min_keyframes:
            logger.warning(
                f"[FoVAdaptor] Only {len(flow_sums)} valid frame pairs "
                f"(recommended >= {self.min_keyframes}); computing rho from available pairs."
            )

        n_pairs = len(flow_sums)
        mean_trans = sum(trans_norms) / n_pairs

        if self.rho_use_deepest_patch and self.rho_patch_size >= 1:
            per_pair_flow = []
            rho_terms = []
            for k, i in enumerate(valid_indices):
                mf = self._mean_flow_for_pair_index(i, flow_sums[k], flow_counts[k])
                per_pair_flow.append(mf)
                rho_terms.append(mf / trans_norms[k])
            mean_flow = float(sum(per_pair_flow) / n_pairs)
            rho = float(sum(rho_terms) / n_pairs)
            mode = f"deepest_{self.rho_patch_size}x{self.rho_patch_size}_patch"
        else:
            mean_flow = sum(flow_sums) / sum(flow_counts)
            rho = mean_flow / mean_trans
            mode = "global_mean_flow"

        # Normalize to [0, 1]
        rho_norm = np.clip((rho - self.rho_min) / (self.rho_max - self.rho_min), 0.0, 1.0)

        logger.info(
            f"[FoVAdaptor] rho = {rho:.2f} ({mode}: mean_flow={mean_flow:.2f}, "
            f"mean_trans={mean_trans:.4f}), rho_norm={rho_norm:.4f}, valid_pairs={n_pairs}"
        )

        return float(rho), float(rho_norm), n_pairs, valid_indices

    @staticmethod
    def _extract_numeric_id(path: str) -> Optional[Tuple[int, int]]:
        """
        Extract the first continuous digit group from filename stem.
        Returns (numeric_value, digit_count_for_padding).
        """
        stem = os.path.splitext(os.path.basename(path))[0]
        m = re.search(r"(\d+)", stem)
        if not m:
            return None
        digits = m.group(1)
        return int(digits), len(digits)

    def _maybe_save_dmcalib_camera_image(self, camera_img: torch.Tensor, image_path: str) -> None:
        if self.evaluation_root is None:
            return
        parsed = self._extract_numeric_id(image_path)
        if parsed is None:
            return
        frame_id, pad_width = parsed
        # camera_img: [3,H,W] in [0,1]
        cam = (
            camera_img.detach()
            .to(torch.float32)
            .clamp(0.0, 1.0)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        cam_u8 = (cam * 255.0).astype(np.uint8)
        out_path = os.path.join(self.dmcalib_camera_out_dir, f"camera_img_{str(frame_id).zfill(pad_width)}.png")
        Image.fromarray(cam_u8).save(out_path)

    def _save_fov_flow_visualizations(self, valid_indices: List[int], first_image_path: str) -> None:
        if self.evaluation_root is None:
            return
        if not valid_indices:
            return
        parsed = self._extract_numeric_id(first_image_path)
        if parsed is None:
            return
        first_id, pad_width = parsed

        # Align dataset file numbering with stream timestamps.
        # We infer an offset so that video.timestamp[0] corresponds to the filename id.
        try:
            t0 = int(round(float(self.video.timestamp[0].item())))
            offset = first_id - t0
        except Exception:
            offset = 0

        flow_h = int(self.video.flow_2d.shape[-2])
        flow_w = int(self.video.flow_2d.shape[-1])

        # Save only a limited number of pairs to avoid flooding disk.
        valid_indices = sorted(valid_indices)[: self.flow_vis_max_pairs]

        # Lazy import to keep module import light.
        from torchvision.utils import flow_to_image as tv_flow_to_image

        for i in valid_indices:
            prev_idx = i - 1
            if prev_idx < 0:
                continue

            prev_ts = int(round(float(self.video.timestamp[prev_idx].item())))
            curr_ts = int(round(float(self.video.timestamp[i].item())))
            prev_id = prev_ts + offset
            curr_id = curr_ts + offset

            prev_id_str = str(prev_id).zfill(pad_width)
            curr_id_str = str(curr_id).zfill(pad_width)

            # Background RGB for arrow plot: resize frame (prev) to flow resolution.
            bg_rgb = self.video.images[prev_idx].permute(1, 2, 0).cpu().numpy()  # [H_out, W_out, 3]
            bg_rgb = cv2.resize(bg_rgb, (flow_w, flow_h), interpolation=cv2.INTER_LINEAR)
            vis_bgr = cv2.cvtColor(bg_rgb, cv2.COLOR_RGB2BGR)

            flow = self.video.flow_2d[i].to(torch.float32).cpu().numpy()  # [2,H,W]
            u = flow[0]
            v = flow[1]
            mag = np.sqrt(u * u + v * v)
            mag_max = float(np.percentile(mag, 95)) + 1e-6

            step = max(1, self.flow_vis_step)
            ys = np.arange(0, flow_h, step)
            xs = np.arange(0, flow_w, step)
            for y in ys:
                for x in xs:
                    fx = float(u[y, x])
                    fy = float(v[y, x])
                    x2 = int(np.clip(x + fx, 0, flow_w - 1))
                    y2 = int(np.clip(y + fy, 0, flow_h - 1))
                    m = np.sqrt(fx * fx + fy * fy)
                    intensity = int(255.0 * np.clip(m / mag_max, 0.0, 1.0))
                    color = (0, intensity, 0)  # Green lines
                    # Skip tiny lines
                    if intensity < 8:
                        continue
                    cv2.line(vis_bgr, (x, y), (x2, y2), color, 1)

            line_out = os.path.join(
                self.flow_arrow_out_dir,
                f"flow_{prev_id_str}_to_{curr_id_str}_line.png",
            )
            cv2.imwrite(line_out, vis_bgr)

            # torchvision flow_to_image: expects [B,2,H,W] float tensor
            flow_for_tv = self.video.flow_2d[i].to(torch.float32).unsqueeze(0).cpu()
            flow_rgb_u8 = tv_flow_to_image(flow_for_tv)[0].permute(1, 2, 0).numpy()  # [H,W,3], uint8
            flow_rgb_out = os.path.join(
                self.flow_rgb_out_dir,
                f"flow_{prev_id_str}_to_{curr_id_str}_flow_to_image.png",
            )
            cv2.imwrite(flow_rgb_out, cv2.cvtColor(flow_rgb_u8, cv2.COLOR_RGB2BGR))

    def _compute_adapted_angular_ranges(self, rho_norm: float) -> Tuple[float, float, float, float]:
        """
        Compute the adapted angular ranges based on rho_norm.

        Args:
            rho_norm: Normalized rho value in [0, 1]

        Returns:
            Tuple of (phi_min, phi_max, theta_min, theta_max)
        """
        # Adapt ranges
        theta_range = self.theta_base + self.alpha * rho_norm
        phi_range = self.phi_base + self.beta * rho_norm

        # phi in [0.0, 1.0]: phi_max = 0.5 + phi_range/2, phi_min = 0.5 - phi_range/2
        phi_max = 0.5 + phi_range / 2.0
        phi_min = 0.5 - phi_range / 2.0

        # Clamp phi to [0, 1]
        phi_min = float(np.clip(phi_min, 0.0, 1.0))
        phi_max = float(np.clip(phi_max, 0.0, 1.0))

        # theta in [-1.0, 1.0]: theta_max = 0 + theta_range/2, theta_min = 0 - theta_range/2
        theta_max = 0.0 + theta_range / 2.0
        theta_min = 0.0 - theta_range / 2.0

        # Clamp theta to [-1, 1]
        theta_min = float(np.clip(theta_min, -1.0, 1.0))
        theta_max = float(np.clip(theta_max, -1.0, 1.0))

        logger.info(
            f"[FoVAdaptor] Adapted ranges: theta=[{theta_min:.4f}, {theta_max:.4f}] "
            f"(range={theta_range:.4f}), phi=[{phi_min:.4f}, {phi_max:.4f}] (range={phi_range:.4f})"
        )

        self.theta_range_adapted = theta_range
        self.phi_range_adapted = phi_range

        return phi_min, phi_max, theta_min, theta_max

    def _ensure_estimator_loaded(self):
        """Lazy load the DM-Calib estimator."""
        if self.estimator is not None:
            return

        try:
            from .dm_calib import DMCalibEstimator
        except ImportError as e:
            logger.error(f"[FoVAdaptor] Failed to import DMCalibEstimator: {e}")
            raise

        self.estimator = DMCalibEstimator(
            pretrained_model_path=self.dm_calib_model_path,
            device=str(self.device),
            processing_res=self.dm_calib_processing_res,
            denoise_steps=self.dm_calib_denoise_steps,
            ensemble_size=self.dm_calib_ensemble_size,
            seed=self.seed,
        )
        logger.info("[FoVAdaptor] DM-Calib estimator loaded")

    def estimate_intrinsics_with_adapted_fov(
        self,
        image_path: str,
        phi_min: float,
        phi_max: float,
        theta_min: float,
        theta_max: float,
    ) -> Tuple[dict, np.ndarray]:
        """
        Estimate camera intrinsics using DM-Calib with adapted FoV angular ranges.

        This uses the same DM-Calib pipeline as dm_calib.py but with modified
        angular ranges passed to calculate_intrinsic().

        Args:
            image_path: Path to input RGB image
            phi_min: Minimum phi value (in normalized angular coords)
            phi_max: Maximum phi value (in normalized angular coords)
            theta_min: Minimum theta value (in normalized angular coords)
            theta_max: Maximum theta value (in normalized angular coords)

        Returns:
            Tuple of (intrinsics_dict, K_matrix)
            - intrinsics_dict: {'fx', 'fy', 'cx', 'cy'}
            - K_matrix: 3x3 camera intrinsic matrix
        """
        self._ensure_estimator_loaded()

        from .dm_calib import preprocess_pad, calculate_intrinsic

        # Read and preprocess image
        input_image = Image.open(image_path).convert('RGB')
        w_ori, h_ori = input_image.size

        max_dim = max(w_ori, h_ori)
        if max_dim > self.dm_calib_processing_res:
            scale = self.dm_calib_processing_res / max_dim
            new_w = int(w_ori * scale)
            new_h = int(h_ori * scale)
            input_image = input_image.resize((new_w, new_h), Image.LANCZOS)

        totensor = torchvision.transforms.ToTensor()
        img = totensor(input_image)
        c, h, w = img.shape

        # Preprocess with padding
        img_pad, pad_left, pad_right, pad_top, pad_bottom = preprocess_pad(
            img, (self.dm_calib_processing_res, self.dm_calib_processing_res)
        )

        # Run DM-Calib inference: call the estimator's pipeline directly
        generator = torch.Generator(device=self.device).manual_seed(self.seed)
        camera_img = self.estimator._pipeline(
            image=img_pad.repeat(self.dm_calib_ensemble_size, 1, 1, 1),
            num_inference_steps=self.dm_calib_denoise_steps,
            guidance_scale=1,
            generator=generator,
        ).images

        # Average ensemble predictions
        camera_img = torch.stack(
            [totensor(camera_img[i]) for i in range(self.dm_calib_ensemble_size)]
        ).mean(0, keepdim=True)

        # Save intermediate "camera image" from DM-Calib for visualization/debugging.
        self._maybe_save_dmcalib_camera_image(camera_img[0], image_path)

        # Calculate intrinsics with adapted angular ranges
        intrin_pred = calculate_intrinsic(
            camera_img[0],
            (pad_left, pad_right, pad_top, pad_bottom),
            mask=None,
            phi_min=phi_min,
            phi_max=phi_max,
            theta_min=theta_min,
            theta_max=theta_max,
        )

        # Build K matrix
        K = np.eye(3)
        K[0, 0] = intrin_pred[0]
        K[1, 1] = intrin_pred[1]
        K[0, 2] = intrin_pred[2]
        K[1, 2] = intrin_pred[3]

        logger.info(f"[FoVAdaptor] Intrinsics (resized): fx={K[0,0]:.2f}, fy={K[1,1]:.2f}, cx={K[0,2]:.2f}, cy={K[1,2]:.2f}")

        # Scale K back to original resolution
        scale_x = w_ori / w
        scale_y = h_ori / h
        K_ori = K.copy()
        K_ori[0, 0] *= scale_x
        K_ori[1, 1] *= scale_y
        K_ori[0, 2] *= scale_x
        K_ori[1, 2] *= scale_y

        intrinsics_dict = {
            'fx': float(K_ori[0, 0]),
            'fy': float(K_ori[1, 1]),
            'cx': float(K_ori[0, 2]),
            'cy': float(K_ori[1, 2]),
        }

        logger.info(f"[FoVAdaptor] Intrinsics (original): fx={K_ori[0,0]:.2f}, fy={K_ori[1,1]:.2f}, "
                    f"cx={K_ori[0,2]:.2f}, cy={K_ori[1,2]:.2f}")

        return intrinsics_dict, K_ori

    def adapt_and_update_intrinsics(
        self,
        image_path: str,
    ) -> Tuple[dict, np.ndarray, float, float]:
        """
        Main entry point: compute rho, adapt FoV, run DM-Calib, and return intrinsics.

        Args:
            image_path: Path to input RGB image for DM-Calib inference

        Returns:
            Tuple of (intrinsics_dict, K_matrix, rho, rho_norm)
        """
        if self.adaptation_done:
            logger.info("[FoVAdaptor] Adaptation already completed, skipping")
            return None, None, self.rho_computed, self.rho_norm_computed

        # Step 1: Compute rho from flow/translation data
        rho, rho_norm, valid_count, valid_indices = self._compute_rho()
        self.rho_computed = rho
        self.rho_norm_computed = rho_norm

        if valid_count < 1:
            logger.warning("[FoVAdaptor] Cannot compute rho, returning None")
            return None, None, rho, rho_norm

        # Save optical flow visualizations used for rho computation.
        # This happens before DM-Calib so you can inspect the motion cues.
        self._save_fov_flow_visualizations(valid_indices, image_path)

        # Step 2: Compute adapted angular ranges
        phi_min, phi_max, theta_min, theta_max = self._compute_adapted_angular_ranges(rho_norm)

        # Step 3: Run DM-Calib with adapted ranges
        intrinsics, K = self.estimate_intrinsics_with_adapted_fov(
            image_path,
            phi_min, phi_max, theta_min, theta_max,
        )

        # Mark as done
        self.adaptation_done = True

        # Clean up GPU memory
        self.cleanup()

        return intrinsics, K, rho, rho_norm

    def cleanup(self):
        """Release DM-Calib model from GPU memory."""
        if self.estimator is not None:
            try:
                self.estimator.cleanup()
                logger.info("[FoVAdaptor] DM-Calib estimator cleaned up")
            except Exception as e:
                logger.warning(f"[FoVAdaptor] Cleanup warning: {e}")
            self.estimator = None
