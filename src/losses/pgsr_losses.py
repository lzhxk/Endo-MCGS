#
# PGSR-style loss functions for EDroidPGSR
# Based on PGSR's loss implementation
#

import torch
import torch.nn.functional as F
from .image import l1_loss, ssim
from .misc import edge_weighted_tv

# Try to import PGSR utilities, fall back to basic implementations if not available
try:
    from ..utils.loss_utils import lncc, get_img_grad_weight
    from ..utils.graphics_utils import patch_offsets, patch_warp
    PGSR_UTILS_AVAILABLE = True
except ImportError:
    print("Warning: PGSR utilities not available, using basic implementations")
    PGSR_UTILS_AVAILABLE = False
    
    # Basic fallback implementations
    def lncc(x, y):
        return torch.tensor(0.0, device=x.device)
    
    def get_img_grad_weight(img):
        return torch.ones_like(img[0:1])
    
    def patch_offsets(size, device):
        return torch.zeros((2*size+1)**2, 2, device=device)
    
    def patch_warp(H, pixels):
        return pixels

def pgsr_mapping_loss(
    image: torch.Tensor,
    depth: torch.Tensor,
    cam,
    rendered_normal=None,
    depth_normal=None,
    plane_depth=None,
    rendered_distance=None,
    iteration=0,
    loss_params=None,
    gaussians=None,
    visibility_filter=None,
    **kwargs
):
    """
    PGSR-style mapping loss combining RGB, depth, and geometric consistency.
    """
    if loss_params is None:
        loss_params = {
            'alpha1': 0.7,  # RGB weight
            'alpha2': 0.2,  # SSIM weight
            'beta1': 2.0,   # Isotropic scale regularizer
            'beta2': 0.0001, # Edge-aware smoothness
            'single_view_weight': 0.1,
            'single_view_weight_from_iter': 1000,
            'multi_view_weight': 0.1,
            'multi_view_weight_from_iter': 2000,
            'scale_loss_weight': 0.01
        }
    
    # Get ground truth image
    gt_image = cam.original_image / 255.0  # Convert to [0,1] range
    
    # Ensure both images are in CHW format
    if image.dim() == 3:
        if image.shape[0] == 3:  # Already CHW
            pass
        elif image.shape[2] == 3:  # HWC format
            image = image.permute(2, 0, 1)
        elif image.shape[1] == 3:  # WCH format
            image = image.permute(1, 2, 0)  # WCH -> CHW
        else:
            raise ValueError(f"Unexpected image shape: {image.shape}")
    
    if gt_image.dim() == 3:
        if gt_image.shape[0] == 3:  # Already CHW
            pass
        elif gt_image.shape[2] == 3:  # HWC format
            gt_image = gt_image.permute(2, 0, 1)
        elif gt_image.shape[1] == 3:  # WCH format
            gt_image = gt_image.permute(1, 2, 0)  # WCH -> CHW
        else:
            raise ValueError(f"Unexpected gt_image shape: {gt_image.shape}")
    
    # Add batch dimension for SSIM (expects 4D tensors: NCHW)
    image_batch = image.unsqueeze(0)  # CHW -> NCHW
    gt_image_batch = gt_image.unsqueeze(0)  # CHW -> NCHW
    
    # Basic RGB loss
    ssim_loss = (1.0 - ssim(image_batch, gt_image_batch))
    l1_loss_val = l1_loss(image, gt_image)
    # image_loss = (1.0 - loss_params['alpha2']) * l1_loss_val + loss_params['alpha2'] * ssim_loss
    image_loss = 2.0 * l1_loss_val + ssim_loss


    # Depth loss (similar structure to RGB loss; use L1 on valid depths)
    depth_term = None
    smooth_term = None
    try:
        supervise_with_prior = bool(loss_params.get('supervise_with_prior', False))
        gt_depth_tensor = None
        if supervise_with_prior and hasattr(cam, 'depth_prior') and cam.depth_prior is not None:
            gt_depth_tensor = cam.depth_prior
        elif hasattr(cam, 'depth') and cam.depth is not None:
            gt_depth_tensor = cam.depth

        if gt_depth_tensor is not None and depth is not None:
            # Ensure CHW with 1 channel
            pred_depth_in = depth
            if pred_depth_in.dim() == 2:
                pred_depth_in = pred_depth_in.unsqueeze(0)
            if pred_depth_in.dim() == 3 and pred_depth_in.shape[0] != 1:
                # If HxWx1 format
                if pred_depth_in.shape[-1] == 1:
                    pred_depth_in = pred_depth_in.permute(2, 0, 1)
            
            gt_depth_in = gt_depth_tensor
            if gt_depth_in.dim() == 2:
                gt_depth_in = gt_depth_in.unsqueeze(0)
            if gt_depth_in.dim() == 3 and gt_depth_in.shape[0] != 1:
                if gt_depth_in.shape[-1] == 1:
                    gt_depth_in = gt_depth_in.permute(2, 0, 1)

            # Align shapes if needed (resize pred to gt size)
            if pred_depth_in.shape[-2:] != gt_depth_in.shape[-2:]:
                pred_depth_resized = F.interpolate(pred_depth_in.unsqueeze(0), size=gt_depth_in.shape[-2:], mode='nearest').squeeze(0)
            else:
                pred_depth_resized = pred_depth_in

            # Valid mask: positive and finite groundtruth
            valid_gt = torch.isfinite(gt_depth_in) & (gt_depth_in > 0)
            valid_pred = torch.isfinite(pred_depth_resized)
            valid = valid_gt & valid_pred
            if valid.any():
                depth_l1 = F.l1_loss(pred_depth_resized[valid], gt_depth_in[valid])
                depth_term = depth_l1
                
                # Edge-weighted smoothness on depth (uses image edges and valid mask)
                try:
                    beta_smooth = float(loss_params.get('beta2', 0.0001))
                    # Ensure ref image is CHW and matches spatial size
                    ref_img = gt_image
                    if ref_img.dim() == 3 and ref_img.shape[-2:] != pred_depth_resized.shape[-2:]:
                        ref_img = F.interpolate(ref_img.unsqueeze(0), size=pred_depth_resized.shape[-2:], mode='bilinear', align_corners=True).squeeze(0)
                    smooth_term = beta_smooth * edge_weighted_tv(pred_depth_resized, ref_img, mask=valid)
                except Exception:
                    smooth_term = None
    except Exception:
        depth_term = None

    # Combine RGB and depth with alpha1 (rgb portion)
    if depth_term is not None:
        alpha1 = float(loss_params.get('alpha1', 0.7))
        total_loss = alpha1 * image_loss + (1.0 - alpha1) * depth_term
    else:
        total_loss = image_loss.clone()
    
    # Add smoothness term if computed
    if smooth_term is not None and torch.isfinite(smooth_term):
        total_loss += smooth_term
    
    # Scale loss (PGSR-style scale regularization)
    if gaussians is not None and visibility_filter is not None and visibility_filter.sum() > 0:
        scale_loss_val = scale_loss(gaussians, visibility_filter, loss_params.get('scale_loss_weight', 0.01))
        total_loss += scale_loss_val
    
    # Single-view geometric loss
    # print(f"iteration single_view_weight",iteration)
    if (iteration > loss_params['single_view_weight_from_iter'] and 
        rendered_normal is not None and depth_normal is not None):

        # print(f"single_view_weight; rendered_normal: {rendered_normal.shape}, depth_normal: {depth_normal.shape}")
        
        weight = loss_params['single_view_weight']
        image_weight = (1.0 - get_img_grad_weight(gt_image))
        image_weight = (image_weight).clamp(0,1).detach() ** 2
        
        normal_loss = weight * (image_weight * (((depth_normal - rendered_normal)).abs().sum(0))).mean()
        total_loss += normal_loss
    
    # Multi-view consistency terms are added externally via multi_view_consistency_loss
    return total_loss

def scale_loss(gaussians, visibility_filter, scale_loss_weight=0.01):
    """
    PGSR-style scale loss that penalizes small Gaussian scales.
    
    Args:
        gaussians: GaussianModel instance
        visibility_filter: Boolean tensor indicating visible Gaussians
        scale_loss_weight: Weight for the scale loss
    
    Returns:
        Scale loss tensor
    """
    if visibility_filter.sum() > 0:
        scale = gaussians.get_scaling[visibility_filter]
        sorted_scale, _ = torch.sort(scale, dim=-1)
        min_scale_loss = sorted_scale[..., 0]  # Get the minimum scale for each Gaussian
        return scale_loss_weight * min_scale_loss.mean()
    else:
        return torch.tensor(0.0, device=gaussians.get_scaling.device)

def multi_view_consistency_loss(
    viewpoint_cam,
    nearest_cam,
    gaussians,
    render_pkg,
    pipe,
    bg_color,
    loss_params,
    iteration
):
    """
    Multi-view geometric and photometric consistency loss.
    Implements a practical subset of PGSR's multi-view terms using plane-induced homographies.
    """
    if nearest_cam is None:
        return torch.tensor(0.0, device=bg_color.device if isinstance(bg_color, torch.Tensor) else 'cuda')

    # Lazy import to avoid circular deps
    try:
        from ..gaussian_splatting.pgsr_renderer import render as pgsr_render
    except Exception:
        # If PGSR renderer is not available, skip multi-view term to avoid instability
        return torch.tensor(0.0, device=bg_color.device if isinstance(bg_color, torch.Tensor) else 'cuda')

    # Render nearest view with plane info
    with torch.no_grad():
        nb_pkg = pgsr_render(nearest_cam, gaussians, pipe, bg_color, return_plane=True, return_depth_normal=True)
    if nb_pkg is None:
        return torch.tensor(0.0, device=bg_color.device if isinstance(bg_color, torch.Tensor) else 'cuda')

    # Prepare inputs
    img_v = render_pkg["render"]          # CHW
    img_n = nb_pkg["render"]               # CHW
    plane_depth_v = render_pkg.get("plane_depth")  # 1HW
    normal_v = render_pkg.get("rendered_normal")   # 3HW

    if plane_depth_v is None or normal_v is None:
        return torch.tensor(0.0, device=img_v.device)

    device = img_v.device

    # Camera intrinsics/extrinsics
    K_v, E_v = viewpoint_cam.get_calib_matrix_nerf(scale=1)
    K_n, E_n = nearest_cam.get_calib_matrix_nerf(scale=1)
    K_v, E_v, K_n, E_n = K_v.to(device), E_v.to(device), K_n.to(device), E_n.to(device)

    # Relative pose (from v to n): x_n ~ R * x_v + t
    R_v = E_v[:3, :3]
    t_v = E_v[3, :3]
    R_n = E_n[:3, :3]
    t_n = E_n[3, :3]
    R_nv = R_n @ R_v.T
    t_nv = (t_n - (R_n @ R_v.T) @ t_v)

    # Sampling grid of centers for local planar patches
    _, height, width = img_v.shape
    grid_step = max(min(height, width) // 32, 8)  # adaptive stride
    ys = torch.arange(grid_step//2, height, grid_step, device=device)
    xs = torch.arange(grid_step//2, width, grid_step, device=device)

    patch_size = 3  # 3x3 local patch
    offsets = patch_offsets(patch_size, device)  # (1, K, 2) with dx, dy
    offsets = offsets.squeeze(0)  # (K, 2)

    photometric_losses = []
    geometric_losses = []

    for y in ys:
        for x in xs:
            d = plane_depth_v[0, y, x]
            n_cam = normal_v[:, y, x]
            # Guard against invalid normals/depth
            if not torch.isfinite(d) or d <= 0 or torch.isnan(n_cam).any():
                continue
            n_cam = n_cam / (n_cam.norm() + 1e-8)

            # Plane-induced homography: H = K_n * (R_nv - t_nv * n^T / d) * K_v^{-1}
            H_nv = R_nv - torch.ger(t_nv, n_cam) / (d + 1e-8)
            H_mat = K_n @ H_nv @ torch.linalg.inv(K_v)

            # Build pixel coords for local patch around (x,y)
            Kp = offsets.shape[0]
            x_f = x.to(dtype=torch.float32)
            y_f = y.to(dtype=torch.float32)
            pts = torch.stack([
                               x_f.repeat(Kp) + offsets[:, 0].to(dtype=torch.float32),
                               y_f.repeat(Kp) + offsets[:, 1].to(dtype=torch.float32),
                               torch.ones(Kp, device=device, dtype=torch.float32)
                              ], dim=0)  # (3, K)

            # Warp to nearest view
            pts_n = H_mat @ pts
            pts_n = pts_n[:2] / (pts_n[2:3] + 1e-8)

            # Normalize to [-1,1] for grid_sample
            gx = (pts_n[0] / (width - 1) * 2 - 1).clamp(-1, 1)
            gy = (pts_n[1] / (height - 1) * 2 - 1).clamp(-1, 1)
            grid = torch.stack([gx, gy], dim=-1).view(1, -1, 1, 2)

            # Sample patches
            patch_v = torch.nn.functional.grid_sample(img_v.unsqueeze(0),
                                                       torch.stack([(x / (width - 1) * 2 - 1).expand(1), (y / (height - 1) * 2 - 1).expand(1)], dim=-1)
                                                       .view(1, 1, 1, 2),
                                                       mode='bilinear', align_corners=True)  # 1,C,1,1
            patch_v = patch_v.view(1, img_v.shape[0], 1)  # 1,C,1

            patch_n = torch.nn.functional.grid_sample(img_n.unsqueeze(0), grid, mode='bilinear', align_corners=True)  # 1,C,K,1
            patch_n = patch_n.view(1, img_n.shape[0], -1)  # 1,C,K

            # Photometric consistency via LNCC on grayscale patches
            if patch_n.numel() > 0 and patch_v.numel() > 0:
                # Expand patch_v to K samples and convert to grayscale by channel average
                patch_v_rep = patch_v.expand(-1, -1, patch_n.shape[-1])  # 1,C,K
                ref_flat = patch_v_rep.mean(dim=1)  # 1,K
                nea_flat = patch_n.mean(dim=1)      # 1,K
                ncc_val, _ = lncc(ref_flat, nea_flat)  # returns (ncc, mask)
                photo = ncc_val.mean()
                if torch.isfinite(photo):
                    photometric_losses.append(photo)

            # Geometric consistency (depth agreement after warp)
            # Reproject patch center to nearest depth using plane geometry: compare with nb plane depth at projected coord
            # Sample nearest plane depth at projected center
            grid_center = torch.stack([(gx.mean()).view(1), (gy.mean()).view(1)], dim=-1).view(1, 1, 1, 2)
            nb_plane_depth = torch.nn.functional.grid_sample(nb_pkg["plane_depth"].unsqueeze(0), grid_center, mode='bilinear', align_corners=True)
            nb_plane_depth = nb_plane_depth.view(1)
            if torch.isfinite(nb_plane_depth):
                geom = (nb_plane_depth - d).abs()
                geometric_losses.append(geom)

    if len(photometric_losses) == 0 and len(geometric_losses) == 0:
        return torch.tensor(0.0, device=device)

    photo_term = torch.stack(photometric_losses).mean() if len(photometric_losses) > 0 else torch.tensor(0.0, device=device)
    geom_term = torch.stack(geometric_losses).mean() if len(geometric_losses) > 0 else torch.tensor(0.0, device=device)

    weight = loss_params.get('multi_view_weight', 0.1)
    total = weight * (photo_term + geom_term)
    return total

