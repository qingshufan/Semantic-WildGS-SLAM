from typing import Dict, Tuple
import torch
from torch import Tensor
from torch.nn import Module
import torch.nn.functional as F
from thirdparty.gaussian_splatting.utils.loss_utils import ssim
from src.utils.dyn_uncertainty import mapping_utils as map_utils


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    conv_y = torch.tensor(
        [[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device="cuda"
    )
    conv_x = torch.tensor(
        [[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device="cuda"
    )
    normalizer = 1.0 / torch.abs(conv_y).sum()
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = normalizer * torch.nn.functional.conv2d(
        p_img, conv_x.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = normalizer * torch.nn.functional.conv2d(
        p_img, conv_y.view(1, 1, 3, 3).repeat(c, 1, 1, 1), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    conv_y = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    conv_x = torch.ones((1, 1, 3, 3), dtype=torch.float32, device="cuda")
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), conv_x.repeat(c, 1, 1, 1), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), conv_y.repeat(c, 1, 1, 1), groups=c
    )

    return img_grad_v[0] == torch.sum(conv_x), img_grad_h[0] == torch.sum(conv_y)


def get_loss_tracking(config, image, depth, opacity, viewpoint, monocular=True, uncertainty=None):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if monocular:
        return get_loss_tracking_rgb(config, image_ab, opacity, viewpoint, uncertainty)
    else:
        raise NotImplementedError(f"Only implemented monocular, not rgbd for uncertainty-aware tracking")
        # return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint, uncertainty)

def get_loss_tracking_rgb(config, image, opacity, viewpoint, uncertainty=None):
    """Compute RGB tracking loss between rendered and ground truth images.
    This function adds uncertainty mask on the original function from MonoGS
    
    Args:
        config: Configuration dictionary containing training parameters
        image: Rendered RGB image tensor (3, H, W)
        opacity: Opacity tensor (1, H, W) 
        viewpoint: Camera object containing ground truth image and gradient mask
        uncertainty: Optional uncertainty estimates (H, W)
    
    Returns:
        Scalar loss tensor
    """
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    # Create mask for valid RGB pixels above intensity threshold
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask

    # Compute L1 loss weighted by opacity
    l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    if uncertainty is not None:
        # Weight loss inversely proportional to uncertainty
        # Higher uncertainty -> lower weight 
        # Zero out weights below 0.1 to ignore highly uncertain regions (Todo: verify this is useful)
        weights = 0.5 / (uncertainty.unsqueeze(0))**2
        weights = torch.where(weights < 0.1, 0.0, weights)
        l1 *= weights
    return l1.mean()

# Not used, but kept for reference
def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, uncertainty=None
):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[None]
    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    opacity_mask = (opacity > 0.95).view(*depth.shape)

    l1_rgb = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    l1_depth = torch.abs(depth * depth_mask - gt_depth * depth_mask)
    return alpha * l1_rgb + (1 - alpha) * l1_depth.mean()


def get_loss_mapping(config, image, depth, viewpoint, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgbd(config, image, depth, viewpoint):
    alpha = config["Training"]["alpha"] if "alpha" in config["Training"] else 0.95
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    gt_image = viewpoint.original_image.cuda()
    _, h, w = gt_image.shape
    mask_shape = (1, h, w)

    gt_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=image.device
    )[
        None
    ]
    loss = 0
    if config["Training"]["ssim_loss"]:
        ssim_loss = 1.0 - ssim(image, gt_image)

    rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    if config["Training"]["ssim_loss"]:
        hyperparameter = config["opt_params"]["lambda_dssim"]
        loss += (1.0 - hyperparameter) * l1_rgb + hyperparameter * ssim_loss
    else:
        loss += l1_rgb

    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)

    return alpha * loss.mean() + (1 - alpha) * l1_depth.mean()


def get_loss_mapping_uncertainty(
    config: Dict,
    rendered_img: Tensor,
    rendered_depth: Tensor,
    viewpoint, # from src.utils.camera_utils import Camera, to avoid loop import
    opacity: Tensor,
    uncertainty_network: Module,
    train_frac: float,
    ssim_frac: float,
    initialization: bool = False,
    freeze_uncertainty_loss: bool = False,  # Renamed parameter
) -> Tuple[Tensor, Tensor]:
    """为SLAM系统计算带不确定性估计的Mapping损失。

    该函数估计每个像素的不确定性，并将RGB损失和深度损失与不确定性权重相结合，以处理动态物体。

    参数:
        config: 配置参数
        rendered_img: 渲染的RGB图像(3, H, W) 高斯训练的初步渲染帧
        rendered_depth: 渲染的深度图(1, H, W)
        viewpoint: 包含真实图像和参考深度的相机
        opacity: 渲染的不透明度掩码(1, H, W)
        uncertainty_network: 用于不确定性预测的MLP(多层感知机)
        train_frac: 用于自适应加权的训练进度(0-1)
        ssim_frac: SSIM损失的权重比例
        initialization: 若为True则跳过曝光补偿
        freeze_uncertainty_loss: 若为True则停止不确定性损失的梯度流动

    返回:
        uncertainty: 每个像素的不确定性估计(H, W)
        total_loss: 组合的映射损失和不确定性损失(标量)
    """
    # Apply exposure compensation if not initialization
    rendered_img = rendered_img if initialization else (
        torch.exp(viewpoint.exposure_a) * rendered_img + viewpoint.exposure_b
    )
    # Get config parameters
    alpha = config["Training"].get("alpha", 0.95)
    rgb_boundary_threshold = config["Training"]["rgb_boundary_threshold"]
    
    # Get reference data
    gt_img = viewpoint.original_image.cuda()
    
    ref_depth = torch.from_numpy(viewpoint.depth).to(
        dtype=torch.float32, device=rendered_img.device
    )[None]

    # 过滤掉真值图像（gt_img）中不符合特定条件的像素
    _, h, w = gt_img.shape
    mask_shape = (1, h, w)
    rgb_pixel_mask = (gt_img.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)

    # Compute SSIM loss if enabled
    ssim_loss = 1.0 - ssim(rendered_img, gt_img) if config["Training"]["ssim_loss"] else 0.0

    # Predict uncertainty from features
    features = viewpoint.features.to(device=rendered_img.device)
    uncertainty = uncertainty_network(features)

    # **** qingshufan modified code start ****
    # uncertainty.fill_(1.0)
    uncertainty_mean = uncertainty.mean()
    uncertainty_threshold = config["uncertainty_params"].get("uncertainty_threshold", 0.5)
    high_uncertainty_loss = 0
    if uncertainty_mean < uncertainty_threshold:
        high_uncertainty_loss = (uncertainty_mean - uncertainty_threshold) ** 2
    # **** qingshufan modified code end ****

    # Compute mapping losses with uncertainty 计算论文公式的L_uncer、L1下的RGB、Depth、改进SSIM（用到L_uncer里了）
    uncer_loss, uncer_resized, l1_rgb, l1_depth = map_utils.compute_mapping_loss_components(
        gt_img,
        rendered_img,
        ref_depth,
        rendered_depth,
        uncertainty,
        opacity.view(*mask_shape),
        train_fraction=train_frac,
        ssim_fraction=ssim_frac,
        uncertainty_config=config["uncertainty_params"],
        mask=rgb_pixel_mask
    )
    
    # **** qingshufan modified code start ****
    if uncer_resized.mean() < uncertainty_threshold:
        uncer_resized = torch.ones_like(uncer_resized)
    # **** qingshufan modified code end ****
    
    # Combine RGB losses 即论文中的L_color
    if config["Training"]["ssim_loss"]:
        lambda_dssim = config["opt_params"]["lambda_dssim"]
        rgb_loss = (1.0 - lambda_dssim) * l1_rgb + lambda_dssim * ssim_loss
    else:
        rgb_loss = l1_rgb

    # Apply uncertainty weighting
    weights = 0.5 / (uncer_resized.unsqueeze(0))**2

    # 将权重低于0.1的区域归零以忽略高度不确定的区域（待办：验证此操作是否有效）
    weights = torch.where(weights < 0.1, 0.0, weights)
    
    rgb_loss = weights * rgb_loss

    # Handle full resolution option
    if config['full_resolution']:
        weights = F.interpolate(
            weights.unsqueeze(0), 
            l1_depth.shape[-2:], 
            mode='bilinear'
        ).squeeze(0)

    # 仅在真实深度(gt_depth)小于渲染深度(rendered_depth)的像素上添加不确定性（添加1.0米的缓冲区）
    # 如果你看到一个移动的干扰物，它必须比静态区域更靠近相机
    # 添加此操作可以有效去除一些悬浮噪点
    uncer_depth_mask = ref_depth < rendered_depth.detach() + 1.0
    l1_depth[uncer_depth_mask] = weights[uncer_depth_mask] * l1_depth[uncer_depth_mask]

    if freeze_uncertainty_loss:
        uncer_loss = uncer_loss.detach()

    # Combine all losses 即论文中的L_render
    total_loss = (
        alpha * rgb_loss.mean() +
        (1 - alpha) * l1_depth.mean() +
        config["uncertainty_params"]["ssim_mult"] * uncer_loss.mean() +
    #**** qingshufan modified code start ****
        high_uncertainty_loss
    #**** qingshufan modified code end ****
    )
    


    return uncertainty, total_loss

def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()