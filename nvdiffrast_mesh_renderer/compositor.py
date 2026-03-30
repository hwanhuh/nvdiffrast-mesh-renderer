from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from .types import RenderImage


@dataclass
class LayerStack:
    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor
    valid: torch.Tensor


class LayerCompositor:
    def composite_mesh_layers(
        self,
        bg_rgb: torch.Tensor,
        bg_alpha: torch.Tensor,
        layers: Sequence[RenderImage],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not layers:
            return bg_rgb, bg_alpha
        rgb_stack = torch.cat([layer.rgb for layer in layers], dim=0)
        alpha_stack = torch.cat([layer.alpha for layer in layers], dim=0)
        depth_stack = torch.cat([layer.depth for layer in layers], dim=0)
        valid_stack = torch.cat([layer.valid for layer in layers], dim=0)
        sortable_depth = torch.where(valid_stack, depth_stack, torch.full_like(depth_stack, -1e9))
        order = torch.argsort(sortable_depth[..., 0], dim=0, descending=True)
        gather_rgb = order[..., None].expand(-1, -1, -1, 3)
        gather_alpha = order[..., None].expand(-1, -1, -1, 1)
        sorted_rgb = torch.gather(rgb_stack, 0, gather_rgb) * (torch.gather(valid_stack.float(), 0, gather_alpha) > 0.5).float()
        sorted_alpha = torch.gather(alpha_stack, 0, gather_alpha) * (torch.gather(valid_stack.float(), 0, gather_alpha) > 0.5).float()
        out_rgb, out_alpha = bg_rgb[0], bg_alpha[0]
        for idx in range(sorted_rgb.shape[0]):
            out_rgb = sorted_rgb[idx] + out_rgb * (1.0 - sorted_alpha[idx])
            out_alpha = sorted_alpha[idx] + out_alpha * (1.0 - sorted_alpha[idx])
        return out_rgb.unsqueeze(0), out_alpha.unsqueeze(0)

    def create_layer_stack(self, bg_rgb: torch.Tensor, bg_alpha: torch.Tensor, capacity: int) -> LayerStack:
        h, w = bg_rgb.shape[1], bg_rgb.shape[2]
        device = bg_rgb.device
        rgb_dtype = bg_rgb.dtype
        alpha_dtype = bg_alpha.dtype
        return LayerStack(
            rgb=torch.zeros((capacity, h, w, 3), dtype=rgb_dtype, device=device),
            alpha=torch.zeros((capacity, h, w, 1), dtype=alpha_dtype, device=device),
            depth=torch.zeros((capacity, h, w, 1), dtype=alpha_dtype, device=device),
            valid=torch.zeros((capacity, h, w, 1), dtype=torch.bool, device=device),
        )

    def accumulate_layer_stack(self, stack: LayerStack, layers: Sequence[RenderImage]) -> LayerStack:
        if not layers:
            return stack
        new_rgb = torch.cat([layer.rgb for layer in layers], dim=0)
        new_alpha = torch.cat([layer.alpha for layer in layers], dim=0)
        new_depth = torch.cat([layer.depth for layer in layers], dim=0)
        new_valid = torch.cat([layer.valid for layer in layers], dim=0)
        rgb_all = torch.cat([stack.rgb, new_rgb], dim=0)
        alpha_all = torch.cat([stack.alpha, new_alpha], dim=0)
        depth_all = torch.cat([stack.depth, new_depth], dim=0)
        valid_all = torch.cat([stack.valid, new_valid], dim=0)
        inf_depth = torch.full_like(depth_all, float('inf'))
        nearest_order = torch.argsort(torch.where(valid_all, depth_all, inf_depth)[..., 0], dim=0, descending=False)
        keep = nearest_order[: stack.rgb.shape[0]]
        gather_rgb = keep[..., None].expand(-1, -1, -1, 3)
        gather_scalar = keep[..., None].expand(-1, -1, -1, 1)
        return LayerStack(
            rgb=torch.gather(rgb_all, 0, gather_rgb),
            alpha=torch.gather(alpha_all, 0, gather_scalar),
            depth=torch.gather(depth_all, 0, gather_scalar),
            valid=torch.gather(valid_all, 0, gather_scalar),
        )

    def composite_layer_stack(
        self,
        bg_rgb: torch.Tensor,
        bg_alpha: torch.Tensor,
        stack: LayerStack,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_rgb, out_alpha = bg_rgb[0], bg_alpha[0]
        for idx in range(stack.rgb.shape[0] - 1, -1, -1):
            valid = stack.valid[idx]
            rgb = torch.where(valid.expand_as(stack.rgb[idx]), stack.rgb[idx], torch.zeros_like(stack.rgb[idx]))
            alpha = torch.where(valid, stack.alpha[idx], torch.zeros_like(stack.alpha[idx]))
            out_rgb = rgb + out_rgb * (1.0 - alpha)
            out_alpha = alpha + out_alpha * (1.0 - alpha)
        return out_rgb.unsqueeze(0), out_alpha.unsqueeze(0)

    def merge_double_sided(self, front: RenderImage, back: RenderImage, alpha_mode: str) -> RenderImage:
        use_front, use_back, visible = self._select_nearest_visible_side(front, back)
        if alpha_mode == "BLEND":
            rgb, alpha = self.composite_mesh_layers(torch.zeros_like(front.rgb), torch.zeros_like(front.alpha), [front, back])
            valid = alpha > 1e-5
            depth = torch.where(use_front, front.depth, torch.where(use_back, back.depth, torch.zeros_like(front.depth)))
            return RenderImage(rgb=rgb, alpha=alpha, depth=torch.where(valid, depth, torch.zeros_like(depth)), valid=valid)
        return RenderImage(
            rgb=torch.where(
                use_front.expand_as(front.rgb),
                front.rgb,
                torch.where(use_back.expand_as(back.rgb), back.rgb, torch.zeros_like(front.rgb)),
            ),
            alpha=torch.where(use_front, front.alpha, torch.where(use_back, back.alpha, torch.zeros_like(front.alpha))),
            depth=torch.where(use_front, front.depth, torch.where(use_back, back.depth, torch.zeros_like(front.depth))),
            valid=visible,
        )

    def _select_nearest_visible_side(
        self,
        front: RenderImage,
        back: RenderImage,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        front_visible = front.valid & (front.alpha > 1e-5)
        back_visible = back.valid & (back.alpha > 1e-5)
        # Depth stores positive camera distance, so smaller values are nearer.
        use_front = front_visible & (~back_visible | (front.depth <= back.depth))
        use_back = back_visible & ~use_front
        return use_front, use_back, front_visible | back_visible

    def compose_overlays(
        self,
        base: Tuple[torch.Tensor, torch.Tensor],
        overlays: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        premultiplied: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        out_rgb, out_alpha = base
        for src_rgb, src_alpha in overlays:
            rgb = src_rgb if premultiplied else src_rgb * src_alpha
            out_rgb = rgb + out_rgb * (1.0 - src_alpha)
            out_alpha = src_alpha + out_alpha * (1.0 - src_alpha)
        return out_rgb, out_alpha
