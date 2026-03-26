from typing import Sequence, Tuple

import torch

from .types import RenderImage


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
