import math
from typing import Optional, Sequence, Tuple

import torch

import nvdiffrast.torch as dr

from .compositor import LayerCompositor
from .config import RenderConfig
from .geometry_pass import GeometryPassRenderer
from .ibl import ImageBasedLighting, distribution_ggx, environment_brdf_approx, fresnel_schlick, fresnel_schlick_roughness, geometry_smith
from .math_utils import safe_normalize
from .textures import sample_texture
from .types import CameraData, RenderImage, RenderLayer


class RenderModeRenderer:
    SUPPORTED_MODES = (
        "beauty",
        "albedo",
        "normal_world",
        "normal_view",
        "face_normal",
        "depth_ndc",
        "depth_linear",
        "mask",
        "triangle_id",
        "uv",
        "roughness",
        "metallic",
        "ao",
        "emissive",
        "wireframe",
        "beauty_plus_wireframe",
    )

    def __init__(
        self,
        config: RenderConfig,
        lights: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        ibl: Optional[ImageBasedLighting],
        geometry: GeometryPassRenderer,
        compositor: LayerCompositor,
    ):
        self.config = config
        self.lights = lights
        self.ibl = ibl
        self.geometry = geometry
        self.compositor = compositor

    def render_mode(self, mode_name: str, layer: RenderLayer, camera: CameraData) -> RenderImage:
        if mode_name == "beauty":
            return self._render_beauty(layer, camera)
        if mode_name == "beauty_plus_wireframe":
            base = self._render_beauty(layer, camera)
            wire = self._render_wireframe(layer)
            rgb, alpha = self.compositor.compose_overlays((base.rgb, base.alpha), [(wire.rgb, wire.alpha)])
            return RenderImage(rgb=rgb, alpha=alpha, depth=base.depth, valid=alpha > 1e-5)
        if mode_name == "wireframe":
            return self._render_wireframe(layer)

        gbuf = layer.geometry
        alpha = self._resolve_alpha(layer)
        if mode_name == "albedo":
            return self._render_buffer(torch.clamp(gbuf.base_rgba[..., :3], min=0.0), alpha, gbuf)
        if mode_name == "normal_world":
            return self._render_buffer(self._encode_normal(self._oriented_world_normal(layer)), alpha, gbuf)
        if mode_name == "normal_view":
            return self._render_buffer(self._encode_normal(self._oriented_view_normal(layer)), alpha, gbuf)
        if mode_name == "face_normal":
            return self._render_buffer(self._encode_normal(self._oriented_face_normal(layer)), alpha, gbuf)
        if mode_name == "depth_ndc":
            return self._render_scalar_buffer(gbuf.rast[..., 2:3], alpha, gbuf, clamp_rgb_min=False)
        if mode_name == "depth_linear":
            return self._render_scalar_buffer(self._resolve_depth(gbuf), alpha, gbuf, clamp_rgb_min=False)
        if mode_name == "mask":
            return self._render_scalar_buffer(torch.ones_like(alpha), alpha, gbuf)
        if mode_name == "triangle_id":
            denom = max(float(gbuf.tri.shape[0]), 1.0)
            return self._render_scalar_buffer((gbuf.triangle_id.float() + 0.5) / denom, alpha, gbuf)
        if mode_name == "uv":
            uv = torch.zeros_like(gbuf.world_pos) if gbuf.uv is None else torch.cat([torch.remainder(gbuf.uv, 1.0), torch.zeros_like(gbuf.uv[..., :1])], dim=-1)
            return self._render_buffer(uv, alpha, gbuf)
        if mode_name == "roughness":
            return self._render_scalar_buffer(gbuf.roughness, alpha, gbuf)
        if mode_name == "metallic":
            return self._render_scalar_buffer(gbuf.metallic, alpha, gbuf)
        if mode_name == "ao":
            return self._render_scalar_buffer(gbuf.ao, alpha, gbuf)
        if mode_name == "emissive":
            return self._render_buffer(torch.clamp(gbuf.emissive, min=0.0), alpha, gbuf)
        raise ValueError(f"Unsupported render mode: {mode_name}")

    def _render_beauty(self, layer: RenderLayer, camera: CameraData) -> RenderImage:
        mesh, gbuf = layer.mesh, layer.geometry
        material = mesh.material
        normal = self._resolve_shading_normal(layer)
        face_normal = self._oriented_face_normal(layer)
        view_dir = safe_normalize(camera.position.view(1, 1, 1, 3) - gbuf.world_pos)
        base_rgb = torch.clamp(gbuf.base_rgba[..., :3], min=0.0)
        alpha = self._resolve_alpha(layer)
        shaded = self._shade(material.workflow, base_rgb, normal, face_normal, view_dir, gbuf.emissive, gbuf.ao, gbuf.roughness, gbuf.metallic, gbuf)
        return self._render_buffer(shaded, alpha, gbuf)

    def _render_wireframe(self, layer: RenderLayer) -> RenderImage:
        gbuf = layer.geometry
        alpha = self._resolve_alpha(layer)
        if self.config.wireframe_opacity <= 0.0 or self.config.wireframe_thickness_px <= 0.0:
            zero_alpha = torch.zeros_like(alpha)
            return RenderImage(rgb=torch.zeros_like(gbuf.world_pos), alpha=zero_alpha, depth=self._resolve_depth(gbuf), valid=zero_alpha > 1e-5)
        _rast, _rast_db, valid, bary_db, bary = self.geometry.render_wireframe_pass(layer)
        if bary_db is None:
            zero_alpha = torch.zeros_like(alpha)
            return RenderImage(rgb=torch.zeros_like(gbuf.world_pos), alpha=zero_alpha, depth=self._resolve_depth(gbuf), valid=zero_alpha > 1e-5)
        edge_distance, edge_index = torch.min(bary, dim=-1, keepdim=True)
        bary_deriv = bary_db.view(*bary.shape[:-1], 3, 2)
        edge_width = torch.gather(bary_deriv.abs().sum(dim=-1), -1, edge_index)
        edge_px = edge_distance / torch.clamp(edge_width, min=1e-6)
        coverage = 1.0 - self._smoothstep(
            torch.full_like(edge_px, max(self.config.wireframe_thickness_px - 0.5, 0.0)),
            torch.full_like(edge_px, self.config.wireframe_thickness_px + 0.5),
            edge_px,
        )
        line_alpha = alpha * coverage * valid.float() * self.config.wireframe_opacity
        wire_color = torch.as_tensor(self.config.wireframe_color, dtype=gbuf.world_pos.dtype, device=gbuf.world_pos.device).view(1, 1, 1, 3)
        return RenderImage(
            rgb=wire_color * line_alpha,
            alpha=line_alpha,
            depth=self._resolve_depth(gbuf),
            valid=line_alpha > 1e-5,
        )

    def _resolve_alpha(self, layer: RenderLayer) -> torch.Tensor:
        material = layer.mesh.material
        alpha = layer.geometry.base_rgba[..., 3:4]
        if material.alpha_mode == "OPAQUE":
            alpha = torch.ones_like(alpha)
        elif material.alpha_mode == "MASK":
            alpha = (alpha >= material.alpha_cutoff).float()
        return alpha * layer.geometry.valid.float()

    def _resolve_shading_normal(self, layer: RenderLayer) -> torch.Tensor:
        mesh, gbuf = layer.mesh, layer.geometry
        normal = self._oriented_world_normal(layer)
        if mesh.material.normal_texture is None or gbuf.uv is None or gbuf.tangent is None:
            return normal
        tangent = safe_normalize(gbuf.tangent[..., :3] - normal * torch.sum(normal * gbuf.tangent[..., :3], dim=-1, keepdim=True))
        bitangent = safe_normalize(torch.cross(normal, tangent, dim=-1) * gbuf.tangent[..., 3:4])
        normal_map = sample_texture(mesh.material.normal_texture, gbuf.uv, uv_da=gbuf.uv_da, boundary_mode="wrap")
        if normal_map is None:
            return normal
        normal_ts = normal_map[..., :3] * 2.0 - 1.0
        normal_ts_xy = normal_ts[..., :2] * mesh.material.normal_scale
        return safe_normalize(tangent * normal_ts_xy[..., :1] + bitangent * normal_ts_xy[..., 1:2] + normal * normal_ts[..., 2:3])

    def _render_buffer(
        self,
        rgb: torch.Tensor,
        alpha: torch.Tensor,
        gbuf,
        antialias: bool = True,
        clamp_rgb_min: bool = True,
    ) -> RenderImage:
        depth = self._resolve_depth(gbuf)
        rgba_depth = torch.cat([rgb * alpha, alpha, depth * alpha], dim=-1)
        if self.config.antialias and antialias:
            rgba_depth = dr.antialias(rgba_depth, gbuf.rast, gbuf.clip_pos, gbuf.tri)
        aa_alpha = torch.clamp(rgba_depth[..., 3:4], 0.0, 1.0)
        aa_depth = torch.where(aa_alpha > 1e-5, rgba_depth[..., 4:5] / torch.clamp(aa_alpha, min=1e-8), torch.zeros_like(aa_alpha))
        aa_rgb = rgba_depth[..., :3] if not clamp_rgb_min else torch.clamp(rgba_depth[..., :3], min=0.0)
        return RenderImage(rgb=aa_rgb, alpha=aa_alpha, depth=aa_depth, valid=aa_alpha > 1e-5)

    def _render_scalar_buffer(
        self,
        value: torch.Tensor,
        alpha: torch.Tensor,
        gbuf,
        antialias: bool = True,
        clamp_rgb_min: bool = True,
    ) -> RenderImage:
        return self._render_buffer(value.expand_as(gbuf.world_pos), alpha, gbuf, antialias=antialias, clamp_rgb_min=clamp_rgb_min)

    def _shade(
        self,
        workflow: str,
        base_rgb: torch.Tensor,
        normal: torch.Tensor,
        face_normal: torch.Tensor,
        view_dir: torch.Tensor,
        emissive: torch.Tensor,
        ao: torch.Tensor,
        roughness: torch.Tensor,
        metallic: torch.Tensor,
        gbuf,
    ) -> torch.Tensor:
        ambient = torch.tensor([0.04, 0.045, 0.05], device=base_rgb.device, dtype=base_rgb.dtype).view(1, 1, 1, 3)
        if workflow != "pbr":
            shaded = emissive + ambient * base_rgb * ao
            for light_dir, light_color in self.lights:
                n_dot_l = torch.clamp(torch.sum(normal * safe_normalize(light_dir.expand_as(normal)), dim=-1, keepdim=True), 0.0, 1.0)
                shaded = shaded + base_rgb * light_color * n_dot_l
            if self.ibl is not None and self.config.env_usage in {"light", "both"}:
                shaded = shaded + base_rgb * self.ibl.diffuse(normal) * ao
            return torch.where(gbuf.valid.expand_as(base_rgb), shaded, torch.zeros_like(base_rgb))
        f0 = 0.04 * (1.0 - metallic) + base_rgb * metallic
        n_dot_v = torch.clamp(torch.sum(normal * view_dir, dim=-1, keepdim=True), min=1e-4, max=1.0)
        shaded = emissive + ambient * base_rgb * (1.0 - metallic) * ao
        for light_dir, light_color in self.lights:
            light = safe_normalize(light_dir.expand_as(normal))
            half_vec = safe_normalize(view_dir + light)
            n_dot_l = torch.clamp(torch.sum(normal * light, dim=-1, keepdim=True), 0.0, 1.0)
            n_dot_h = torch.clamp(torch.sum(normal * half_vec, dim=-1, keepdim=True), 0.0, 1.0)
            h_dot_v = torch.clamp(torch.sum(half_vec * view_dir, dim=-1, keepdim=True), 0.0, 1.0)
            fresnel = fresnel_schlick(h_dot_v, f0)
            specular = distribution_ggx(n_dot_h, roughness) * geometry_smith(n_dot_v, n_dot_l, roughness) * fresnel
            specular = specular / torch.clamp(4.0 * n_dot_v * n_dot_l, min=1e-4)
            diffuse = (1.0 - fresnel) * (1.0 - metallic) * base_rgb / math.pi
            shaded = shaded + (diffuse + specular) * light_color * n_dot_l
        if self.ibl is not None and self.config.env_usage in {"light", "both"}:
            diffuse_weight = (1.0 - fresnel_schlick_roughness(n_dot_v, f0, roughness)) * (1.0 - metallic)
            shaded = shaded + base_rgb * diffuse_weight * self.ibl.diffuse(normal) * ao
            shaded = shaded + self.ibl.specular(normal, view_dir, roughness) * environment_brdf_approx(f0, roughness, n_dot_v)
        return torch.where(gbuf.valid.expand_as(base_rgb), shaded, torch.zeros_like(base_rgb))

    def _oriented_world_normal(self, layer: RenderLayer) -> torch.Tensor:
        normal = layer.geometry.normal_world
        return safe_normalize(-normal if layer.geometry.side == "back" else normal)

    def _oriented_view_normal(self, layer: RenderLayer) -> torch.Tensor:
        normal = layer.geometry.normal_view
        return safe_normalize(-normal if layer.geometry.side == "back" else normal)

    def _oriented_face_normal(self, layer: RenderLayer) -> torch.Tensor:
        normal = layer.geometry.face_normal_world
        return safe_normalize(-normal if layer.geometry.side == "back" else normal)

    def _resolve_depth(self, gbuf) -> torch.Tensor:
        return torch.where(gbuf.valid, -gbuf.view_pos[..., 2:3], torch.zeros_like(gbuf.valid, dtype=gbuf.view_pos.dtype))

    def _encode_normal(self, normal: torch.Tensor) -> torch.Tensor:
        return normal * 0.5 + 0.5

    def _smoothstep(self, edge0: torch.Tensor, edge1: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        t = torch.clamp((x - edge0) / torch.clamp(edge1 - edge0, min=1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)
