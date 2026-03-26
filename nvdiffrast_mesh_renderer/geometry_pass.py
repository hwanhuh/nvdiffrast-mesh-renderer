from typing import List, Optional, Tuple

import torch

import nvdiffrast.torch as dr

from .config import RenderConfig
from .math_utils import safe_normalize
from .textures import sample_texture
from .types import CameraData, GeometryBuffer, MeshData, RenderLayer


class GeometryPassRenderer:
    def __init__(self, glctx: dr.RasterizeCudaContext, config: RenderConfig):
        self.glctx = glctx
        self.config = config

    def render_geometry_pass(self, mesh: MeshData, camera: CameraData) -> List[RenderLayer]:
        clip_pos = torch.matmul(mesh.positions_h, camera.mvp.t()).contiguous()
        view_attr = torch.matmul(mesh.positions_h, camera.view.t())[:, :3].contiguous()
        front_tri, front_normals, back_tri, back_normals = self._split_triangles_by_facing(mesh.faces, mesh.face_normals, clip_pos)
        cull_backfaces = self.config.cull_mode == "force" or (self.config.cull_mode == "auto" and not mesh.material.double_sided)
        layers = []
        front = self._render_layer(mesh, camera, clip_pos, view_attr, front_tri, front_normals, side="front")
        if front is not None:
            layers.append(front)
        if not cull_backfaces:
            back = self._render_layer(mesh, camera, clip_pos, view_attr, back_tri, back_normals, side="back")
            if back is not None:
                layers.append(back)
        return layers

    def _render_layer(
        self,
        mesh: MeshData,
        camera: CameraData,
        clip_pos: torch.Tensor,
        view_attr: torch.Tensor,
        tri: torch.Tensor,
        face_normals: torch.Tensor,
        side: str,
    ) -> Optional[RenderLayer]:
        if tri.shape[0] == 0:
            return None
        clip_pos_batch = clip_pos.unsqueeze(0)
        rast, rast_db = self._rasterize_triangles(clip_pos_batch, tri)
        valid = rast[..., 3:] > 0
        if not valid.any():
            return None
        uvw = torch.cat([rast[..., :2], 1.0 - rast[..., :1] - rast[..., 1:2]], dim=-1)
        world_pos, _ = dr.interpolate(mesh.positions.contiguous(), rast, tri)
        view_pos, _ = dr.interpolate(view_attr, rast, tri)
        normal_world, _ = dr.interpolate(mesh.normals.contiguous(), rast, tri)
        uv, uv_da = self._interpolate_optional(mesh.uv, rast, tri, rast_db, "all")
        tangent, _ = self._interpolate_optional(mesh.tangents, rast, tri)
        vertex_color, _ = self._interpolate_optional(mesh.vertex_colors, rast, tri)
        face_normal_world = self._gather_face_normals(face_normals, rast)
        view_rot = camera.view[:3, :3]
        normal_world = safe_normalize(normal_world)
        face_normal_world = safe_normalize(face_normal_world)
        base_rgba, emissive, ao, roughness, metallic = self._sample_material_channels(mesh, rast, uv, uv_da, vertex_color)
        return RenderLayer(
            mesh=mesh,
            geometry=GeometryBuffer(
                rast=rast,
                rast_db=rast_db,
                valid=valid,
                triangle_id=(rast[..., 3:4].long() - 1).clamp(min=0),
                uvw=uvw,
                barycentric=uvw,
                world_pos=world_pos,
                view_pos=view_pos,
                normal_world=normal_world,
                normal_view=safe_normalize(torch.matmul(normal_world, view_rot.t())),
                face_normal_world=face_normal_world,
                face_normal_view=safe_normalize(torch.matmul(face_normal_world, view_rot.t())),
                uv=uv,
                uv_da=uv_da,
                tangent=tangent,
                vertex_color=vertex_color,
                base_rgba=base_rgba,
                emissive=emissive,
                ao=ao,
                roughness=roughness,
                metallic=metallic,
                clip_pos=clip_pos_batch,
                tri=tri,
                side=side,
            ),
        )

    def render_wireframe_pass(self, layer: RenderLayer) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        gbuf = layer.geometry
        if gbuf.tri.shape[0] == 0:
            shape = (1, self.config.resolution, self.config.resolution, 1)
            empty = torch.zeros(shape, dtype=gbuf.rast.dtype, device=gbuf.rast.device)
            return empty, None, empty.bool(), None, torch.zeros(shape[:-1] + (3,), dtype=gbuf.rast.dtype, device=gbuf.rast.device)
        expanded_clip = gbuf.clip_pos[:, gbuf.tri.reshape(-1).long(), :].contiguous()
        tri_count = gbuf.tri.shape[0]
        wire_tri = torch.arange(tri_count * 3, device=gbuf.tri.device, dtype=gbuf.tri.dtype).reshape(tri_count, 3).contiguous()
        corner_bary = torch.eye(3, dtype=gbuf.rast.dtype, device=gbuf.rast.device).repeat(tri_count, 1).contiguous()
        rast, rast_db = self._rasterize_triangles(expanded_clip, wire_tri)
        valid = rast[..., 3:] > 0
        if not valid.any():
            return rast, rast_db, valid, None, torch.zeros_like(rast[..., :3])
        bary, bary_db = dr.interpolate(corner_bary, rast, wire_tri, rast_db=rast_db, diff_attrs="all")
        return rast, rast_db, valid, bary_db, bary

    def _split_triangles_by_facing(
        self,
        faces: torch.Tensor,
        face_normals: torch.Tensor,
        clip_pos: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tri_clip = clip_pos[faces]
        w = torch.where(torch.abs(tri_clip[..., 3:4]) < 1e-8, torch.full_like(tri_clip[..., 3:4], 1e-8), tri_clip[..., 3:4])
        tri_ndc_xy = tri_clip[..., :2] / w
        e1 = tri_ndc_xy[:, 1] - tri_ndc_xy[:, 0]
        e2 = tri_ndc_xy[:, 2] - tri_ndc_xy[:, 0]
        front_mask = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]) >= 0.0
        if front_mask.all():
            return faces, face_normals, faces.new_zeros((0, 3)), face_normals.new_zeros((0, 3))
        if (~front_mask).all():
            return faces.new_zeros((0, 3)), face_normals.new_zeros((0, 3)), faces, face_normals
        return (
            faces[front_mask].contiguous(),
            face_normals[front_mask].contiguous(),
            faces[~front_mask].contiguous(),
            face_normals[~front_mask].contiguous(),
        )

    def _interpolate_optional(
        self,
        attr: Optional[torch.Tensor],
        rast: torch.Tensor,
        tri: torch.Tensor,
        rast_db: Optional[torch.Tensor] = None,
        diff_attrs: Optional[str] = None,
    ):
        return (None, None) if attr is None else dr.interpolate(attr.contiguous(), rast, tri.contiguous(), rast_db=rast_db, diff_attrs=diff_attrs)

    def _gather_face_normals(self, face_normals: torch.Tensor, rast: torch.Tensor) -> torch.Tensor:
        tri_idx = (rast[..., 3].long() - 1).clamp(min=0)
        return face_normals[tri_idx.reshape(-1)].reshape(rast.shape[0], rast.shape[1], rast.shape[2], 3)

    def _rasterize_triangles(
        self,
        clip_pos: torch.Tensor,
        tri: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Keep rasterization behind one entrypoint so a future DepthPeeler path can slot in here.
        return dr.rasterize(self.glctx, clip_pos, tri.contiguous(), [self.config.resolution, self.config.resolution], grad_db=False)

    def _sample_material_channels(
        self,
        mesh: MeshData,
        rast: torch.Tensor,
        uv: Optional[torch.Tensor],
        uv_da: Optional[torch.Tensor],
        vertex_color: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        material = mesh.material
        image_shape = (rast.shape[0], rast.shape[1], rast.shape[2])
        base_rgba = material.base_color_factor.view(1, 1, 1, 4).expand(*image_shape, 4)
        base_tex = sample_texture(material.base_color_texture, uv, uv_da=uv_da, boundary_mode="wrap")
        if base_tex is not None:
            base_tex = torch.cat([base_tex, torch.ones_like(base_tex[..., :1])], dim=-1) if base_tex.shape[-1] == 3 else base_tex
            base_rgba = base_rgba * base_tex[..., :4]
        if vertex_color is not None:
            color_rgba = torch.cat([vertex_color, torch.ones_like(vertex_color[..., :1])], dim=-1) if vertex_color.shape[-1] == 3 else vertex_color
            base_rgba = base_rgba * color_rgba[..., :4]
        emissive = material.emissive_factor.view(1, 1, 1, 3).expand(*image_shape, 3)
        emissive_tex = sample_texture(material.emissive_texture, uv, uv_da=uv_da, boundary_mode="wrap")
        if emissive_tex is not None:
            emissive = emissive * emissive_tex[..., :3]
        ao = torch.ones((*image_shape, 1), dtype=base_rgba.dtype, device=base_rgba.device)
        ao_tex = sample_texture(material.occlusion_texture, uv, uv_da=uv_da, boundary_mode="wrap")
        if ao_tex is not None:
            ao = ao * (1.0 - material.occlusion_strength + material.occlusion_strength * ao_tex[..., :1])
        metallic = torch.full((*image_shape, 1), fill_value=material.metallic_factor, dtype=base_rgba.dtype, device=base_rgba.device)
        roughness = torch.full((*image_shape, 1), fill_value=material.roughness_factor, dtype=base_rgba.dtype, device=base_rgba.device)
        mr_tex = sample_texture(material.metallic_roughness_texture, uv, uv_da=uv_da, boundary_mode="wrap")
        if mr_tex is not None:
            if mr_tex.shape[-1] >= 2:
                roughness = roughness * mr_tex[..., 1:2]
            if mr_tex.shape[-1] >= 3:
                metallic = metallic * mr_tex[..., 2:3]
        return (
            base_rgba.contiguous(),
            emissive.contiguous(),
            ao.contiguous(),
            torch.clamp(roughness, 0.045, 1.0).contiguous(),
            torch.clamp(metallic, 0.0, 1.0).contiguous(),
        )
