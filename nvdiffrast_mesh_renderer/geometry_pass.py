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
        clip_pos_batch = clip_pos.unsqueeze(0)
        view_attr = torch.matmul(mesh.positions_h, camera.view.t())[:, :3].contiguous()
        front_tri, front_normals, back_tri, back_normals = self._split_triangles_by_facing(
            mesh.faces,
            mesh.positions,
            mesh.face_normals,
            camera.position,
        )
        cull_backfaces = self.config.cull_mode == "force" or (self.config.cull_mode == "auto" and not mesh.material.double_sided)
        front_layer_limit = 1 if cull_backfaces else self.config.double_sided_depth_peels
        layers = self._render_side_layers(mesh, camera, clip_pos_batch, view_attr, front_tri, front_normals, side="front", max_layers=front_layer_limit)
        if not cull_backfaces:
            layers.extend(self._render_side_layers(mesh, camera, clip_pos_batch, view_attr, back_tri, back_normals, side="back", max_layers=self.config.double_sided_depth_peels))
        return layers

    def _render_side_layers(
        self,
        mesh: MeshData,
        camera: CameraData,
        clip_pos_batch: torch.Tensor,
        view_attr: torch.Tensor,
        tri: torch.Tensor,
        face_normals: torch.Tensor,
        side: str,
        max_layers: int,
    ) -> List[RenderLayer]:
        if tri.shape[0] == 0:
            return []
        tri = tri.contiguous()
        if max_layers <= 1:
            layer = self._render_layer(mesh, camera, clip_pos_batch, view_attr, tri, face_normals, side)
            return [] if layer is None else [layer]
        layers: List[RenderLayer] = []
        # Peel sequential depth layers within one winding bucket so self-occluding double-sided shells can contribute more than one surface.
        with dr.DepthPeeler(
            self.glctx,
            clip_pos_batch,
            tri,
            [self.config.resolution, self.config.resolution],
            grad_db=False,
        ) as peeler:
            for _ in range(max_layers):
                rast, rast_db = peeler.rasterize_next_layer()
                layer = self._build_layer_from_raster(mesh, camera, clip_pos_batch, view_attr, tri, face_normals, side, rast, rast_db)
                if layer is None:
                    break
                layers.append(layer)
        return layers

    def _render_layer(
        self,
        mesh: MeshData,
        camera: CameraData,
        clip_pos_batch: torch.Tensor,
        view_attr: torch.Tensor,
        tri: torch.Tensor,
        face_normals: torch.Tensor,
        side: str,
    ) -> Optional[RenderLayer]:
        if tri.shape[0] == 0:
            return None
        rast, rast_db = self._rasterize_triangles(clip_pos_batch, tri)
        return self._build_layer_from_raster(mesh, camera, clip_pos_batch, view_attr, tri, face_normals, side, rast, rast_db)

    def _build_layer_from_raster(
        self,
        mesh: MeshData,
        camera: CameraData,
        clip_pos_batch: torch.Tensor,
        view_attr: torch.Tensor,
        tri: torch.Tensor,
        face_normals: torch.Tensor,
        side: str,
        rast: torch.Tensor,
        rast_db: Optional[torch.Tensor],
    ) -> Optional[RenderLayer]:
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
        valid = gbuf.valid
        if not valid.any():
            return gbuf.rast, gbuf.rast_db, valid, None, torch.zeros_like(gbuf.barycentric)
        if gbuf.rast_db is None:
            return gbuf.rast, gbuf.rast_db, valid, None, gbuf.barycentric
        du_dx = gbuf.rast_db[..., 0:1]
        du_dy = gbuf.rast_db[..., 1:2]
        dv_dx = gbuf.rast_db[..., 2:3]
        dv_dy = gbuf.rast_db[..., 3:4]
        dw_dx = -(du_dx + dv_dx)
        dw_dy = -(du_dy + dv_dy)
        bary_db = torch.cat([du_dx, du_dy, dv_dx, dv_dy, dw_dx, dw_dy], dim=-1).contiguous()
        return gbuf.rast, gbuf.rast_db, valid, bary_db, gbuf.barycentric

    def _split_triangles_by_facing(
        self,
        faces: torch.Tensor,
        positions: torch.Tensor,
        face_normals: torch.Tensor,
        camera_position: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        tri_centroids = positions[faces].mean(dim=1)
        view_dir = camera_position.view(1, 3) - tri_centroids
        front_mask = torch.sum(face_normals * view_dir, dim=-1) >= 0.0
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
        metallic = torch.full((*image_shape, 1), fill_value=material.metallic_factor, dtype=base_rgba.dtype, device=base_rgba.device)
        roughness = torch.full((*image_shape, 1), fill_value=material.roughness_factor, dtype=base_rgba.dtype, device=base_rgba.device)
        if material.occlusion_texture is not None and material.occlusion_texture is material.metallic_roughness_texture:
            orm_tex = sample_texture(material.metallic_roughness_texture, uv, uv_da=uv_da, boundary_mode="wrap")
            if orm_tex is not None:
                ao = ao * (1.0 - material.occlusion_strength + material.occlusion_strength * orm_tex[..., :1])
                if orm_tex.shape[-1] >= 2:
                    roughness = roughness * orm_tex[..., 1:2]
                if orm_tex.shape[-1] >= 3:
                    metallic = metallic * orm_tex[..., 2:3]
        else:
            ao_tex = sample_texture(material.occlusion_texture, uv, uv_da=uv_da, boundary_mode="wrap")
            if ao_tex is not None:
                ao = ao * (1.0 - material.occlusion_strength + material.occlusion_strength * ao_tex[..., :1])
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
            torch.clamp(roughness, 0.02, 1.0).contiguous(),
            torch.clamp(metallic, 0.0, 1.0).contiguous(),
        )
