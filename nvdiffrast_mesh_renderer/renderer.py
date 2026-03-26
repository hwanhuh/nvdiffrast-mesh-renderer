import pathlib
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

import nvdiffrast.torch as dr

from . import util

from .beauty import RenderModeRenderer
from .compositor import LayerCompositor
from .config import RenderConfig
from .environment import EnvironmentService
from .geometry_pass import GeometryPassRenderer
from .geometry_utils import scene_bounds
from .ibl import ImageBasedLighting
from .logging_utils import RunLogger
from .postprocess import ImagePostprocessor
from .scene_builder import SceneBuilder
from .textures import TextureCache
from .types import CameraData, EnvironmentData, MeshData, RenderImage


@dataclass
class PreparedAssets:
    meshes: list[MeshData]
    lights: list[tuple[torch.Tensor, torch.Tensor]]
    env: EnvironmentData | None
    ibl: ImageBasedLighting | None
    center: np.ndarray
    radius: float


@dataclass
class PreparedScene:
    meshes: list[MeshData]
    camera: CameraData
    lights: list[tuple[torch.Tensor, torch.Tensor]]
    ibl: ImageBasedLighting | None
    bg_rgb: torch.Tensor
    bg_alpha: torch.Tensor


class RenderModeRegistry:
    def __init__(self, renderer: RenderModeRenderer):
        self.renderer = renderer
        self._modes: Dict[str, None] = {name: None for name in renderer.SUPPORTED_MODES}

    def render(self, name: str, *args, **kwargs):
        if name not in self._modes:
            raise ValueError(f"Unsupported render mode: {name}")
        return self.renderer.render_mode(name, *args, **kwargs)


class SceneRenderer:
    def __init__(
        self,
        config: RenderConfig,
        device: torch.device | None = None,
        environment_service: EnvironmentService | None = None,
        logger: RunLogger | None = None,
    ):
        self.config = config
        self.device = torch.device("cuda") if device is None else torch.device(device)
        self.logger = logger
        self.glctx = dr.RasterizeCudaContext(device=self.device)
        # Mesh/material textures are renderer-local so batch jobs can clear them aggressively between meshes.
        self.cache = TextureCache(self.device)
        self.scene_builder = SceneBuilder(self.cache, self.device)
        self.scene_builder.configure_geometry_preprocess(config)
        self.environment = (
            EnvironmentService(TextureCache(self.device, max_file_entries=4))
            if environment_service is None
            else environment_service
        )
        self.geometry = GeometryPassRenderer(self.glctx, config)
        self.compositor = LayerCompositor()
        self.postprocessor = ImagePostprocessor(config)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)

    def clear_texture_cache(self) -> None:
        self.cache.clear()

    def prepare_assets(self, input_path: pathlib.Path) -> PreparedAssets:
        meshes = self.scene_builder.load_meshes(input_path)
        if not meshes:
            raise RuntimeError(f"No renderable meshes found in {input_path}")
        pbr_count = sum(mesh.material.workflow == "pbr" for mesh in meshes)
        self._log(f"Loaded {len(meshes)} mesh(es): {pbr_count} PBR, {len(meshes) - pbr_count} diffuse")
        center, radius = scene_bounds(meshes)
        lights = self.scene_builder.build_lights(self.config.light_intensity)
        env = self.environment.build(self.config)
        ibl = ImageBasedLighting(env, self.config.env_diffuse_samples, self.device) if env is not None else None
        return PreparedAssets(meshes=meshes, lights=lights, env=env, ibl=ibl, center=center, radius=radius)

    def prepare_view(self, assets: PreparedAssets, config: RenderConfig | None = None) -> PreparedScene:
        current_config = self.config if config is None else config
        camera = self.scene_builder.build_camera_from_bounds(assets.center, assets.radius, current_config)
        bg_rgb, bg_alpha = self.environment.render_background(camera, current_config, assets.env, self.device)
        return PreparedScene(
            meshes=assets.meshes,
            camera=camera,
            lights=assets.lights,
            ibl=assets.ibl,
            bg_rgb=bg_rgb,
            bg_alpha=bg_alpha,
        )

    def prepare_scene(self, input_path: pathlib.Path) -> PreparedScene:
        return self.prepare_view(self.prepare_assets(input_path))

    def _build_render_registry(self, prepared: PreparedScene) -> RenderModeRegistry:
        return RenderModeRegistry(RenderModeRenderer(self.config, prepared.lights, prepared.ibl, self.geometry, self.compositor))

    def _finalize_mesh_layers(self, mesh_layers: list[RenderImage], alpha_mode: str) -> list[RenderImage]:
        if len(mesh_layers) == 2 and self.config.double_sided_depth_peels <= 1:
            return [self.compositor.merge_double_sided(mesh_layers[0], mesh_layers[1], alpha_mode)]
        return list(mesh_layers)

    def _postprocess_layers(self, prepared: PreparedScene, layer_images: Sequence[RenderImage], render_mode: str):
        final_rgb, final_alpha = self.compositor.composite_mesh_layers(prepared.bg_rgb, prepared.bg_alpha, layer_images)
        return self.postprocessor.postprocess(final_rgb, final_alpha, render_mode=render_mode)

    def render_prepared_modes(
        self,
        prepared: PreparedScene,
        render_modes: Sequence[str],
    ) -> dict[str, np.ndarray]:
        modes = tuple(render_modes)
        registry = self._build_render_registry(prepared)
        layer_images_by_mode: dict[str, list[RenderImage]] = {mode: [] for mode in modes}
        for mesh in prepared.meshes:
            geometry_layers = self.geometry.render_geometry_pass(mesh, prepared.camera)
            if not geometry_layers:
                continue
            mesh_layers_by_mode: dict[str, list[RenderImage]] = {mode: [] for mode in modes}
            for layer in geometry_layers:
                for mode in modes:
                    mesh_layers_by_mode[mode].append(registry.render(mode, layer, prepared.camera))
            for mode, mesh_layers in mesh_layers_by_mode.items():
                layer_images_by_mode[mode].extend(self._finalize_mesh_layers(mesh_layers, mesh.material.alpha_mode))
        return {
            mode: self._postprocess_layers(prepared, layer_images_by_mode[mode], mode)
            for mode in modes
        }

    def render_prepared(self, prepared: PreparedScene, render_mode: str | None = None):
        mode = self.config.render_mode if render_mode is None else render_mode
        return self.render_prepared_modes(prepared, (mode,))[mode]

    def render(self, input_path: pathlib.Path, render_mode: str | None = None, prepared: PreparedScene | None = None):
        scene = self.prepare_scene(input_path) if prepared is None else prepared
        return self.render_prepared(scene, render_mode=render_mode)

    def save_image(self, image, output_path: pathlib.Path) -> pathlib.Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.postprocessor.save(output_path, image)
        self._log(f"Saved render to {output_path}")
        if self.config.display:
            util.display_image(image[..., :3], size=self.config.resolution, title=str(output_path))
        return output_path

    def render_to_file(
        self,
        output_path: pathlib.Path | None = None,
        render_mode: str | None = None,
        prepared: PreparedScene | None = None,
    ) -> pathlib.Path:
        image = self.render(pathlib.Path(self.config.input), render_mode=render_mode, prepared=prepared)
        return self.save_image(image, pathlib.Path(self.config.output) if output_path is None else output_path)
