import pathlib
from dataclasses import dataclass
from typing import Dict

import torch

import nvdiffrast.torch as dr

from . import util

from .beauty import RenderModeRenderer
from .compositor import LayerCompositor
from .config import RenderConfig
from .environment import EnvironmentService
from .geometry_pass import GeometryPassRenderer
from .ibl import ImageBasedLighting
from .postprocess import ImagePostprocessor
from .scene_builder import SceneBuilder
from .textures import TextureCache
from .types import CameraData, EnvironmentData, MeshData


@dataclass
class PreparedAssets:
    meshes: list[MeshData]
    lights: list[tuple[torch.Tensor, torch.Tensor]]
    env: EnvironmentData | None
    ibl: ImageBasedLighting | None


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
    def __init__(self, config: RenderConfig, device: torch.device | None = None):
        self.config = config
        self.device = torch.device("cuda") if device is None else device
        self.glctx = dr.RasterizeCudaContext(device=self.device)
        self.cache = TextureCache(self.device)
        self.scene_builder = SceneBuilder(self.cache, self.device)
        self.environment = EnvironmentService(self.cache)
        self.geometry = GeometryPassRenderer(self.glctx, config)
        self.compositor = LayerCompositor()
        self.postprocessor = ImagePostprocessor(config)

    def prepare_assets(self, input_path: pathlib.Path) -> PreparedAssets:
        meshes = self.scene_builder.load_meshes(input_path)
        if not meshes:
            raise RuntimeError(f"No renderable meshes found in {input_path}")
        pbr_count = sum(mesh.material.workflow == "pbr" for mesh in meshes)
        print(f"Loaded {len(meshes)} mesh(es): {pbr_count} PBR, {len(meshes) - pbr_count} diffuse")
        lights = self.scene_builder.build_lights(self.config.light_intensity)
        env = self.environment.build(self.config)
        ibl = ImageBasedLighting(env, self.config.env_diffuse_samples, self.device) if env is not None else None
        return PreparedAssets(meshes=meshes, lights=lights, env=env, ibl=ibl)

    def prepare_view(self, assets: PreparedAssets, config: RenderConfig | None = None) -> PreparedScene:
        current_config = self.config if config is None else config
        camera = self.scene_builder.build_camera(assets.meshes, current_config)
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

    def render_prepared(self, prepared: PreparedScene, render_mode: str | None = None):
        mode = self.config.render_mode if render_mode is None else render_mode
        registry = RenderModeRegistry(RenderModeRenderer(self.config, prepared.lights, prepared.ibl, self.geometry, self.compositor))
        layer_images = []
        for mesh in prepared.meshes:
            mesh_layers = [registry.render(mode, layer, prepared.camera) for layer in self.geometry.render_geometry_pass(mesh, prepared.camera)]
            if len(mesh_layers) == 2:
                layer_images.append(self.compositor.merge_double_sided(mesh_layers[0], mesh_layers[1], mesh.material.alpha_mode))
            elif mesh_layers:
                layer_images.append(mesh_layers[0])
        final_rgb, final_alpha = self.compositor.composite_mesh_layers(prepared.bg_rgb, prepared.bg_alpha, layer_images)
        return self.postprocessor.postprocess(final_rgb, final_alpha, render_mode=mode)

    def render(self, input_path: pathlib.Path, render_mode: str | None = None, prepared: PreparedScene | None = None):
        scene = self.prepare_scene(input_path) if prepared is None else prepared
        return self.render_prepared(scene, render_mode=render_mode)

    def save_image(self, image, output_path: pathlib.Path) -> pathlib.Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.postprocessor.save(output_path, image)
        print(f"Saved render to {output_path}")
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
