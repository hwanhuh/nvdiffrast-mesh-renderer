import os
import pathlib
from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
import torch

import nvdiffrast.torch as dr

from . import util

from .beauty import RenderModeRenderer
from .compositor import LayerCompositor, LayerStack
from .config import RenderConfig
from .environment import EnvironmentService
from .geometry_pass import GeometryPassRenderer
from .geometry_utils import scene_bounds
from .image_io import to_numpy_image
from .ibl import ImageBasedLighting
from .logging_utils import RunLogger
from .postprocess import ImagePostprocessor
from .scene_builder import PreloadedSceneAsset, PreloadedSceneSummary, SceneBuilder
from .textures import TextureCache
from .types import CameraData, EnvironmentData, MeshData, RenderImage

STREAM_LAYER_CAP = max(int(os.environ.get('NVDIFFRAST_STREAM_LAYER_CAP', '16')), 1)
STREAM_VIEW_BATCH_SIZE = max(int(os.environ.get('NVDIFFRAST_STREAM_VIEW_BATCH_SIZE', '8')), 1)
STREAM_CACHE_CLEAR_INTERVAL = max(int(os.environ.get('NVDIFFRAST_STREAM_CACHE_CLEAR_INTERVAL', '4')), 1)
STREAM_MIN_MESHES = max(int(os.environ.get('NVDIFFRAST_STREAM_MIN_MESHES', '64')), 0)
STREAM_MIN_VERTICES = max(int(os.environ.get('NVDIFFRAST_STREAM_MIN_VERTICES', '750000')), 0)
STREAM_MIN_FACES = max(int(os.environ.get('NVDIFFRAST_STREAM_MIN_FACES', '750000')), 0)


@dataclass
class PreparedAssets:
    meshes: list[MeshData] | None
    preloaded_scene: PreloadedSceneAsset | None
    lights: list[tuple[torch.Tensor, torch.Tensor]]
    env: EnvironmentData | None
    ibl: ImageBasedLighting | None
    center: np.ndarray
    radius: float
    mesh_count: int
    pbr_count: int


@dataclass
class PreparedScene:
    meshes: list[MeshData] | None
    preloaded_scene: PreloadedSceneAsset | None
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
        self.scene_builder = SceneBuilder(self.cache, self.device, texture_map_max_size=config.texture_map_max_size)
        self.environment = (
            EnvironmentService(TextureCache(self.device, max_file_entries=4))
            if environment_service is None
            else environment_service
        )
        self.geometry = GeometryPassRenderer(self.glctx, config)
        self.compositor = LayerCompositor()
        self.postprocessor = ImagePostprocessor(config, device=self.device)

    def _log(self, message: str) -> None:
        if self.logger is not None:
            self.logger.log(message)

    def clear_texture_cache(self) -> None:
        self.cache.clear()

    def _set_mesh_texture_mipmaps_enabled(self, enabled: bool) -> None:
        self.cache.set_build_mipmaps(enabled)

    def _should_stream_summary(self, summary: PreloadedSceneSummary) -> bool:
        if STREAM_MIN_MESHES > 0 and summary.mesh_count >= STREAM_MIN_MESHES:
            return True
        if STREAM_MIN_VERTICES > 0 and summary.vertex_count >= STREAM_MIN_VERTICES:
            return True
        if STREAM_MIN_FACES > 0 and summary.face_count >= STREAM_MIN_FACES:
            return True
        return False

    def should_stream_preloaded_scene(self, preloaded_scene: PreloadedSceneAsset) -> bool:
        return self._should_stream_summary(self.scene_builder.summarize_preloaded(preloaded_scene))

    def _shared_preloaded_scene(self, prepared_scenes: Sequence[PreparedScene]) -> PreloadedSceneAsset | None:
        if not prepared_scenes:
            return None
        shared = prepared_scenes[0].preloaded_scene
        if shared is None:
            return None
        for prepared in prepared_scenes[1:]:
            other = prepared.preloaded_scene
            if other is None:
                return None
            if other is not shared and other.path != shared.path:
                return None
        return shared

    def _flush_streaming_cache(self) -> None:
        self.clear_texture_cache()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def prepare_assets(
        self,
        input_path: pathlib.Path,
        preloaded_scene: PreloadedSceneAsset | None = None,
        force_streaming: bool | None = None,
    ) -> PreparedAssets:
        if preloaded_scene is None:
            self._set_mesh_texture_mipmaps_enabled(True)
            meshes = self.scene_builder.load_meshes(input_path)
            if not meshes:
                raise RuntimeError(f"No renderable meshes found in {input_path}")
            pbr_count = sum(mesh.material.workflow == "pbr" for mesh in meshes)
            self._log(f"Loaded {len(meshes)} mesh(es): {pbr_count} PBR, {len(meshes) - pbr_count} diffuse")
            center, radius = scene_bounds(meshes)
            mesh_count = len(meshes)
            preloaded_for_render = None
        else:
            summary = self.scene_builder.summarize_preloaded(preloaded_scene)
            if summary.mesh_count == 0:
                raise RuntimeError(f"No renderable meshes found in {input_path}")
            use_streaming = self._should_stream_summary(summary) if force_streaming is None else bool(force_streaming)
            self._set_mesh_texture_mipmaps_enabled(not use_streaming)
            if use_streaming:
                self._log(
                    f"Loaded {summary.mesh_count} mesh(es): {summary.pbr_count} PBR, "
                    f"{summary.mesh_count - summary.pbr_count} diffuse using streaming path "
                    f"(vertices={summary.vertex_count}, faces={summary.face_count}, "
                    f"texture_cap={preloaded_scene.texture_map_max_size or 'full'}, material_mips=off, "
                    f"antialias={'on' if self.config.antialias else 'off'}, depth_peels={self.config.double_sided_depth_peels})"
                )
                center, radius = self.scene_builder.bounds_from_preloaded(preloaded_scene)
                meshes = None
                preloaded_for_render = preloaded_scene
            else:
                meshes = self.scene_builder.load_meshes_from_preloaded(preloaded_scene)
                self._log(
                    f"Loaded {summary.mesh_count} mesh(es): {summary.pbr_count} PBR, "
                    f"{summary.mesh_count - summary.pbr_count} diffuse using in-memory fast path "
                    f"(vertices={summary.vertex_count}, faces={summary.face_count}, "
                    f"texture_cap={preloaded_scene.texture_map_max_size or 'full'}, material_mips=on, "
                    f"antialias={'on' if self.config.antialias else 'off'}, depth_peels={self.config.double_sided_depth_peels})"
                )
                center, radius = scene_bounds(meshes)
                preloaded_for_render = None
            mesh_count = summary.mesh_count
            pbr_count = summary.pbr_count
        lights = self.scene_builder.build_lights(self.config.light_intensity)
        env = self.environment.build(self.config)
        ibl = ImageBasedLighting(env, self.config.env_diffuse_samples, self.device) if env is not None else None
        return PreparedAssets(
            meshes=meshes,
            preloaded_scene=preloaded_for_render,
            lights=lights,
            env=env,
            ibl=ibl,
            center=center,
            radius=radius,
            mesh_count=mesh_count,
            pbr_count=pbr_count,
        )

    def prepare_view(self, assets: PreparedAssets, config: RenderConfig | None = None, light_seed: int | None = None) -> PreparedScene:
        current_config = self.config if config is None else config
        camera = (
            self.scene_builder.build_camera(assets.meshes, current_config)
            if assets.meshes is not None
            else self.scene_builder.build_camera_from_bounds(assets.center, assets.radius, current_config)
        )
        lights = assets.lights if light_seed is None else self.scene_builder.build_view_seeded_lights(
            current_config.light_intensity,
            camera_direction=(-camera.forward).detach().cpu().numpy(),
            view_seed=light_seed,
        )
        bg_rgb, bg_alpha = self.environment.render_background(camera, current_config, assets.env, self.device)
        return PreparedScene(
            meshes=assets.meshes,
            preloaded_scene=assets.preloaded_scene,
            camera=camera,
            lights=lights,
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

    def _postprocess_layer_stack(self, prepared: PreparedScene, layer_stack: LayerStack, render_mode: str):
        final_rgb, final_alpha = self.compositor.composite_layer_stack(prepared.bg_rgb, prepared.bg_alpha, layer_stack)
        return self.postprocessor.postprocess(final_rgb, final_alpha, render_mode=render_mode)

    def _render_prepared_modes_streaming(
        self,
        prepared: PreparedScene,
        render_modes: Sequence[str],
    ) -> dict[str, np.ndarray]:
        return self._render_prepared_modes_streaming_batch([prepared], render_modes)[0]

    def _render_prepared_modes_streaming_batch(
        self,
        prepared_scenes: Sequence[PreparedScene],
        render_modes: Sequence[str],
    ) -> list[dict[str, np.ndarray]]:
        prepared_batch = list(prepared_scenes)
        if not prepared_batch:
            return []
        shared_scene = self._shared_preloaded_scene(prepared_batch)
        if shared_scene is None:
            return [self.render_prepared_modes(prepared, render_modes) for prepared in prepared_batch]
        modes = tuple(render_modes)
        registries = [self._build_render_registry(prepared) for prepared in prepared_batch]
        layer_stacks_by_scene = [
            {
                mode: self.compositor.create_layer_stack(prepared.bg_rgb, prepared.bg_alpha, capacity=STREAM_LAYER_CAP)
                for mode in modes
            }
            for prepared in prepared_batch
        ]
        self._log(
            f"Streaming scene mesh-by-mesh across {len(prepared_batch)} view(s) with "
            f"layer_cap={STREAM_LAYER_CAP}, cache_clear_interval={STREAM_CACHE_CLEAR_INTERVAL}"
        )
        try:
            for mesh_index, entry in enumerate(self.scene_builder.iter_preloaded_mesh_entries(shared_scene), start=1):
                mesh = None
                try:
                    mesh = self.scene_builder.load_mesh_from_entry(entry)
                    for prepared_index, (prepared, registry) in enumerate(zip(prepared_batch, registries)):
                        geometry_layers = self.geometry.render_geometry_pass(mesh, prepared.camera)
                        if not geometry_layers:
                            continue
                        mesh_layers_by_mode = {mode: [] for mode in modes}
                        for layer in geometry_layers:
                            for mode in modes:
                                mesh_layers_by_mode[mode].append(registry.render(mode, layer, prepared.camera))
                        for mode, mesh_layers in mesh_layers_by_mode.items():
                            finalized = self._finalize_mesh_layers(mesh_layers, mesh.material.alpha_mode)
                            layer_stacks_by_scene[prepared_index][mode] = self.compositor.accumulate_layer_stack(
                                layer_stacks_by_scene[prepared_index][mode],
                                finalized,
                            )
                finally:
                    mesh = None
                    if mesh_index % STREAM_CACHE_CLEAR_INTERVAL == 0:
                        self._flush_streaming_cache()
        finally:
            self._flush_streaming_cache()
        return [
            {
                mode: self._postprocess_layer_stack(prepared, layer_stacks_by_scene[index][mode], mode)
                for mode in modes
            }
            for index, prepared in enumerate(prepared_batch)
        ]

    def render_prepared_modes(
        self,
        prepared: PreparedScene,
        render_modes: Sequence[str],
    ) -> dict[str, np.ndarray]:
        if prepared.preloaded_scene is not None:
            return self._render_prepared_modes_streaming(prepared, render_modes)
        modes = tuple(render_modes)
        registry = self._build_render_registry(prepared)
        layer_images_by_mode: dict[str, list[RenderImage]] = {mode: [] for mode in modes}
        assert prepared.meshes is not None
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

    def render_prepared_batch(self, prepared_scenes: Sequence[PreparedScene], render_mode: str | None = None) -> list[np.ndarray]:
        prepared_batch = list(prepared_scenes)
        if not prepared_batch:
            return []
        mode = self.config.render_mode if render_mode is None else render_mode
        if self._shared_preloaded_scene(prepared_batch) is None:
            return [self.render_prepared(prepared, render_mode=mode) for prepared in prepared_batch]
        outputs: list[np.ndarray] = []
        for start in range(0, len(prepared_batch), STREAM_VIEW_BATCH_SIZE):
            chunk = prepared_batch[start: start + STREAM_VIEW_BATCH_SIZE]
            outputs.extend(result[mode] for result in self._render_prepared_modes_streaming_batch(chunk, (mode,)))
        return outputs

    def render(self, input_path: pathlib.Path, render_mode: str | None = None, prepared: PreparedScene | None = None):
        scene = self.prepare_scene(input_path) if prepared is None else prepared
        return self.render_prepared(scene, render_mode=render_mode)

    def save_image(self, image, output_path: pathlib.Path) -> pathlib.Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.postprocessor.save(output_path, image)
        self._log(f"Saved render to {output_path}")
        if self.config.display:
            util.display_image(to_numpy_image(image)[..., :3], size=self.config.resolution, title=str(output_path))
        return output_path

    def render_to_file(
        self,
        output_path: pathlib.Path | None = None,
        render_mode: str | None = None,
        prepared: PreparedScene | None = None,
    ) -> pathlib.Path:
        image = self.render(pathlib.Path(self.config.input), render_mode=render_mode, prepared=prepared)
        return self.save_image(image, pathlib.Path(self.config.output) if output_path is None else output_path)
