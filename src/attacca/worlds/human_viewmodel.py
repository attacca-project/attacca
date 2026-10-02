"""Renderer-side capture of the first-person hand mask, depth and entity-ID image for each observation, using the engine patch built from human_engine/."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import zipfile

import numpy as np


RENDERER_VIEWMODEL_MASK_CONTRACT = (
    "minecraft_java_1_16_5_same_tick_hand_only_alpha/v1")
RENDERER_VIEWMODEL_INFO_FIELD = "xbench_renderer_viewmodel_mask"
RENDERER_VIEWMODEL_DEPTH_FIELD = "xbench_renderer_viewmodel_depth"
RENDERER_ENTITY_ID_FIELD = "xbench_renderer_entity_id"
RENDERER_ENTITY_ID_VALID_FIELD = "xbench_renderer_entity_id_valid"
RENDERER_SCENE_ALIVE_CLASSES_FIELD = "xbench_renderer_scene_alive_classes_mask"
RENDERER_UNIQUE_WOLF_HEALTH_FIELD = "xbench_renderer_unique_wolf_health"
RENDERER_ENTITY_ID_MAX_SEMANTIC_CODE = 29
RENDERER_ENGINE_BUILD_VERSION = "renderer_transport_v7"
RENDERER_VIEWMODEL_ROW_RUNS_FIELD = "renderer_viewmodel_mask_runs_yx"
RENDERER_VIEWMODEL_ROW_SHAPE_FIELD = "renderer_viewmodel_mask_shape"
RENDERER_VIEWMODEL_ROW_COUNT_FIELD = "renderer_viewmodel_mask_pixel_count"


def _renderer_mask_engine_root() -> Path:
    from minestudio.simulator.minerl.env.malmo import InstanceManager

    current = Path(InstanceManager.MINECRAFT_DIR).resolve()
    marker = current / "XBENCH_RENDERER_MASK_ENGINE.json"
    if marker.is_file():
        return current
    source_jar = current / "build" / "libs" / "mcprec-6.13.jar"
    if not source_jar.is_file():
        raise RuntimeError(f"MineStudio engine jar is missing: {source_jar}")
    java_dir = Path(__file__).with_name("human_engine")
    java_sources = sorted(java_dir.glob("*.java"))
    if len(java_sources) != 2:
        raise RuntimeError("renderer mask engine requires exactly two Java sources")
    identity = hashlib.sha256()
    identity.update(RENDERER_ENGINE_BUILD_VERSION.encode())
    stat = source_jar.stat()
    identity.update(f"{source_jar}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    for source in java_sources:
        identity.update(source.name.encode())
        identity.update(source.read_bytes())
    digest = identity.hexdigest()[:20]
    cache_parent = Path(tempfile.gettempdir()) / "xbench_renderer_mask_engine"
    target_root = cache_parent / digest
    target_jar = target_root / "build" / "libs" / source_jar.name
    target_marker = target_root / marker.name
    cache_parent.mkdir(parents=True, exist_ok=True)
    lock_path = cache_parent / f"{digest}.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if target_jar.is_file() and target_marker.is_file():
            return target_root
        build_root = Path(tempfile.mkdtemp(
            prefix=f"{digest}.building.", dir=cache_parent))
        try:
            classes = build_root / "classes"
            classes.mkdir()
            javac = shutil.which("javac")
            java = shutil.which("java")
            if not javac or not java:
                raise RuntimeError(
                    "renderer mask engine requires javac and java")
            subprocess.run([
                javac, "-source", "8", "-target", "8", "-cp",
                str(source_jar), "-d", str(classes),
                *map(str, java_sources),
            ], check=True)
            subprocess.run([
                java, "-cp", f"{source_jar}:{classes}",
                "RendererMaskJarPatcher", str(source_jar), str(classes),
            ], check=True)
            staged_jar = build_root / source_jar.name
            shutil.copy2(source_jar, staged_jar)
            helper_classes = sorted(
                (classes / "com/minerl/multiagent/recorder").glob(
                    "XBenchViewmodelCapture*.class"))
            helper_entries = [
                str(path.relative_to(classes)) for path in helper_classes]
            if len(helper_entries) < 5:
                raise RuntimeError(
                    "renderer entity-ID helper inner classes are missing")
            replace_entries = [
                "com/minerl/multiagent/recorder/PlayRecorder.class",
                *helper_entries,
                "net/minecraft/client/renderer/GameRenderer.class",
            ]
            rewritten_jar = build_root / (source_jar.name + ".rewrite")
            with zipfile.ZipFile(staged_jar) as src_archive, \
                    zipfile.ZipFile(rewritten_jar, "w",
                                    compression=zipfile.ZIP_DEFLATED) as dst_archive:
                pending = dict.fromkeys(replace_entries)
                for info in src_archive.infolist():
                    if info.filename in pending:
                        dst_archive.write(
                            classes / info.filename, info.filename,
                            compress_type=zipfile.ZIP_DEFLATED)
                        pending.pop(info.filename)
                    else:
                        dst_archive.writestr(info, src_archive.read(info.filename))
                for name in pending:
                    dst_archive.write(classes / name, name,
                                      compress_type=zipfile.ZIP_DEFLATED)
            rewritten_jar.replace(staged_jar)
            with zipfile.ZipFile(staged_jar) as archive:
                helper = archive.read(
                    "com/minerl/multiagent/recorder/XBenchViewmodelCapture.class")
                recorder = archive.read(
                    "com/minerl/multiagent/recorder/PlayRecorder.class")
                renderer = archive.read(
                    "net/minecraft/client/renderer/GameRenderer.class")
                archived_helpers = {
                    name for name in archive.namelist()
                    if name.startswith(
                        "com/minerl/multiagent/recorder/"
                        "XBenchViewmodelCapture")
                    and name.endswith(".class")}
            if (b"XBenchViewmodelCapture" not in recorder
                    or b"XBenchViewmodelCapture" not in renderer
                    or b"glReadPixels" not in helper
                    or set(helper_entries) != archived_helpers):
                raise RuntimeError("renderer mask engine bytecode audit failed")
            target_jar.parent.mkdir(parents=True, exist_ok=True)
            staged_jar.replace(target_jar)
            target_marker.write_text(json.dumps({
                "contract": RENDERER_VIEWMODEL_MASK_CONTRACT,
                "source_jar": str(source_jar),
                "source_size": stat.st_size,
                "cache_identity": digest,
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        finally:
            shutil.rmtree(build_root, ignore_errors=True)
    return target_root


def alpha_u8_to_viewmodel_mask(alpha_u8, *, shape=(360, 640)) -> np.ndarray:
    alpha = np.asarray(alpha_u8)
    if alpha.shape != tuple(shape) or alpha.dtype != np.uint8:
        raise ValueError(
            f"renderer alpha mask must be uint8{tuple(shape)}, got "
            f"{alpha.dtype}{alpha.shape}")
    if np.any((alpha != 254) & (alpha != 255)):
        raise ValueError("renderer alpha mask must contain only 254/255")
    return np.ascontiguousarray(alpha == 254, dtype=np.uint8)


def decode_entity_id_transport(id_rgb, valid_u8, *, shape=(360, 640)):
    rgb = np.asarray(id_rgb)
    valid = np.asarray(valid_u8)
    expected_rgb = tuple(shape) + (3,)
    if rgb.dtype != np.uint8 or rgb.shape != expected_rgb:
        raise ValueError(
            f"renderer entity IDs must be uint8{expected_rgb}, got "
            f"{rgb.dtype}{rgb.shape}")
    if valid.dtype != np.uint8 or valid.shape != tuple(shape):
        raise ValueError(
            f"renderer entity validity must be uint8{tuple(shape)}, got "
            f"{valid.dtype}{valid.shape}")
    markers = np.unique(valid)
    if markers.size != 1 or int(markers[0]) not in (253, 254):
        raise ValueError(
            "renderer entity validity must be uniformly 253 or 254")
    if int(markers[0]) == 253:
        return np.zeros(tuple(shape), dtype=np.uint32), False
    semantic = rgb[..., 0].astype(np.uint32)
    entity_id = ((rgb[..., 1].astype(np.uint32) << 8)
                 | rgb[..., 2].astype(np.uint32))
    if np.any((semantic == 0) != (entity_id == 0)):
        raise ValueError(
            "renderer entity background/identity channels disagree")
    if np.any(semantic > RENDERER_ENTITY_ID_MAX_SEMANTIC_CODE):
        raise ValueError("renderer entity semantic code is unsupported")
    return np.ascontiguousarray((semantic << 16) | entity_id), True


def decode_scene_alive_classes_transport(value_rgba, *, shape=(360, 640)) -> int:
    value = np.asarray(value_rgba)
    expected = tuple(shape) + (4,)
    if value.dtype != np.uint8 or value.shape != expected:
        raise ValueError(
            f"renderer scene alive classes must be uint8{expected}, got "
            f"{value.dtype}{value.shape}")
    channels = []
    for channel in range(4):
        markers = np.unique(value[..., channel])
        if markers.size != 1:
            raise ValueError("renderer scene alive-class mask must be uniform")
        channels.append(int(markers[0]))
    return (channels[0] | (channels[1] << 8) | (channels[2] << 16)
            | (channels[3] << 24))


def decode_unique_wolf_health_transport(value_rgba, *, shape=(360, 640)) -> float:
    value = np.asarray(value_rgba)
    expected = tuple(shape) + (4,)
    if value.dtype != np.uint8 or value.shape != expected:
        raise ValueError(
            f"renderer wolf Health must be uint8{expected}, got "
            f"{value.dtype}{value.shape}")
    first = np.ascontiguousarray(value[0, 0])
    if np.any(value != first):
        raise ValueError("renderer wolf Health transport must be uniform")
    return float(first.view("<f4")[0])


def encode_mask_runs_yx(mask) -> list[list[int]]:
    value = np.asarray(mask)
    if value.ndim != 2 or value.dtype != np.uint8:
        raise ValueError("viewmodel mask must be a 2-D uint8 array")
    if np.any((value != 0) & (value != 1)):
        raise ValueError("viewmodel mask must be binary")
    runs: list[list[int]] = []
    for y, row in enumerate(value):
        padded = np.pad(row, (1, 1), constant_values=0)
        edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
        for x0, x1 in edges.reshape(-1, 2):
            runs.append([int(y), int(x0), int(x1)])
    return runs


def decode_mask_runs_yx(runs, shape) -> np.ndarray:
    if (not isinstance(shape, Sequence) or isinstance(shape, (str, bytes))
            or len(shape) != 2):
        raise ValueError("viewmodel mask shape must contain height,width")
    height, width = shape
    if (isinstance(height, bool) or isinstance(width, bool)
            or not isinstance(height, int) or not isinstance(width, int)
            or height <= 0 or width <= 0):
        raise ValueError("viewmodel mask shape must contain positive integers")
    if not isinstance(runs, list):
        raise ValueError("viewmodel mask runs must be a list")
    mask = np.zeros((height, width), dtype=np.uint8)
    previous = (-1, -1)
    for index, run in enumerate(runs):
        if (not isinstance(run, list) or len(run) != 3
                or any(isinstance(item, bool) or not isinstance(item, int)
                       for item in run)):
            raise ValueError(f"viewmodel mask run {index} is malformed")
        y, x0, x1 = run
        if not (0 <= y < height and 0 <= x0 < x1 <= width):
            raise ValueError(f"viewmodel mask run {index} is outside its shape")
        if (y, x0) <= previous:
            raise ValueError("viewmodel mask runs must be strictly ordered")
        if mask[y, x0:x1].any():
            raise ValueError("viewmodel mask runs overlap")
        mask[y, x0:x1] = 1
        previous = (y, x0)
    return mask


def renderer_mask_row_fields(mask) -> dict:
    value = np.asarray(mask, dtype=np.uint8)
    if value.shape != (360, 640) or np.any((value != 0) & (value != 1)):
        raise ValueError("renderer mask must be binary uint8[360,640]")
    return {
        RENDERER_VIEWMODEL_ROW_SHAPE_FIELD: [360, 640],
        RENDERER_VIEWMODEL_ROW_RUNS_FIELD: encode_mask_runs_yx(value),
        RENDERER_VIEWMODEL_ROW_COUNT_FIELD: int(value.sum()),
    }


def renderer_mask_from_row(row: Mapping) -> np.ndarray:
    if not isinstance(row, Mapping):
        raise ValueError("renderer mask row must be a mapping")
    mask = decode_mask_runs_yx(
        row.get(RENDERER_VIEWMODEL_ROW_RUNS_FIELD),
        row.get(RENDERER_VIEWMODEL_ROW_SHAPE_FIELD))
    count = row.get(RENDERER_VIEWMODEL_ROW_COUNT_FIELD)
    if (isinstance(count, bool) or not isinstance(count, int)
            or count != int(mask.sum())):
        raise ValueError("renderer mask pixel count mismatch")
    return mask


def install_renderer_viewmodel_observation() -> None:
    from minestudio.simulator import MinecraftSim
    from minestudio.simulator.minerl.env.malmo import InstanceManager
    from minestudio.simulator.minerl.herobraine.env_specs.human_survival_specs import (
        HumanSurvival,
    )
    from minestudio.simulator.minerl.herobraine.hero.handlers.agent.observations.pov import (
        POVObservation,
    )
    from minestudio.simulator.minerl.herobraine.hero import spaces

    if getattr(HumanSurvival, "_xbench_renderer_rgba_installed", False):
        return
    InstanceManager.MINECRAFT_DIR = str(_renderer_mask_engine_root())

    class RendererMaskPOVObservation(POVObservation):
        def __init__(self, video_resolution):
            super().__init__(video_resolution, include_depth=True)
            self.video_depth = 20
            self.space = spaces.Box(
                0, 255, list(video_resolution)[::-1] + [20], dtype=np.uint8)

        def xml_template(self) -> str:
            return """
                <VideoProducer want_depth="true">
                    <Width>{{ video_width }}</Width>
                    <Height>{{ video_height }}</Height>
                    <DepthScaling min="0" max="1" autoscale="false"/>
                </VideoProducer>"""

        def __or__(self, other):
            if (isinstance(other, POVObservation)
                    and other.video_resolution == self.video_resolution
                    and bool(other.include_depth)):
                return RendererMaskPOVObservation(self.video_resolution)
            raise ValueError("Incompatible renderer-mask POV observables")

    original_create = HumanSurvival.create_observables
    original_wrap = MinecraftSim._wrap_obs_info

    def create_observables_rgba(self):
        values = original_create(self)
        replaced = 0
        output = []
        for value in values:
            if isinstance(value, POVObservation):
                output.append(RendererMaskPOVObservation(value.video_resolution))
                replaced += 1
            else:
                output.append(value)
        if replaced != 1:
            raise RuntimeError(
                f"renderer RGBA seam expected one POV observable, found {replaced}")
        return output

    def wrap_obs_info_rgb_only(self, obs, info):
        if not isinstance(obs, Mapping) or "pov" not in obs:
            raise RuntimeError("renderer RGBA observation lacks pov")
        rgbad = np.asarray(obs["pov"])
        height = int(self.render_size[1])
        width = int(self.render_size[0])
        expected = (height, width, 20)
        if rgbad.dtype != np.uint8 or rgbad.shape != expected:
            raise RuntimeError(
                f"renderer RGBAD POV must be uint8{expected}, got "
                f"{rgbad.dtype}{rgbad.shape}")
        mask = alpha_u8_to_viewmodel_mask(rgbad[..., 3], shape=(height, width))
        depth = (np.ascontiguousarray(rgbad[..., 4:8])
                 .view("<f4").reshape(height, width))
        entity_ids, entity_ids_valid = decode_entity_id_transport(
            rgbad[..., 8:11], rgbad[..., 11], shape=(height, width))
        alive_classes_mask = decode_scene_alive_classes_transport(
            rgbad[..., 12:16], shape=(height, width))
        wolf_health = decode_unique_wolf_health_transport(
            rgbad[..., 16:20], shape=(height, width))
        rgb_obs = dict(obs)
        rgb_obs["pov"] = np.ascontiguousarray(rgbad[..., :3])
        wrapped_obs, wrapped_info = original_wrap(self, rgb_obs, info)
        wrapped_info[RENDERER_VIEWMODEL_INFO_FIELD] = mask
        wrapped_info[RENDERER_VIEWMODEL_DEPTH_FIELD] = depth
        wrapped_info[RENDERER_ENTITY_ID_FIELD] = entity_ids
        wrapped_info[RENDERER_ENTITY_ID_VALID_FIELD] = int(entity_ids_valid)
        wrapped_info[RENDERER_SCENE_ALIVE_CLASSES_FIELD] = alive_classes_mask
        wrapped_info[RENDERER_UNIQUE_WOLF_HEALTH_FIELD] = wolf_health
        self.info[RENDERER_VIEWMODEL_INFO_FIELD] = mask
        self.info[RENDERER_VIEWMODEL_DEPTH_FIELD] = depth
        self.info[RENDERER_ENTITY_ID_FIELD] = entity_ids
        self.info[RENDERER_ENTITY_ID_VALID_FIELD] = int(entity_ids_valid)
        self.info[RENDERER_SCENE_ALIVE_CLASSES_FIELD] = alive_classes_mask
        self.info[RENDERER_UNIQUE_WOLF_HEALTH_FIELD] = wolf_health
        return wrapped_obs, wrapped_info

    HumanSurvival.create_observables = create_observables_rgba
    MinecraftSim._wrap_obs_info = wrap_obs_info_rgb_only
    HumanSurvival._xbench_renderer_rgba_installed = True
