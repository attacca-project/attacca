#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO) + "/src")
if str(REPO / "src/attacca/evaluation") not in sys.path:
    sys.path.insert(0, str(REPO / "src/attacca/evaluation"))

os.environ.setdefault("MINESTUDIO_DIR", str(REPO / ".minestudio"))
os.environ.setdefault("MINESTUDIO_GPU_RENDER", "1")
os.environ.setdefault("RENDER_DEVICES", "0")


PHASE_NAMES = ("EXPLORE", "APPROACH", "INTERACT")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_digest(path: Path) -> str:
    path = Path(path)
    if path.is_file():
        return sha256_file(path)
    if path.is_dir():
        children = sorted(child for child in path.rglob("*") if child.is_file())
        if not children:
            raise ValueError(f"checkpoint directory is empty: {path}")
        digest = hashlib.sha256()
        for child in children:
            digest.update(str(child.relative_to(path)).encode("utf-8"))
            digest.update(b"\0")
            digest.update(sha256_file(child).encode("ascii"))
            digest.update(b"\0")
        return digest.hexdigest()
    raise FileNotFoundError(path)


def atomic_json(path: Path, value: Mapping | Sequence) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite value in JSON audit")
        return value
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "detach"):
        value = value.detach().float().cpu().numpy()
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    raise TypeError(f"unsupported JSON value {type(value)!r}")


def machine_record() -> dict:
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        gpu = "unavailable"
    return {
        "gpu_names": [line for line in gpu.splitlines() if line],
    }


def normalize_name(value: Any) -> str:
    name = str(value or "").lower().strip().split(" ", 1)[0]
    return name.split(":")[-1].split("/")[-1]


def normalized_counter(raw: Any) -> dict[str, int]:
    if not isinstance(raw, Mapping):
        return {}
    output: dict[str, int] = {}
    for key, value in raw.items():
        try:
            if hasattr(value, "detach"):
                value = value.detach().cpu().numpy()
            scalar = int(float(np.asarray(value).reshape(-1)[0]))
        except (TypeError, ValueError, IndexError):
            continue
        name = normalize_name(key)
        if name:
            output[name] = output.get(name, 0) + scalar
    return dict(sorted(output.items()))


def positive_counter_delta(current: Mapping[str, int],
                           initial: Mapping[str, int]) -> dict[str, int]:
    return {
        key: int(current.get(key, 0)) - int(initial.get(key, 0))
        for key in sorted(set(current) | set(initial))
        if int(current.get(key, 0)) - int(initial.get(key, 0)) > 0
    }


def phase_probabilities(latents: Mapping, *,
                        required: bool = True) -> list[float] | None:
    raw = latents.get("phase_logits") if isinstance(latents, Mapping) else None
    if raw is None:
        if not required:
            return None
        raise RuntimeError("checkpoint did not expose phase_logits")
    if hasattr(raw, "detach"):
        raw = raw.detach().float().cpu().numpy()
    logits = np.asarray(raw, dtype=np.float64).reshape(-1)
    if logits.size != 3 or not bool(np.isfinite(logits).all()):
        raise RuntimeError(f"phase_logits shape/value drift: {logits}")
    exp = np.exp(logits - float(logits.max()))
    probability = exp / float(exp.sum())
    return [float(value) for value in probability]


def renderer_surface_distance_at_pixel(
        depth_window: np.ndarray, xy: tuple[int, int], *,
        near: float = 0.05, far: float = 256.0,
        vertical_fov_degrees: float = 70.0) -> float | None:
    depth = np.asarray(depth_window)
    if depth.shape != (360, 640):
        raise ValueError(f"renderer depth must be (360,640), got {depth.shape}")
    x, y = xy
    if not (0 <= x < 640 and 0 <= y < 360):
        raise ValueError("depth pixel is outside the native frame")
    value = float(depth[y, x])
    if not math.isfinite(value) or value < 0.0 or value >= 1.0:
        return None
    n, f = float(near), float(far)
    m10 = -(f + n) / (f - n)
    m14 = -2.0 * f * n / (f - n)
    ndc = 2.0 * value - 1.0
    denominator = ndc + m10
    if denominator == 0.0:
        return None
    z_view = m14 / denominator
    if not math.isfinite(z_view) or z_view <= 0.0:
        return None
    focal = 180.0 / math.tan(math.radians(vertical_fov_degrees) / 2.0)
    ax = (float(x) + 0.5 - 320.0) / focal
    ay = (float(y) + 0.5 - 180.0) / focal
    return float(z_view * math.sqrt(1.0 + ax * ax + ay * ay))


def press_hotbar1(world) -> None:
    from minestudio.utils.vpt_lib.action_mapping import Buttons
    factored = {
        "buttons": np.zeros((1, len(Buttons.ALL)), np.int64),
        "camera": np.array([[world._null_bin, world._null_bin]], np.int64),
    }
    factored["buttons"][0, Buttons.ALL.index("hotbar.1")] = 1
    agent_action = world._mapper.from_factored(factored)
    action = {
        "buttons": np.asarray(agent_action["buttons"]).reshape(1),
        "camera": np.asarray(agent_action["camera"]).reshape(1),
    }
    world.obs, _, _, _, world.info = world.sim.step(action)


def voxel_types(world, x0: int, x1: int, y0: int, y1: int,
                z0: int, z1: int) -> list[str]:
    ax, ay, az = world.get_pos()
    bx, by, bz = int(math.floor(ax)), int(math.floor(ay)), int(math.floor(az))
    action = world.sim.noop_action()
    action["voxels"] = np.array([
        x0 - bx, x1 + 1 - bx,
        y0 - by, y1 + 1 - by,
        z0 - bz, z1 + 1 - bz,
    ], dtype=np.int32)
    world.obs, _, _, _, world.info = world.sim.step(action)
    return [str(row.get("type", ""))
            for row in (world.info.get("voxels") or [])]


def boot(seed: int, *, runtime_overlay: Path):
    from attacca.evaluation.policy import boot_world
    from attacca.evaluation.mine_scene import Stager
    from attacca.worlds.human_viewmodel import install_renderer_viewmodel_observation

    install_renderer_viewmodel_observation()
    for attempt in range(1, 4):
        world = None
        try:
            world = boot_world(
                int(seed), biome="plains", action_type="agent",
                runtime_overlay=runtime_overlay,
            )
            stager = Stager(world)
            stager.prep_world()
            world.cmd("/gamerule doMobLoot false")
            world.cmd("/gamerule sendCommandFeedback false")
            return world, stager
        except BaseException as exc:
            if world is not None:
                try:
                    world.close()
                except BaseException:
                    pass
            broken_pipe = (
                isinstance(exc, BrokenPipeError)
                or "BrokenPipe" in repr(exc)
                or "Connection refused" in repr(exc))
            if not broken_pipe or attempt >= 3:
                raise
            print(
                f"[boot] connection error on boot attempt {attempt}/3; "
                f"restarting the simulator for seed={seed}",
                flush=True)
    raise AssertionError("simulator boot did not complete")


def _support_audit(world, x0: int, x1: int, floor_y: int,
                   z0: int, z1: int) -> dict:
    blocks = voxel_types(world, x0, x1, floor_y, floor_y, z0, z1)
    expected = (x1 - x0 + 1) * (z1 - z0 + 1)
    bare = [normalize_name(value) for value in blocks]
    passed = len(bare) == expected and all(value == "grass_block" for value in bare)
    if not passed:
        from collections import Counter
        raise RuntimeError(
            "per-column support audit failed: "
            f"expected={expected} observed={len(bare)} census={dict(Counter(bare))}")
    return {
        "contract": "per_column_grass_support_voxel_audit/v1",
        "passed": True,
        "columns": expected,
        "support_block": "minecraft:grass_block",
        "bounds_inclusive": [x0, x1, floor_y, floor_y, z0, z1],
    }


def _summon_command(spec: Mapping, x: float, y: float, z: float, *,
                    health: float, no_ai: bool = True,
                    yaw: float = 180.0) -> str:
    nbt_parts = [
        'Tags:["xh_scene","%s"]' % spec["tag"],
        f"NoAI:{1 if no_ai else 0}b",
        "Silent:1b",
        "PersistenceRequired:1b",
        f"Health:{float(health):.1f}f",
        ("Attributes:[{Name:\"minecraft:generic.max_health\",Base:"
         f"{float(health):.1f}d}}]"),
        f"Rotation:[{float(yaw):.1f}f,0.0f]",
    ]
    extra = str(spec.get("nbt") or "").strip().strip(",")
    if extra:
        nbt_parts.append(extra)
    return (
        f"/summon minecraft:{spec['entity']} {x:.1f} {y:.1f} {z:.1f} "
        "{" + ",".join(nbt_parts) + "}"
    )


def _save_rgb(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray(np.ascontiguousarray(rgb, dtype=np.uint8), mode="RGB").save(path)


def _save_mask(path: Path, mask: np.ndarray) -> None:
    from PIL import Image
    Image.fromarray((np.asarray(mask) > 0).astype(np.uint8) * 255,
                    mode="L").save(path)


def settle_wall_clock_ui(world, *, minimum_ticks: int = 120,
                         minimum_seconds: float = 7.0,
                         maximum_seconds: float = 45.0) -> dict:
    started = time.monotonic()
    ticks = 0
    clear_started = None
    final_neutral_fraction = None
    while True:
        world.step_noop()
        ticks += 1
        rgb = np.asarray(world.info["pov"], dtype=np.uint8)
        crop = rgb[:75, -180:]
        neutral = crop.max(axis=2).astype(np.int16) - crop.min(axis=2)
        final_neutral_fraction = float((neutral < 10).mean())
        now = time.monotonic()
        if final_neutral_fraction < 0.12:
            clear_started = now if clear_started is None else clear_started
        else:
            clear_started = None
        elapsed = now - started
        clear_wall = 0.0 if clear_started is None else now - clear_started
        if (ticks >= int(minimum_ticks)
                and elapsed >= float(minimum_seconds)
                and clear_wall >= 1.0):
            break
        if elapsed >= float(maximum_seconds):
            raise RuntimeError(
                "top-right UI toast did not clear before camera-on boundary: "
                f"neutral_fraction={final_neutral_fraction:.4f}")
    return {
        "ticks": int(ticks),
        "wall_seconds": float(time.monotonic() - started),
        "performed_camera_off": True,
        "top_right_neutral_fraction_final": final_neutral_fraction,
        "top_right_clear_wall_seconds_required": 1.0,
        "top_right_neutral_fraction_threshold": 0.12,
    }


def _pose(world) -> dict[str, float]:
    x, y, z = world.get_pos()
    return {
        "x": float(x), "y": float(y), "z": float(z),
        "yaw": float(world.get_yaw()), "pitch": float(world.get_pitch()),
    }


def _action_flag(env_action: Mapping, key: str) -> bool:
    try:
        return bool(np.asarray(env_action.get(key, 0)).reshape(-1)[0])
    except (TypeError, ValueError, IndexError):
        return False


def phase_summary(rows: Sequence[Mapping]) -> dict:
    usable = [row for row in rows
              if isinstance(row.get("phase_probabilities"), list)]
    by_exact_label = {}
    for phase_id, phase_name in enumerate(PHASE_NAMES):
        selected = [row for row in usable
                    if int(row["phase_label"]) == phase_id]
        if selected:
            array = np.asarray(
                [row["phase_probabilities"] for row in selected],
                dtype=np.float64)
            by_exact_label[phase_name] = {
                "n": len(selected),
                "mean_probabilities_EXP_APP_INT": [
                    float(value) for value in array.mean(axis=0)],
                "argmax_accuracy_against_exact_label": float(np.mean(
                    np.argmax(array, axis=1) == phase_id)),
            }
        else:
            by_exact_label[phase_name] = {"n": 0}
    attack_rows = [row for row in usable if bool(row.get("attack"))]
    attack_array = (
        np.asarray([row["phase_probabilities"] for row in attack_rows],
                   dtype=np.float64)
        if attack_rows else None)
    sequence = [int(np.argmax(row["phase_probabilities"])) for row in usable]
    compressed = []
    for value in sequence:
        if not compressed or compressed[-1] != value:
            compressed.append(value)
    return {
        "contract": "exact_entity_id_visibility_and_3_block_reach_phase/v2",
        "phase_order": list(PHASE_NAMES),
        "frames": len(usable),
        "by_exact_label": by_exact_label,
        "attack_frames": {
            "n": len(attack_rows),
            "mean_probabilities_EXP_APP_INT": (
                None if attack_array is None else
                [float(value) for value in attack_array.mean(axis=0)]),
            "interact_argmax_rate": (
                None if attack_array is None else
                float(np.mean(np.argmax(attack_array, axis=1) == 2))),
        },
        "compressed_argmax_sequence": [PHASE_NAMES[value]
                                        for value in compressed],
    }
