"""Controlled MineStudio world wrapper: boot (optionally from a saved-world snapshot), commands and stepping."""
from __future__ import annotations
import math
import os
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, PrevActionCallback
from attacca.runtime.minestudio_runtime import install_minecraft_logger_lifecycle_patch
from attacca.worlds.world_snapshot import install_snapshot_world_generator
from attacca.worlds.world_snapshot import validate_world_snapshot
from attacca.worlds.world_snapshot import validate_world_snapshot_archive


class ControlledWorldMS:
    def __init__(self, cfg):
        w = cfg["world"]
        snapshot_dir = w.get("world_snapshot_dir")
        snapshot_sha256 = w.get("world_snapshot_sha256")
        snapshot_archive = w.get("world_snapshot_archive")
        snapshot_archive_sha256 = w.get("world_snapshot_archive_sha256")
        if snapshot_sha256 is not None and snapshot_dir is None:
            raise ValueError(
                "world_snapshot_sha256 requires world_snapshot_dir")
        if snapshot_dir is not None and snapshot_archive is None:
            raise ValueError(
                "world_snapshot_dir requires world_snapshot_archive")
        if snapshot_archive_sha256 is not None and snapshot_archive is None:
            raise ValueError(
                "world_snapshot_archive_sha256 requires world_snapshot_archive")
        snapshot = (
            validate_world_snapshot(snapshot_dir, snapshot_sha256)
            if snapshot_dir is not None else None
        )
        validated_snapshot_archive = (
            validate_world_snapshot_archive(
                snapshot_archive, snapshot, snapshot_archive_sha256)
            if snapshot is not None else None)
        self.world_snapshot_dir = (
            None if snapshot is None else str(snapshot.directory))
        self._snapshot_reset_used = False
        install_minecraft_logger_lifecycle_patch()
        runtime_overlay = w.get("runtime_overlay")
        if not runtime_overlay:
            raise ValueError("world config requires runtime_overlay")
        runtime_overlay = os.path.abspath(os.path.expanduser(str(runtime_overlay)))
        options_path = os.path.join(runtime_overlay, "options.txt")
        if not os.path.isfile(options_path):
            raise FileNotFoundError(f"runtime overlay has no options.txt: {runtime_overlay}")
        from minestudio.simulator.minerl.env.malmo import InstanceManager
        InstanceManager.RUNTIME_DIR = runtime_overlay
        H, W = w["image_size"]
        init_cmds = [
            "/gamerule sendCommandFeedback false",
            "/time set 6000", "/weather clear",
            "/gamerule doDaylightCycle false",
            "/gamerule randomTickSpeed 0",
            "/gamerule doMobSpawning false",
            "/gamerule doWeatherCycle false",
            "/gamerule announceAdvancements false",
            "/difficulty peaceful",
        ]
        self.action_type = w.get("action_type", "env")
        self.sim = MinecraftSim(
            action_type=self.action_type,
            obs_size=(W, H),
            render_size=tuple(w.get("render_size", [640, 360])),
            preferred_spawn_biome=(
                None if snapshot is not None else w.get("biome", "plains")),
            seed=cfg["experiment"]["seed"],
            callbacks=[CommandsCallback(init_cmds), PrevActionCallback()],
        )
        if snapshot is not None:
            install_snapshot_world_generator(
                self.sim.env.task, snapshot,
                archive=validated_snapshot_archive,
                archive_sha256=snapshot_archive_sha256)
        if self.action_type == "agent":
            self._mapper = self.sim.action_mapper
            self._null_bin = self._mapper.camera_null_bin
        self.obs = None
        self.info = None

    def reset(self):
        if self.world_snapshot_dir is not None and self._snapshot_reset_used:
            raise RuntimeError(
                "world snapshots require a fresh subprocess for every reset")
        self.obs, self.info = self.sim.reset()
        if self.world_snapshot_dir is not None:
            self._snapshot_reset_used = True
        return self.obs, self.info

    def _noop(self):
        return self.sim.noop_action()

    def step_noop(self):
        self.obs, _, _, _, self.info = self.sim.step(self._noop())
        return self.obs

    def get_yaw(self):
        return float(self.info["player_pos"]["yaw"])

    def get_pitch(self):
        return float(self.info["player_pos"]["pitch"])

    def get_pos(self):
        p = self.info["player_pos"]
        return float(p["x"]), float(p["y"]), float(p["z"])

    def cmd(self, c):
        self.sim.env.execute_cmd(c)

    def _ground_y(self, x, z, ref_y, lift=50, max_steps=60):
        top = float(ref_y + lift)
        self.cmd(f"/tp @a {x} {top} {z}")
        prev_pos = (float(x), top, float(z))
        stable_ticks = 0
        last_pos = prev_pos
        last_movement = float("inf")
        for _ in range(max_steps):
            self.step_noop()
            pos = tuple(float(value) for value in self.get_pos())
            last_movement = math.sqrt(sum((cur - prev) ** 2
                                          for cur, prev in zip(pos, prev_pos)))
            if pos[1] < top - 1.0 and last_movement < 0.005:
                stable_ticks += 1
                if stable_ticks >= 4:
                    return pos[1]
            else:
                stable_ticks = 0
            prev_pos = pos
            last_pos = pos
        raise RuntimeError(
            "ground probe did not reach a stable landing "
            f"at ({float(x):.3f}, {float(z):.3f}) after {max_steps} ticks "
            f"(last_y={last_pos[1]:.3f}, movement={last_movement:.6f})"
        )

    def close(self):
        try:
            self.sim.close()
        except Exception:
            pass
