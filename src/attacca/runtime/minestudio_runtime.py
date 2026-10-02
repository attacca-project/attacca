"""Runtime settings applied before the MineStudio simulator is constructed."""
from __future__ import annotations

import functools
import threading


_LOGGER_LIFECYCLE_PATCH = "_attacca_logger_lifecycle_patch_v1"
_PATCH_LOCK = threading.Lock()


def install_minecraft_logger_lifecycle_patch() -> bool:
    from minestudio.simulator.minerl.env.malmo import MinecraftInstance

    instance_cls = MinecraftInstance

    with _PATCH_LOCK:
        original_launch = instance_cls.launch
        if getattr(original_launch, _LOGGER_LIFECYCLE_PATCH, False):
            return False

        @functools.wraps(original_launch)
        def launch_with_live_logger(self, *args, **kwargs):
            previous_running = bool(getattr(self, "running", False))
            self.running = True
            try:
                return original_launch(self, *args, **kwargs)
            except BaseException:
                self.running = previous_running
                raise

        setattr(launch_with_live_logger, _LOGGER_LIFECYCLE_PATCH, True)
        instance_cls.launch = launch_with_live_logger
        return True
