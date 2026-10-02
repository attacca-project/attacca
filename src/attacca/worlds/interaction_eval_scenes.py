from __future__ import annotations

from typing import Mapping


MINE_INTERACTION_ID = 2


def portal_activation_success(
        queried_types: Mapping[tuple[int, int, int], str],
        scene) -> bool:
    frame_complete = all(
        str(queried_types.get(cell, "air")).removeprefix("minecraft:")
        == "obsidian" for cell in scene.complete_frame_cells)
    portal_present = any(
        str(queried_types.get(cell, "air")).removeprefix("minecraft:")
        == "nether_portal" for cell in scene.portal_interior_cells)
    return bool(frame_complete and portal_present)


def portal_resource_conversion_counts(
        queried_types: Mapping[tuple[int, int, int], str],
        scene) -> dict[str, int]:
    kinds = [
        str(queried_types.get(cell, "air")).removeprefix("minecraft:")
        for cell in scene.lava_cells]
    return {
        "lava": sum(kind == "lava" for kind in kinds),
        "obsidian": sum(kind == "obsidian" for kind in kinds),
        "cobblestone": sum(kind == "cobblestone" for kind in kinds),
        "other": sum(kind not in {"lava", "obsidian", "cobblestone"}
                     for kind in kinds),
    }


def portal_resource_conversion_success(
        queried_types: Mapping[tuple[int, int, int], str],
        scene) -> bool:
    counts = portal_resource_conversion_counts(queried_types, scene)
    return bool(
        counts["obsidian"] == len(scene.lava_cells)
        and counts["lava"] == 0
        and counts["cobblestone"] == 0
        and counts["other"] == 0)

