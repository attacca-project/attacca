from __future__ import annotations

from copy import deepcopy
import math
from typing import Sequence

from attacca.worlds.coal_cow_furnace_chain_scene import CoalCowFurnaceScene
from attacca.worlds.coal_cow_furnace_chain_scene import build_scene_commands as build_base_scene_commands
from attacca.worlds.coal_cow_furnace_chain_scene import validate_scene_manifest as validate_base_scene_manifest
from attacca.worlds.diamond_pickaxe_chain_scene import Cell
from attacca.worlds.diamond_pickaxe_chain_scene import SceneBuildResult
from attacca.worlds.diamond_pickaxe_chain_scene import _base_kind
from attacca.worlds.diamond_pickaxe_chain_scene import _shortest_path


SCENE_CONTRACT = "xbench_coal_cow_furnace_wolf_scene/v3"
WOLF_STAGE_KEY = "wolf_feed"
WOLF_INTERACTION_ID = 3
WOLF_TAG = "xb_ccfw_wolf"
WOLF_INITIAL_HEALTH = 10.0
WOLF_NORMAL_MAX_HEALTH = 20.0
PROTECTION_MODE = "invulnerable"


def _uuid_int_array(values: Sequence[int]) -> tuple[int, int, int, int]:
    if len(values) != 4:
        raise ValueError("Minecraft UUID int-array must contain four integers")
    result = tuple(int(value) for value in values)
    if any(value < -(1 << 31) or value >= (1 << 31) for value in result):
        raise ValueError("Minecraft UUID int-array element is outside signed int32")
    return result


def wolf_cell(scene: CoalCowFurnaceScene, reset_bounds: dict) -> Cell:
    fx, fy, fz = scene.furnace_cell
    x_min = int(reset_bounds["x_min"])
    x_max = int(reset_bounds["x_max"])
    mirrored_x = x_min + (x_max - fx)
    return mirrored_x - 2, fy, fz - 2


def wolf_position(scene: CoalCowFurnaceScene, reset_bounds: dict
                  ) -> tuple[float, float, float]:
    x, y, z = wolf_cell(scene, reset_bounds)
    return x + 0.5, float(y), z + 0.5


def wolf_stance_cells(scene: CoalCowFurnaceScene, reset_bounds: dict
                      ) -> tuple[Cell, ...]:
    x, y, z = wolf_cell(scene, reset_bounds)
    return ((x, y, z + 2), (x - 2, y, z), (x + 2, y, z))


def _summon_command(position: Sequence[float], owner_uuid: Sequence[int]) -> str:
    owner = _uuid_int_array(owner_uuid)
    invulnerable = ",Invulnerable:1b"
    owner_nbt = ",".join(str(value) for value in owner)
    return (
        f"/summon minecraft:wolf {float(position[0]):.1f} "
        f"{float(position[1]):.1f} {float(position[2]):.1f} "
        "{Owner:[I;" + owner_nbt + "],Sitting:1b,PersistenceRequired:1b,"
        f"Health:{WOLF_INITIAL_HEALTH:.1f}f{invulnerable},"
        f'Tags:["{WOLF_TAG}"]}}')


def build_scene_commands(
        scene: CoalCowFurnaceScene, *, owner_uuid: Sequence[int]) -> SceneBuildResult:
    owner = _uuid_int_array(owner_uuid)
    base = build_base_scene_commands(scene)
    validate_base_scene_manifest(base, raise_on_error=True)
    manifest = deepcopy(base.manifest)
    blocks = dict(base.final_blocks)
    bounds = manifest["reset_bounds"]
    cell = wolf_cell(scene, bounds)
    position = wolf_position(scene, bounds)
    support = cell[0], cell[1] - 1, cell[2]
    body_above = cell[0], cell[1] + 1, cell[2]
    if _base_kind(blocks.get(support, "air")) in {
            "air", "water", "lava"}:
        raise ValueError(f"wolf support column is unsafe: {support}")
    if any(_base_kind(blocks.get(point, "air")) != "air"
           for point in (cell, body_above)):
        raise ValueError(f"wolf body column is obstructed: {cell}")

    selector = f"@e[type=minecraft:wolf,tag={WOLF_TAG},limit=1]"
    target_max_health = WOLF_NORMAL_MAX_HEALTH
    commands = [*base.commands,
                _summon_command(position, owner),
                (f"/attribute {selector} minecraft:generic.max_health "
                 f"base set {WOLF_INITIAL_HEALTH:.1f}"),
                (f"/effect give {selector} minecraft:instant_health "
                 "1 0 true"),
                (f"/attribute {selector} minecraft:generic.max_health "
                 f"base set {target_max_health:.1f}")]

    ccfw_stages = ("coal_ore_mine", "cow_hunt", "furnace_find")
    manifest.update({
        "contract": SCENE_CONTRACT,
        "base_scene_contract": base.manifest["contract"],
        "layout_id": f"{manifest['layout_id']}_wolf_terminal_v1",
        "policy_stage_keys": [*ccfw_stages, WOLF_STAGE_KEY],
        "target_kinds": ["coal_ore", "cow", "furnace", "wolf"],
        "success_quotas": [1, 1, 1, 1],
        "interaction_ids": [2, 0, 3, WOLF_INTERACTION_ID],
    })
    manifest["target_cells"] = {
        **{key: manifest["target_cells"][key] for key in ccfw_stages},
        WOLF_STAGE_KEY: [list(cell)]}
    stances = wolf_stance_cells(scene, bounds)
    manifest["stage_stance_cells"] = {
        **{key: manifest["stage_stance_cells"][key] for key in ccfw_stages},
        WOLF_STAGE_KEY: [list(value) for value in stances],
    }
    manifest["target_support_cells"] = {
        **{key: manifest["target_support_cells"][key] for key in ccfw_stages},
        WOLF_STAGE_KEY: [list(support)]}
    manifest["verified_stand_cells"] = [
        *manifest["verified_stand_cells"],
        *(list(value) for value in stances),
    ]
    navigation = {tuple(map(int, value))
                  for value in manifest["navigation_cells"]}
    prior_path = manifest["navigation_paths"]["furnace_find"]
    departure = tuple(map(int, prior_path[-1]))
    wolf_path = _shortest_path(navigation, departure, stances)
    manifest["navigation_paths"] = {
        **{key: manifest["navigation_paths"][key] for key in ccfw_stages},
        WOLF_STAGE_KEY: [list(value) for value in wolf_path],
    }
    distance = math.dist(
        (scene.furnace_cell[0] + .5, scene.furnace_cell[2] + .5),
        (position[0], position[2]))
    manifest["navigation_distances_blocks"] = {
        **manifest["navigation_distances_blocks"],
        "furnace_to_wolf": float(distance),
    }
    visibility = deepcopy(manifest["start_visibility_contract"])
    visibility["coal_must_be_visible"] = False
    visibility["coal_must_be_hidden"] = True
    visibility["hidden_stage_keys"] = [
        "coal_ore_mine", "cow_hunt", "furnace_find", WOLF_STAGE_KEY]
    visibility["wolf_feed"] = (
        "frame-zero exact visible pixels must be zero after moving the wolf "
        "two blocks west and two blocks north from the mirror point")
    manifest["start_visibility_contract"] = visibility
    manifest["wolf"] = {
        "entity": "minecraft:wolf",
        "unique_tag": WOLF_TAG,
        "cell": list(cell),
        "spawn_position": list(position),
        "support_cell": list(support),
        "support_kind": _base_kind(blocks[support]),
        "furnace_position": [scene.furnace_cell[0] + .5,
                             float(scene.furnace_cell[1]),
                             scene.furnace_cell[2] + .5],
        "furnace_distance_blocks": float(distance),
        "placement": "mirror_point_minus_x2_z2_in_empty_terminal_quadrant",
        "owner_uuid_int_array": list(owner),
        "owner_source": "/data get entity @p UUID at runtime",
        "tamed_by_owner": True,
        "Sitting": True,
        "PersistenceRequired": True,
        "Invulnerable": True,
        "initial_health": WOLF_INITIAL_HEALTH,
        "max_health": target_max_health,
        "protection_mode": PROTECTION_MODE,
        "health_setup_after_owner_load": (
            "set max_health=10.0, clamp through instant_health's setHealth, "
            f"then raise max_health to {target_max_health:.1f}"),
    }
    narrative = {
        key: value for key, value in manifest["causal_narrative"].items()
        if key != "beef_cook"}
    narrative["furnace_find"] = (
        "Find the exact furnace and open it within one policy budget.")
    manifest["causal_narrative"] = {
        **narrative,
        WOLF_STAGE_KEY: (
            "Find the sitting tamed wolf and use the one cooked beef to heal it."),
        "scored_endpoint": (
            "policy exact-wolf USE with cooked-beef decrement and live Health increase"),
    }
    manifest["system_controls"] = {
        **manifest["system_controls"],
        "wolf_transition": (
            "existing real-GUI furnace cook, real-GUI cooked-beef hotbar move, "
            "then camera/movement/use remain policy-owned"),
    }
    manifest["staging"] = {
        **manifest["staging"],
        "wolf_summoned_after_mob_purge": True,
        "wolf_owner_uuid_runtime_readback_required": True,
    }
    result = SceneBuildResult(tuple(commands), manifest, blocks)
    validate_scene_manifest(result, raise_on_error=True)
    return result


def validate_scene_manifest(result: SceneBuildResult, *,
                            raise_on_error: bool = False) -> dict:
    manifest, blocks = result.manifest, result.final_blocks
    issues: list[str] = []
    if manifest.get("contract") != SCENE_CONTRACT:
        issues.append("wolf scene contract mismatch")
    if tuple(manifest.get("policy_stage_keys", ()))[-1:] != (WOLF_STAGE_KEY,):
        issues.append("wolf_feed is not terminal")
    if tuple(manifest.get("interaction_ids", ()))[-1:] != (WOLF_INTERACTION_ID,):
        issues.append("wolf_feed interaction ID mismatch")
    wolf = manifest.get("wolf", {})
    cell = tuple(wolf.get("cell", ()))
    support = tuple(wolf.get("support_cell", ()))
    if len(cell) != 3 or len(support) != 3:
        issues.append("wolf cell/support missing")
    else:
        if _base_kind(blocks.get(support, "air")) in {
                "air", "water", "lava"}:
            issues.append("wolf support is unsafe")
        if any(_base_kind(blocks.get(point, "air")) != "air"
               for point in (cell, (cell[0], cell[1] + 1, cell[2]))):
            issues.append("wolf body column is obstructed")
    if len(tuple(wolf.get("owner_uuid_int_array", ()))) != 4:
        issues.append("wolf owner UUID int-array missing")
    if not manifest.get("navigation_paths", {}).get(WOLF_STAGE_KEY):
        issues.append("wolf stance is unreachable")
    if manifest.get("target_kinds", [])[-1:] != ["wolf"]:
        issues.append("wolf target kind missing")
    report = {"passed": not issues, "issues": issues,
              "contract": "coal_cow_furnace_wolf_scene_pure_audit/v1"}
    if raise_on_error and issues:
        raise ValueError("; ".join(issues))
    return report
