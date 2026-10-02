"""Mine target roster, target and confuser role assignment, and ore-cluster shape parity."""
import zlib

import numpy as np

PICKAXE_LEVEL = {
    "wooden_pickaxe": 0, "golden_pickaxe": 0, "stone_pickaxe": 1,
    "iron_pickaxe": 2, "diamond_pickaxe": 3, "netherite_pickaxe": 4,
}

MINE_KIT_ITEM = "minecraft:iron_pickaxe"

EPISODE_BIOME_PLACEMENT_SOURCE = "biome_plausible_placement"
NATURAL_EXACT_EXPOSED_BY_STONE_CARVE_SOURCE = (
    "natural_exact_exposed_by_stone_carve")
COUNTERFACTUAL_PAIRED_SCENE_CONTRACT = "same_physical_scene_goal_role_only/v1"

HARVEST_RULES = {
    "coal_ore": {
        "target_family": "ore", "harvest_mode": "pickaxe_tier",
        "minimum_pickaxe_level": 0, "drop_items": ("coal",),
        "stage_policy": "natural_or_exposed_stone",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "iron_ore": {
        "target_family": "ore", "harvest_mode": "pickaxe_tier",
        "minimum_pickaxe_level": 1, "drop_items": ("iron_ore",),
        "stage_policy": "natural_or_exposed_stone",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "gold_ore": {
        "target_family": "ore", "harvest_mode": "pickaxe_tier",
        "minimum_pickaxe_level": 2, "drop_items": ("gold_ore",),
        "stage_policy": "natural_or_exposed_stone",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "nether_gold_ore": {
        "target_family": "ore", "harvest_mode": "pickaxe_tier",
        "minimum_pickaxe_level": 0, "drop_items": ("gold_nugget",),
        "stage_policy": "natural_or_exposed_stone",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "lapis_ore": {
        "target_family": "ore", "harvest_mode": "pickaxe_tier",
        "minimum_pickaxe_level": 1, "drop_items": ("lapis_lazuli",),
        "stage_policy": "natural_or_exposed_stone",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "diamond_ore": {
        "target_family": "ore", "harvest_mode": "pickaxe_tier",
        "minimum_pickaxe_level": 2, "drop_items": ("diamond",),
        "stage_policy": "natural_or_exposed_stone",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "emerald_ore": {
        "target_family": "ore", "harvest_mode": "pickaxe_tier",
        "minimum_pickaxe_level": 2, "drop_items": ("emerald",),
        "stage_policy": "natural_or_exposed_stone",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "redstone_ore": {
        "target_family": "ore", "harvest_mode": "pickaxe_tier",
        "minimum_pickaxe_level": 2, "drop_items": ("redstone",),
        "stage_policy": "natural_or_exposed_stone",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "oak_log": {
        "target_family": "log", "harvest_mode": "any_tool",
        "minimum_pickaxe_level": None, "drop_items": ("oak_log",),
        "stage_policy": "exposed_stone_replacement_only",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "birch_log": {
        "target_family": "log", "harvest_mode": "any_tool",
        "minimum_pickaxe_level": None, "drop_items": ("birch_log",),
        "stage_policy": "exposed_stone_replacement_only",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "spruce_log": {
        "target_family": "log", "harvest_mode": "any_tool",
        "minimum_pickaxe_level": None, "drop_items": ("spruce_log",),
        "stage_policy": "exposed_stone_replacement_only",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "dark_oak_log": {
        "target_family": "log", "harvest_mode": "any_tool",
        "minimum_pickaxe_level": None, "drop_items": ("dark_oak_log",),
        "stage_policy": "exposed_stone_replacement_only",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "pumpkin": {
        "target_family": "produce", "harvest_mode": "any_tool",
        "minimum_pickaxe_level": None, "drop_items": ("pumpkin",),
        "stage_policy": "exposed_stone_replacement_only",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "melon": {
        "target_family": "produce", "harvest_mode": "any_tool",
        "minimum_pickaxe_level": None, "drop_items": ("melon_slice",),
        "stage_policy": "exposed_stone_replacement_only",
        "episode_source_policy": EPISODE_BIOME_PLACEMENT_SOURCE,
    },
    "mossy_cobblestone": {
        "target_family": "full_cube", "harvest_mode": "pickaxe_tier",
        "minimum_pickaxe_level": 0, "drop_items": ("mossy_cobblestone",),
        "stage_policy": "exposed_stone_replacement_only",
        "episode_source_policy": "blocked_no_truthful_route",
    },
}

MINE_POC_REMOVED_CLASSES = ("spruce_log", "dark_oak_log")

MINE_TARGET_POOL = (
    "coal_ore", "iron_ore", "gold_ore", "lapis_ore",
    "diamond_ore", "emerald_ore", "redstone_ore",
    "oak_log", "birch_log",
    "pumpkin", "melon", "mossy_cobblestone",
)
MINE_EPISODE_ENABLED_POOL = (
    "coal_ore", "iron_ore", "gold_ore", "lapis_ore",
    "diamond_ore", "emerald_ore", "redstone_ore",
    "oak_log", "birch_log",
    "pumpkin", "melon",
)
MINE_POC_ORE_POOL = (
    "coal_ore", "iron_ore", "gold_ore", "lapis_ore",
    "diamond_ore", "emerald_ore", "redstone_ore",
)
MINE_HELDOUT_EVAL_POOL = ("nether_gold_ore",)

MINE_ORE_POOL = MINE_TARGET_POOL

_FACE_NEIGHBOURS = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))


def bare_block(name):
    return str(name).split(":")[-1].split(" ")[0].strip()


def pickaxe_level(kit_item=MINE_KIT_ITEM):
    bare = bare_block(kit_item)
    if bare not in PICKAXE_LEVEL:
        raise ValueError(f"unknown mining tool {kit_item!r}; add it to PICKAXE_LEVEL")
    return int(PICKAXE_LEVEL[bare])


def harvest_rule(target_kind):
    kind = bare_block(target_kind)
    if kind not in HARVEST_RULES:
        raise ValueError(f"unknown mine target {kind!r}; add it to HARVEST_RULES")
    rule = dict(HARVEST_RULES[kind])
    rule["drop_items"] = tuple(bare_block(q) for q in rule["drop_items"])
    return rule


def episode_source_policy(target_kind):
    policy = str(harvest_rule(target_kind).get("episode_source_policy") or "")
    if policy not in {
            EPISODE_BIOME_PLACEMENT_SOURCE,
            "blocked_no_truthful_route",
    }:
        raise ValueError(
            f"mine target {bare_block(target_kind)!r} has invalid episode source policy "
            f"{policy!r}")
    return policy


def episode_target_generation_enabled(target_kind):
    kind = bare_block(target_kind)
    return (kind not in MINE_POC_REMOVED_CLASSES
            and episode_source_policy(kind) != "blocked_no_truthful_route")


def episode_staging_provenance(target_kind, source, original):
    kind = bare_block(target_kind)
    if kind in MINE_POC_REMOVED_CLASSES:
        raise ValueError(
            f"mine episode target {kind!r} is not in the Mine target roster")
    source = str(source)
    original = bare_block(original)
    policy = episode_source_policy(kind)
    if policy == "blocked_no_truthful_route":
        raise ValueError(f"mine episode target {kind!r} has no truthful source route")
    if source != EPISODE_BIOME_PLACEMENT_SOURCE:
        raise ValueError(
            f"biome-plausible mine episode target {kind!r} requires source="
            f"{EPISODE_BIOME_PLACEMENT_SOURCE!r}; got {source!r}")
    family = str(harvest_rule(kind)["target_family"])
    if family == "ore":
        allowed_originals = {"stone"}
        template = "certified_exposed_stone_replacement"
    elif family == "log":
        allowed_originals = {
            "air", "cave_air", "void_air", "grass", "tall_grass",
            "fern", "large_fern", "snow",
        }
        template = "grounded_vertical_trunk_segment"
    elif family == "produce":
        allowed_originals = {
            "air", "cave_air", "void_air", "grass", "tall_grass",
            "fern", "large_fern", "snow",
        }
        template = "solid_supported_crop_patch"
    else:
        raise ValueError(
            f"mine episode target {kind!r} has no biome-plausible family template")
    if original not in allowed_originals:
        raise ValueError(
            f"biome-plausible {family} target {kind!r} cannot replace "
            f"{original!r}; allowed={sorted(allowed_originals)}")
    return {
        "staging_provenance": f"{EPISODE_BIOME_PLACEMENT_SOURCE}/v1",
        "placement_source": EPISODE_BIOME_PLACEMENT_SOURCE,
        "placement_template": template,
        "episode_source_policy": policy,
        "controlled_poc": True,
        "target_naturally_generated": False,
        "source_block_kind": original,
        "target_family": family,
        "goal_bank_natural_only_unchanged": True,
    }


def validate_mine_pool(pool=MINE_TARGET_POOL, kit_item=MINE_KIT_ITEM):
    pool = tuple(bare_block(c) for c in pool)
    if len(pool) < 2:
        raise ValueError(f"mine target pool needs >=2 classes to rotate roles, got {pool}")
    if len(set(pool)) != len(pool):
        raise ValueError(f"mine target pool has duplicates: {pool}")
    unknown = [c for c in pool if c not in HARVEST_RULES]
    if unknown:
        raise ValueError(f"unknown target classes {unknown}; add them to HARVEST_RULES")
    removed = [c for c in pool if c in MINE_POC_REMOVED_CLASSES]
    if removed:
        raise ValueError(
            f"mine target classes {removed} are excluded from the Mine target roster")
    level = pickaxe_level(kit_item)
    too_hard = []
    for kind in pool:
        rule = harvest_rule(kind)
        episode_source_policy(kind)
        mode = rule.get("harvest_mode")
        drops = tuple(rule.get("drop_items") or ())
        if not drops or any(not bare_block(item) for item in drops):
            raise ValueError(f"mine target {kind!r} has no valid drop_items")
        if mode == "pickaxe_tier":
            required = rule.get("minimum_pickaxe_level")
            if not isinstance(required, int) or required < 0:
                raise ValueError(
                    f"mine target {kind!r} has invalid minimum_pickaxe_level {required!r}")
            if required > level:
                too_hard.append((kind, required))
        elif mode == "any_tool":
            if rule.get("minimum_pickaxe_level") is not None:
                raise ValueError(
                    f"any-tool target {kind!r} must not declare a pickaxe tier")
        else:
            raise ValueError(f"mine target {kind!r} has unknown harvest_mode {mode!r}")
    if too_hard:
        raise ValueError(
            f"{kit_item} is harvest level {level}; {too_hard} would break WITHOUT dropping. "
            "Use a higher-tier MINE_KIT_ITEM or remove those classes from MINE_TARGET_POOL.")
    return pool


_VALIDATED_MINE_TARGET_POOL = validate_mine_pool()
if tuple(kind for kind in MINE_TARGET_POOL
         if episode_target_generation_enabled(kind)) != MINE_EPISODE_ENABLED_POOL:
    raise ValueError(
        "MINE_EPISODE_ENABLED_POOL disagrees with episode_source_policy")
if (not set(MINE_POC_ORE_POOL).issubset(MINE_EPISODE_ENABLED_POOL)
        or any(harvest_rule(kind)["target_family"] != "ore"
               for kind in MINE_POC_ORE_POOL)):
    raise ValueError("MINE_POC_ORE_POOL must be an episode-enabled all-ore subset")
if (set(MINE_POC_REMOVED_CLASSES) & set(MINE_TARGET_POOL)
        or any(episode_target_generation_enabled(kind)
               for kind in MINE_POC_REMOVED_CLASSES)):
    raise ValueError("excluded classes appear in the Mine target roster")


def ore_role_seed(cell, world_seed, layout_seed):
    key = f"oreclass:{cell}:w{int(world_seed)}:l{int(layout_seed)}"
    return int(zlib.crc32(key.encode()))


def assign_ore_roles(cell, world_seed, layout_seed, pool=MINE_TARGET_POOL,
                     n_distractor_classes=1, forced_target=None):
    pool = validate_mine_pool(pool)
    n = int(n_distractor_classes)
    if not 0 <= n <= len(pool) - 1:
        raise ValueError(f"n_distractor_classes must be 0..{len(pool) - 1}, got {n}")
    seed = ore_role_seed(cell, world_seed, layout_seed)
    rng = np.random.default_rng(seed)
    target = pool[int(rng.integers(len(pool)))]
    if forced_target is not None:
        target = bare_block(forced_target)
        if target not in pool:
            raise ValueError(f"forced target {target!r} is not in the pool {pool}")
    rest = [c for c in pool if c != target]
    order = rng.permutation(len(rest))
    distractors = tuple(rest[int(i)] for i in order[:n])
    return {"cell": str(cell), "seed": seed, "pool": pool, "target": target,
            "distractors": distractors,
            "forced": forced_target is not None}


def cluster_sizes(cells):
    remaining = {tuple(int(v) for v in c) for c in cells}
    sizes = []
    while remaining:
        seed = min(remaining)
        stack, comp = [seed], set()
        remaining.discard(seed)
        while stack:
            cur = stack.pop()
            comp.add(cur)
            for dx, dy, dz in _FACE_NEIGHBOURS:
                nxt = (cur[0] + dx, cur[1] + dy, cur[2] + dz)
                if nxt in remaining:
                    remaining.discard(nxt)
                    stack.append(nxt)
        sizes.append(len(comp))
    return tuple(sorted(sizes, reverse=True))


def goal_key(cell, target_kind=None):
    if str(cell) == "place_block":
        kind = bare_block(target_kind or "")
        if not kind:
            raise ValueError("semantic USE marker goal requires target_kind")
        return f"mine_{kind}"
    if str(cell) == "hunt":
        kind = str(target_kind or "").strip().split(":")[-1]
        if kind == "pig":
            return "hunt_pig"
        return f"hunt_{kind}" if kind else "hunt"
    if str(cell) != "mine":
        return str(cell)
    kind = bare_block(target_kind or "")
    if not kind:
        raise ValueError("mine goal key requires the episode's target ore class")
    return f"mine_{kind}"


