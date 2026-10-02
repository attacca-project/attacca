from __future__ import annotations

import math
import time
from types import MappingProxyType

import cv2
import numpy as np

W_PX, H_PX = 640, 360
FOV_V = math.radians(70.0)
FY = (H_PX / 2.0) / math.tan(FOV_V / 2.0)
EYE = 1.62

NATIVE_UI_STATIC_RECTS_XYXY = MappingProxyType({
    "bottom_health_hunger_hotbar": (228, 316, 412, 360),
    "held_item_conservative": (500, 170, 640, 360),
    "top_overlay": (0, 0, 640, 22),
})


def native_ui_pixel_exclusion_mask(*, include_top_overlay=False):
    excluded9 = np.zeros((H_PX, W_PX), dtype=np.uint8)
    rect_names9 = [
        "bottom_health_hunger_hotbar",
        "held_item_conservative",
    ]
    if include_top_overlay:
        rect_names9.append("top_overlay")
    for name9 in rect_names9:
        x09, y09, x19, y19 = NATIVE_UI_STATIC_RECTS_XYXY[name9]
        excluded9[y09:y19, x09:x19] = 1
    return excluded9


PASSTHROUGH = {
    "air", "cave_air", "void_air",
    "grass", "tall_grass", "fern", "large_fern", "dead_bush", "sweet_berry_bush",
    "seagrass", "tall_seagrass", "kelp", "kelp_plant", "sugar_cane",
    "poppy", "dandelion", "oxeye_daisy", "azure_bluet", "cornflower", "allium",
    "blue_orchid", "red_tulip", "orange_tulip", "white_tulip", "pink_tulip",
    "lily_of_the_valley", "wither_rose", "sunflower", "lilac", "rose_bush", "peony",
    "brown_mushroom", "red_mushroom",
    "snow",
    "torch", "wall_torch", "redstone_torch", "vine",
    "wheat", "carrots", "potatoes", "beetroots",
}

CUTOUT_GEOMETRY = {
    "grass", "tall_grass", "fern", "large_fern", "dead_bush",
    "sweet_berry_bush", "seagrass", "tall_seagrass", "kelp", "kelp_plant",
    "sugar_cane", "poppy", "dandelion", "oxeye_daisy", "azure_bluet",
    "cornflower", "allium", "blue_orchid", "red_tulip", "orange_tulip",
    "white_tulip", "pink_tulip", "lily_of_the_valley", "wither_rose",
    "sunflower", "lilac", "rose_bush", "peony", "brown_mushroom",
    "red_mushroom", "snow", "torch", "wall_torch", "redstone_torch",
    "vine", "wheat", "carrots", "potatoes", "beetroots",
}

VANILLA_1_16_CROSS_ALPHA_MASKS = {
    "grass": 0xfffffffe7ffffffdb77bd76d5dad2fec2b642bb44ab216a814a0249000000000,
    "tall_grass": 0xffffffffffffffffffffffefffee7fee7feb77fb77fb777a7d762d542d543d56,
    "fern": 0x01c0008001c003e003e06ff07ff57fff2aaf01ca07e003f0008001c001c00080,
    "large_fern": 0x3fbc3ffc1ff87ff8fffe7ffe1ffc3ff87ffc3ffe7ffffffefffc7f767ffe3ffc,
    "dead_bush": 0x03800f801bc003e00730038007c00fe03bb863dc01e00b60073005100c100800,
}

VANILLA_1_16_CROSS_OFFSET_TYPES = {
    "grass": "xyz",
    "fern": "xyz",
    "tall_grass": "xz",
    "large_fern": "xz",
    "dead_bush": "none",
}

_ALPHA_CUTOUT_INDEX_CACHE_MAX = 8
_ALPHA_CUTOUT_INDEX_CACHE = {}


def _bare(t: str) -> str:
    return t[10:] if t.startswith("minecraft:") else t


def certified_pixel_runs_yx(mask):
    arr9 = np.asarray(mask, dtype=np.uint8)
    if arr9.ndim != 2:
        raise ValueError("certified pixel mask must be two-dimensional")
    runs9 = []
    for yy9 in range(arr9.shape[0]):
        xs9 = np.flatnonzero(arr9[yy9])
        if not len(xs9):
            continue
        start9 = previous9 = int(xs9[0])
        for raw_x9 in xs9[1:]:
            xx9 = int(raw_x9)
            if xx9 == previous9 + 1:
                previous9 = xx9
                continue
            runs9.append([int(yy9), start9, previous9])
            start9 = previous9 = xx9
        runs9.append([int(yy9), start9, previous9])
    return runs9


_GRID_MISSING = object()


class RevisionTrackedGrid(dict):

    def __init__(self, *args, **kwargs):
        dict.__init__(self)
        self._revision = 0
        self._batch_depth = 0
        self._batch_changed = False
        if args or kwargs:
            dict.update(self, *args, **kwargs)

    @property
    def revision(self):
        return int(self._revision)

    def begin_revision_batch(self):
        self._batch_depth += 1

    def end_revision_batch(self):
        if self._batch_depth <= 0:
            raise RuntimeError("revision batch underflow")
        self._batch_depth -= 1
        if self._batch_depth == 0 and self._batch_changed:
            self._revision += 1
            self._batch_changed = False

    def _mark_semantic_change(self):
        if self._batch_depth:
            self._batch_changed = True
        else:
            self._revision += 1

    def __setitem__(self, key, value):
        old9 = dict.get(self, key, _GRID_MISSING)
        if old9 is not _GRID_MISSING and old9 == value:
            return
        dict.__setitem__(self, key, value)
        self._mark_semantic_change()

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        self._mark_semantic_change()

    def clear(self):
        if not self:
            return
        dict.clear(self)
        self._mark_semantic_change()

    def pop(self, key, default=_GRID_MISSING):
        if key not in self:
            if default is _GRID_MISSING:
                raise KeyError(key)
            return default
        value9 = dict.pop(self, key)
        self._mark_semantic_change()
        return value9

    def popitem(self):
        value9 = dict.popitem(self)
        self._mark_semantic_change()
        return value9

    def setdefault(self, key, default=None):
        if key in self:
            return dict.__getitem__(self, key)
        dict.__setitem__(self, key, default)
        self._mark_semantic_change()
        return default

    def update(self, *args, **kwargs):
        incoming9 = dict(*args, **kwargs)
        changed9 = False
        for key9, value9 in incoming9.items():
            old9 = dict.get(self, key9, _GRID_MISSING)
            if old9 is _GRID_MISSING or old9 != value9:
                dict.__setitem__(self, key9, value9)
                changed9 = True
        if changed9:
            self._mark_semantic_change()

    def __ior__(self, other):
        self.update(other)
        return self


class OccupancyMap:

    def __init__(self, water_occludes: bool = False, leaves_occlude: bool = True):
        self._grid = RevisionTrackedGrid()
        self.boxes: list[tuple] = []
        self.water_occludes = water_occludes
        self.leaves_occlude = leaves_occlude
        self._structural_class_index_cache = None
        self._visible_surface_scene_cache = None

    @property
    def grid(self):
        return self._grid

    @grid.setter
    def grid(self, value):
        self._grid = (value if isinstance(value, RevisionTrackedGrid)
                      else RevisionTrackedGrid(value))
        self._structural_class_index_cache = None
        self._visible_surface_scene_cache = None

    @property
    def revision(self):
        return int(self._grid.revision)

    def ingest(self, voxels_list, player_pos, box):
        px = int(math.floor(float(player_pos[0])))
        py = int(math.floor(float(player_pos[1])))
        pz = int(math.floor(float(player_pos[2])))
        ab = (px + int(box[0]), px + int(box[1]), py + int(box[2]),
              py + int(box[3]), pz + int(box[4]), pz + int(box[5]))
        present = {}
        for e in voxels_list:
            c = (px + int(e["x"]), py + int(e["y"]), pz + int(e["z"]))
            present[c] = e["type"]
        removed = []
        self.grid.begin_revision_batch()
        try:
            vol = max(0, ab[1] - ab[0]) * max(0, ab[3] - ab[2]) * max(0, ab[5] - ab[4])
            if vol and vol < len(self.grid):
                for x in range(ab[0], ab[1]):
                    for y in range(ab[2], ab[3]):
                        for z in range(ab[4], ab[5]):
                            c = (x, y, z)
                            old = self.grid.get(c)
                            if old is not None and c not in present:
                                removed.append((c, old))
                                del self.grid[c]
            else:
                inside = [c for c in self.grid
                          if ab[0] <= c[0] < ab[1] and ab[2] <= c[1] < ab[3] and ab[4] <= c[2] < ab[5]]
                for c in inside:
                    if c not in present:
                        removed.append((c, self.grid[c]))
                        del self.grid[c]
            self.grid.update(present)
        finally:
            self.grid.end_revision_batch()
        if not self.boxes or self.boxes[-1] != ab:
            self.boxes.append(ab)
        return removed

    def known(self, c) -> bool:
        x, y, z = c
        for (x0, x1, y0, y1, z0, z1) in reversed(self.boxes):
            if x0 <= x < x1 and y0 <= y < y1 and z0 <= z < z1:
                return True
        return False

    def type_at(self, c):
        return self.grid.get(tuple(int(v) for v in c))

    def solid_at(self, c) -> bool:
        t = self.grid.get(c)
        if t is None:
            return False
        b = _bare(t)
        if b in PASSTHROUGH:
            return False
        if b == "water" or b == "bubble_column":
            return self.water_occludes
        if b.endswith("_leaves"):
            return self.leaves_occlude
        return True


def mine_diagonal_swept_envelope_clear(occ, origin, destination):
    try:
        origin9 = tuple(int(q9) for q9 in origin)
        destination9 = tuple(int(q9) for q9 in destination)
        exact9 = (
            len(origin9) == 3 and len(destination9) == 3
            and all(float(origin[i9]) == float(origin9[i9]) for i9 in range(3))
            and all(float(destination[i9]) == float(destination9[i9])
                    for i9 in range(3)))
    except (TypeError, ValueError, OverflowError):
        return False
    dx9 = destination9[0] - origin9[0]
    dz9 = destination9[2] - origin9[2]
    if (not exact9 or abs(dx9) != 1 or abs(dz9) != 1
            or abs(destination9[1] - origin9[1]) > 1):
        return False
    shoulder_columns9 = (
        (origin9[0] + dx9, origin9[2]),
        (origin9[0], origin9[2] + dz9),
    )
    lower_y9 = min(origin9[1], destination9[1])
    upper_y9 = max(origin9[1], destination9[1]) + 1
    forbidden9 = {"water", "lava", "bubble_column"}
    for shoulder_x9, shoulder_z9 in shoulder_columns9:
        if not any(mine_stance_known_clear(
                occ, (shoulder_x9, candidate_y9, shoulder_z9))
                for candidate_y9 in (
                    origin9[1], origin9[1] + 1, origin9[1] - 1)):
            return False
        for shoulder_y9 in range(lower_y9, upper_y9 + 1):
            shoulder9 = (shoulder_x9, shoulder_y9, shoulder_z9)
            if not occ.known(shoulder9):
                return False
            raw9 = occ.grid.get(shoulder9)
            bare9 = _bare(str(raw9 or ""))
            if (bare9 in forbidden9
                    or (raw9 is not None and bare9 not in PASSTHROUGH)):
                return False
    return True


def raycast(origin_f, dir_f, max_dist, occ: OccupancyMap,
            ignore=frozenset(), extra_solid=frozenset(), unknown_is_solid=False,
            thin_solid=None):
    o = np.asarray(origin_f, dtype=np.float64)
    d = np.asarray(dir_f, dtype=np.float64)
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return None, 0.0
    d = d / n
    cell = [int(math.floor(o[i])) for i in range(3)]

    def hit(c):
        return c not in ignore and (
            occ.solid_at(c) or c in extra_solid
            or (unknown_is_solid and not occ.known(c)))

    c0 = tuple(cell)
    if hit(c0):
        return c0, 0.0
    step, t_max, t_delta = [0, 0, 0], [math.inf] * 3, [math.inf] * 3
    for i in range(3):
        if d[i] > 1e-12:
            step[i] = 1
            t_max[i] = (cell[i] + 1 - o[i]) / d[i]
            t_delta[i] = 1.0 / d[i]
        elif d[i] < -1e-12:
            step[i] = -1
            t_max[i] = (cell[i] - o[i]) / d[i]
            t_delta[i] = -1.0 / d[i]
    while True:
        axis = 0
        if t_max[1] < t_max[axis]:
            axis = 1
        if t_max[2] < t_max[axis]:
            axis = 2
        t = t_max[axis]
        if t > max_dist:
            return None, max_dist
        cell[axis] += step[axis]
        t_max[axis] += t_delta[axis]
        c = tuple(cell)
        if hit(c):
            return c, t
        if thin_solid is not None:
            top = thin_solid.get(c)
            if top is not None:
                t_exit = min(t_max[0], t_max[1], t_max[2], max_dist)
                if min(o[1] + d[1] * t, o[1] + d[1] * t_exit) < float(top):
                    return c, t


def _signed_bits(value, bits):
    mask = (1 << int(bits)) - 1
    value = int(value) & mask
    sign = 1 << (int(bits) - 1)
    return value - (1 << int(bits)) if value & sign else value


def vanilla_position_random_x0z(x, z):
    x_term = _signed_bits(int(x) * 3129871, 32)
    z_term = _signed_bits(int(z) * 116129781, 64)
    mixed = _signed_bits(
        (x_term & ((1 << 64) - 1)) ^ (z_term & ((1 << 64) - 1)), 64)
    mixed = _signed_bits(mixed * mixed * 42317861 + mixed * 11, 64)
    return int(mixed >> 16)


def vanilla_cross_model_offset(cell, block_type):
    bare = _bare(str(block_type))
    offset_type = VANILLA_1_16_CROSS_OFFSET_TYPES.get(bare)
    if offset_type is None:
        raise ValueError(f"unsupported vanilla cross model: {block_type!r}")
    if offset_type == "none":
        return (0.0, 0.0, 0.0)
    random_value = vanilla_position_random_x0z(int(cell[0]), int(cell[2]))
    dx = (((random_value & 15) / 15.0) - 0.5) * 0.5
    dy = ((((random_value >> 4) & 15) / 15.0) - 1.0) * 0.2
    dz = ((((random_value >> 8) & 15) / 15.0) - 0.5) * 0.5
    return (float(dx), float(dy if offset_type == "xyz" else 0.0), float(dz))


def _cross_alpha_entry(origin, direction, cell, t_enter, t_exit, alpha_mask,
                       model_offset=(0.0, 0.0, 0.0)):
    o9 = np.asarray(origin, dtype=np.float64)
    d9 = np.asarray(direction, dtype=np.float64)
    c9 = (np.asarray(cell, dtype=np.float64)
          + np.asarray(model_offset, dtype=np.float64))
    lo_t9 = max(0.0, float(t_enter) - 1e-8)
    hi_t9 = float(t_exit) + 1e-8
    hits9 = []

    def opaque9(u9, local_y9):
        inset9 = 0.8 / 16.0
        if not (inset9 - 1e-7 <= u9 <= 1.0 - inset9 + 1e-7
                and -1e-7 <= local_y9 <= 1.0 + 1e-7):
            return False
        texture_u9 = ((float(u9) - inset9) /
                      max(1e-12, 1.0 - 2.0 * inset9))
        tx9 = min(15, max(0, int(math.floor(texture_u9 * 16.0))))
        ty9 = min(15, max(0, int(math.floor((1.0 - float(local_y9)) * 16.0))))
        mirror_x9 = 15 - tx9
        return bool(((int(alpha_mask) >> (ty9 * 16 + tx9)) & 1)
                    or ((int(alpha_mask) >> (ty9 * 16 + mirror_x9)) & 1))

    planes9 = (
        (d9[0] - d9[2], (c9[0] - c9[2]) - (o9[0] - o9[2]), 0),
        (d9[0] + d9[2], (c9[0] + c9[2] + 1.0) - (o9[0] + o9[2]), 1),
    )
    for denom9, numer9, plane9 in planes9:
        if abs(float(denom9)) < 1e-12:
            if abs(float(numer9)) > 1e-8:
                continue
            tt9 = 0.5 * (lo_t9 + hi_t9)
        else:
            tt9 = float(numer9) / float(denom9)
        if tt9 < lo_t9 or tt9 > hi_t9:
            continue
        point9 = o9 + d9 * tt9
        local9 = point9 - c9
        u9 = (0.5 * (local9[0] + local9[2]) if plane9 == 0
              else 0.5 * (local9[0] - local9[2] + 1.0))
        if opaque9(u9, local9[1]):
            hits9.append(tt9)
    return min(hits9) if hits9 else None


def _alpha_cutout_overlap_index(alpha_cutout_cells):
    if not alpha_cutout_cells:
        return {}

    cacheable9 = isinstance(alpha_cutout_cells, dict)
    cache_key9 = id(alpha_cutout_cells)
    if cacheable9:
        cached9 = _ALPHA_CUTOUT_INDEX_CACHE.get(cache_key9)
        if (cached9 is not None
                and cached9[0] is alpha_cutout_cells
                and alpha_cutout_cells == cached9[1]):
            _ALPHA_CUTOUT_INDEX_CACHE.pop(cache_key9)
            _ALPHA_CUTOUT_INDEX_CACHE[cache_key9] = cached9
            return cached9[2]

    overlap9 = {}
    for raw_source9, raw_bare9 in alpha_cutout_cells.items():
        if (not isinstance(raw_source9, tuple)
                or len(raw_source9) != 3):
            continue
        try:
            source9 = tuple(int(q9) for q9 in raw_source9)
        except (TypeError, ValueError, OverflowError):
            continue
        if source9 != raw_source9:
            continue
        bare9 = _bare(str(raw_bare9))
        mask9 = VANILLA_1_16_CROSS_ALPHA_MASKS.get(bare9)
        if mask9 is None:
            continue
        candidate9 = (
            source9, mask9, vanilla_cross_model_offset(source9, bare9))
        for current_y9 in (source9[1], source9[1] - 1):
            for current_x9 in range(source9[0] - 1, source9[0] + 2):
                for current_z9 in range(source9[2] - 1, source9[2] + 2):
                    overlap9.setdefault(
                        (current_x9, current_y9, current_z9), []).append(
                            candidate9)
    overlap9 = {cell9: tuple(candidates9)
                for cell9, candidates9 in overlap9.items()}

    if cacheable9:
        snapshot9 = dict(alpha_cutout_cells)
        _ALPHA_CUTOUT_INDEX_CACHE.pop(cache_key9, None)
        _ALPHA_CUTOUT_INDEX_CACHE[cache_key9] = (
            alpha_cutout_cells, snapshot9, overlap9)
        while len(_ALPHA_CUTOUT_INDEX_CACHE) > _ALPHA_CUTOUT_INDEX_CACHE_MAX:
            oldest9 = next(iter(_ALPHA_CUTOUT_INDEX_CACHE))
            _ALPHA_CUTOUT_INDEX_CACHE.pop(oldest9)
    return overlap9


def raycast_visual_alpha_cutout(
        origin_f, dir_f, max_dist, occ: OccupancyMap, *,
        ignore=frozenset(), extra_solid=frozenset(),
        alpha_cutout_cells=None, unknown_is_solid=False, thin_solid=None):
    o9 = np.asarray(origin_f, dtype=np.float64)
    d9 = np.asarray(dir_f, dtype=np.float64)
    norm9 = float(np.linalg.norm(d9))
    if norm9 < 1e-12:
        return None, 0.0
    d9 = d9 / norm9
    max_dist9 = float(max_dist)
    cutout_index9 = _alpha_cutout_overlap_index(alpha_cutout_cells)
    if not cutout_index9:
        return raycast(
            o9, d9, max_dist9, occ, ignore=ignore,
            extra_solid=extra_solid,
            unknown_is_solid=unknown_is_solid, thin_solid=thin_solid)
    cell9 = [int(math.floor(o9[i9])) for i9 in range(3)]
    step9, t_max9, t_delta9 = [0, 0, 0], [math.inf] * 3, [math.inf] * 3
    for i9 in range(3):
        if d9[i9] > 1e-12:
            step9[i9] = 1
            t_max9[i9] = (cell9[i9] + 1 - o9[i9]) / d9[i9]
            t_delta9[i9] = 1.0 / d9[i9]
        elif d9[i9] < -1e-12:
            step9[i9] = -1
            t_max9[i9] = (cell9[i9] - o9[i9]) / d9[i9]
            t_delta9[i9] = -1.0 / d9[i9]

    entry9 = 0.0
    while entry9 <= max_dist9 + 1e-9:
        current9 = tuple(cell9)
        exit9 = min(min(t_max9), max_dist9)
        if current9 not in ignore:
            if (occ.solid_at(current9) or current9 in extra_solid
                    or (unknown_is_solid and not occ.known(current9))):
                return current9, entry9
            top9 = None if thin_solid is None else thin_solid.get(current9)
            if top9 is not None and entry9 > 0.0:
                if min(o9[1] + d9[1] * entry9,
                       o9[1] + d9[1] * exit9) < float(top9):
                    return current9, entry9
            alpha_hits9 = []
            for source9, mask9, offset9 in cutout_index9.get(current9, ()):
                alpha_entry9 = _cross_alpha_entry(
                    o9, d9, source9, entry9, exit9, mask9,
                    model_offset=offset9)
                if alpha_entry9 is not None:
                    alpha_hits9.append((float(alpha_entry9), source9))
            if alpha_hits9:
                alpha_entry9, source9 = min(alpha_hits9)
                if alpha_entry9 <= max_dist9 + 1e-9:
                    return source9, alpha_entry9
        if min(t_max9) > max_dist9:
            return None, max_dist9
        axis9 = 0
        if t_max9[1] < t_max9[axis9]:
            axis9 = 1
        if t_max9[2] < t_max9[axis9]:
            axis9 = 2
        entry9 = float(t_max9[axis9])
        cell9[axis9] += step9[axis9]
        t_max9[axis9] += t_delta9[axis9]
    return None, max_dist9


def visible_face_aim(eye_f, block_xyz, occ: OccupancyMap, inset=1e-3):
    if not 0.0 < float(inset) < 0.5:
        raise ValueError("inset must be strictly between 0 and 0.5")

    block = tuple(int(v) for v in block_xyz)
    if not occ.known(block) or not occ.solid_at(block):
        return None

    eye = np.asarray(eye_f, dtype=np.float64)
    center = np.asarray(block, dtype=np.float64) + 0.5
    faces = (
        (1, 0, 0), (-1, 0, 0),
        (0, 1, 0), (0, -1, 0),
        (0, 0, 1), (0, 0, -1),
    )
    candidates = []
    for order, normal_t in enumerate(faces):
        normal = np.asarray(normal_t, dtype=np.float64)
        neighbour = tuple(block[i] + normal_t[i] for i in range(3))
        if not occ.known(neighbour) or occ.solid_at(neighbour):
            continue

        face_center = center + 0.5 * normal
        eye_from_face = eye - face_center
        outward = float(np.dot(eye_from_face, normal))
        if outward <= 1e-9:
            continue

        point = center + (0.5 - float(inset)) * normal
        ray = point - eye
        dist = float(np.linalg.norm(ray))
        if dist <= 1e-9:
            continue
        hit, _ = raycast(eye, ray, dist + 1e-6, occ,
                         unknown_is_solid=True)
        if hit != block:
            continue

        facing = outward / max(float(np.linalg.norm(eye_from_face)), 1e-12)
        candidates.append((-facing, dist, order, point, normal_t))

    if not candidates:
        return None
    _, _, _, point, normal = min(candidates, key=lambda q: q[:3])
    return tuple(float(v) for v in point), normal


def visual_occluders(occ: OccupancyMap, eye_f=None, *,
                     cutout_as_full_cube=True):
    eye_cell = (None if eye_f is None else
                tuple(int(math.floor(float(v))) for v in eye_f))
    return frozenset(
        c for c, raw in occ.grid.items()
        if c != eye_cell and not occ.solid_at(c)
        and _bare(str(raw)) not in {"air", "cave_air", "void_air"}
        and (bool(cutout_as_full_cube) or _bare(str(raw)) not in CUTOUT_GEOMETRY)
    )


SNOW_ONE_LAYER_RENDER_HEIGHT = 2.0 / 16.0
SNOW_LAYER_VISUAL_HEIGHT = 0.30
THIN_LAYER_GEOMETRY = {"snow"}


def thin_visual_occluders(occ: OccupancyMap, eye_f=None, *,
                          layer_height=SNOW_LAYER_VISUAL_HEIGHT):
    layer_height9 = float(layer_height)
    if not math.isfinite(layer_height9) or not 0.0 < layer_height9 <= 1.0:
        raise ValueError("snow layer_height must be finite and in (0, 1]")
    eye_cell = (None if eye_f is None else
                tuple(int(math.floor(float(v))) for v in eye_f))
    return {
        c: c[1] + layer_height9
        for c, raw in occ.grid.items()
        if c != eye_cell and not occ.solid_at(c)
        and _bare(str(raw)) in THIN_LAYER_GEOMETRY
    }


def visual_alpha_cutout_cells(occ: OccupancyMap, eye_f=None):
    return {
        tuple(cell9): _bare(str(raw9))
        for cell9, raw9 in occ.grid.items()
        if _bare(str(raw9)) in VANILLA_1_16_CROSS_ALPHA_MASKS
    }


def shared_visible_surface_scene(occ: OccupancyMap, eye_f=None, *,
                                 cutout_as_full_cube=False):
    eye_cell9 = (None if eye_f is None else
                 tuple(int(math.floor(float(value9))) for value9 in eye_f))
    cache_key9 = (
        id(occ.grid), int(occ.revision), bool(occ.water_occludes),
        bool(occ.leaves_occlude), bool(cutout_as_full_cube))
    cache9 = getattr(occ, "_visible_surface_scene_cache", None)
    cache_hit9 = bool(
        isinstance(cache9, dict) and cache9.get("key") == cache_key9)
    if not cache_hit9:
        base_visual9 = set()
        base_alpha9 = {}
        base_thin9 = {}
        for cell9, raw9 in occ.grid.items():
            cell9 = tuple(cell9)
            kind9 = _bare(str(raw9))
            solid9 = occ.solid_at(cell9)
            if kind9 in VANILLA_1_16_CROSS_ALPHA_MASKS:
                base_alpha9[cell9] = kind9
            if not solid9 and kind9 in THIN_LAYER_GEOMETRY:
                base_thin9[cell9] = (
                    cell9[1] + SNOW_ONE_LAYER_RENDER_HEIGHT)
            if (not solid9
                    and kind9 not in {"air", "cave_air", "void_air"}
                    and (bool(cutout_as_full_cube)
                         or kind9 not in CUTOUT_GEOMETRY)):
                base_visual9.add(cell9)
        cache9 = {
            "key": cache_key9,
            "base_visual_extra_solid": frozenset(base_visual9),
            "base_alpha_cutout_visual": MappingProxyType(dict(base_alpha9)),
            "base_thin_visual_solid": MappingProxyType(dict(base_thin9)),
            "base_candidate_face_blockers": frozenset(
                base_visual9 - set(base_alpha9) - set(base_thin9)),
        }
        occ._visible_surface_scene_cache = cache9
    visual9 = cache9["base_visual_extra_solid"]
    thin9 = cache9["base_thin_visual_solid"]
    face_blockers9 = cache9["base_candidate_face_blockers"]
    if eye_cell9 is not None:
        if eye_cell9 in visual9:
            visual9 = visual9 - frozenset((eye_cell9,))
        if eye_cell9 in thin9:
            thin9 = MappingProxyType({
                cell9: top9 for cell9, top9 in thin9.items()
                if cell9 != eye_cell9})
        if eye_cell9 in face_blockers9:
            face_blockers9 = face_blockers9 - frozenset((eye_cell9,))
    return {
        "visual_extra_solid": frozenset(visual9),
        "alpha_cutout_visual": cache9["base_alpha_cutout_visual"],
        "thin_visual_solid": thin9,
        "candidate_face_blockers": frozenset(face_blockers9),
        "occupancy_grid_passes": int(not cache_hit9),
        "static_scene_cache_hit": int(cache_hit9),
        "occupancy_revision": int(occ.revision),
    }


CHOSEN_VISIBLE_SURFACE_MASK_SEMANTIC = (
    "chosen_instance_actual_visible_surface/v1")


def _normalise_visible_surface_aabbs(target_aabbs):
    boxes9 = []
    for index9, raw9 in enumerate(target_aabbs):
        arr9 = np.asarray(raw9, dtype=np.float64)
        if arr9.shape == (2, 3):
            lo9, hi9 = arr9[0], arr9[1]
        elif arr9.shape == (6,):
            lo9, hi9 = arr9[:3], arr9[3:]
        else:
            raise ValueError(
                f"target AABB {index9} must have shape (2,3) or (6,), "
                f"got {arr9.shape}")
        if not np.isfinite(arr9).all():
            raise ValueError(f"target AABB {index9} contains a non-finite value")
        if np.any(hi9 <= lo9):
            raise ValueError(
                f"target AABB {index9} must have strictly positive XYZ extent")
        boxes9.append((lo9.copy(), hi9.copy()))
    if not boxes9:
        raise ValueError("target_aabbs must contain at least one box")
    return tuple(boxes9)


def _aabb_owned_cells(aabbs9):
    cells9 = set()
    for lo9, hi9 in aabbs9:
        for xx9 in range(int(math.floor(lo9[0])), int(math.ceil(hi9[0]))):
            for yy9 in range(int(math.floor(lo9[1])), int(math.ceil(hi9[1]))):
                for zz9 in range(int(math.floor(lo9[2])), int(math.ceil(hi9[2]))):
                    cells9.add((xx9, yy9, zz9))
    return frozenset(cells9)


_AABB_EDGE_INDICES = (
    (0, 1), (0, 2), (0, 4),
    (1, 3), (1, 5),
    (2, 3), (2, 6),
    (3, 7),
    (4, 5), (4, 6),
    (5, 7), (6, 7),
)


def _project_aabb_candidate_mask(eye9, right9, up9, fwd9, focal9,
                                 lo9, hi9, near_clip9=0.05):
    corners9 = np.asarray([
        (xx9, yy9, zz9)
        for xx9 in (lo9[0], hi9[0])
        for yy9 in (lo9[1], hi9[1])
        for zz9 in (lo9[2], hi9[2])
    ], dtype=np.float64)
    rel9 = corners9 - eye9
    depths9 = rel9 @ fwd9
    clipped9 = [corners9[index9]
                for index9 in range(8)
                if depths9[index9] >= near_clip9]
    for left9, right_index9 in _AABB_EDGE_INDICES:
        d09, d19 = float(depths9[left9]), float(depths9[right_index9])
        if ((d09 < near_clip9 and d19 < near_clip9)
                or (d09 >= near_clip9 and d19 >= near_clip9)):
            continue
        alpha9 = ((near_clip9 - d09) / (d19 - d09))
        clipped9.append(
            corners9[left9] + alpha9 * (corners9[right_index9] - corners9[left9]))
    if len(clipped9) < 3:
        return None

    clipped9 = np.asarray(clipped9, dtype=np.float64)
    rel9 = clipped9 - eye9
    depths9 = rel9 @ fwd9
    px9 = W_PX / 2.0 + focal9 * (rel9 @ right9) / depths9
    py9 = H_PX / 2.0 - focal9 * (rel9 @ up9) / depths9
    points9 = np.stack((px9, py9), axis=1)
    points9[:, 0] = np.clip(points9[:, 0], -4 * W_PX, 5 * W_PX)
    points9[:, 1] = np.clip(points9[:, 1], -4 * H_PX, 5 * H_PX)
    if ((points9[:, 0] < 0).all() or (points9[:, 0] >= W_PX).all()
            or (points9[:, 1] < 0).all() or (points9[:, 1] >= H_PX).all()):
        return None
    polygon9 = cv2.convexHull(points9.astype(np.float32)).astype(np.int32)
    mask9 = np.zeros((H_PX, W_PX), dtype=np.uint8)
    cv2.fillConvexPoly(mask9, polygon9, 1)
    return mask9 if mask9.any() else None


def projected_full_cube_candidate_stats(
        eye_f, yaw_deg, pitch_deg, block_xyz, *, fy=None,
        pixel_exclusion_mask=None):
    cell9 = tuple(int(value9) for value9 in block_xyz)
    eye9 = np.asarray(tuple(float(value9) for value9 in eye_f), dtype=np.float64)
    if eye9.shape != (3,) or not np.isfinite(eye9).all():
        raise ValueError("eye_f must contain three finite XYZ coordinates")
    focal9 = float(FY if fy is None else fy)
    right9, up9, fwd9 = _basis(float(yaw_deg), float(pitch_deg))
    mask9 = _project_aabb_candidate_mask(
        eye9, right9, up9, fwd9, focal9,
        np.asarray(cell9, dtype=np.float64),
        np.asarray(tuple(value9 + 1 for value9 in cell9), dtype=np.float64))
    if mask9 is None:
        return {"projected_area_px": 0, "projected_short_side_px": 0,
                "projected_bbox_px": None}
    if pixel_exclusion_mask is not None:
        exclusion9 = np.asarray(pixel_exclusion_mask)
        if exclusion9.shape != (H_PX, W_PX):
            raise ValueError(
                f"pixel_exclusion_mask must have shape {(H_PX, W_PX)}")
        mask9[np.asarray(exclusion9, dtype=bool)] = 0
    ys9, xs9 = np.nonzero(mask9)
    if not len(xs9):
        return {"projected_area_px": 0, "projected_short_side_px": 0,
                "projected_bbox_px": None}
    bbox9 = [int(xs9.min()), int(ys9.min()),
             int(xs9.max()), int(ys9.max())]
    return {
        "projected_area_px": int(len(xs9)),
        "projected_short_side_px": int(min(
            bbox9[2] - bbox9[0] + 1, bbox9[3] - bbox9[1] + 1)),
        "projected_bbox_px": bbox9,
    }


def _ray_aabb_surface_entry(origin9, direction9, lo9, hi9):
    near9, far9 = -math.inf, math.inf
    near_normal9 = far_normal9 = None
    for axis9 in range(3):
        value9 = float(direction9[axis9])
        origin_axis9 = float(origin9[axis9])
        if abs(value9) < 1e-12:
            if origin_axis9 < lo9[axis9] or origin_axis9 > hi9[axis9]:
                return None
            continue
        t_lo9 = (float(lo9[axis9]) - origin_axis9) / value9
        t_hi9 = (float(hi9[axis9]) - origin_axis9) / value9
        lo_normal9 = [0, 0, 0]
        hi_normal9 = [0, 0, 0]
        lo_normal9[axis9] = -1
        hi_normal9[axis9] = 1
        if t_lo9 > t_hi9:
            t_lo9, t_hi9 = t_hi9, t_lo9
            lo_normal9, hi_normal9 = hi_normal9, lo_normal9
        if t_lo9 > near9:
            near9, near_normal9 = t_lo9, tuple(lo_normal9)
        if t_hi9 < far9:
            far9, far_normal9 = t_hi9, tuple(hi_normal9)
        if far9 < near9 - 1e-10:
            return None
    if far9 < 0.0:
        return None
    if near9 >= 0.0:
        return float(near9), near_normal9
    return float(far9), far_normal9


_AUTO_THIN_VISUAL_OCCLUDERS = object()


_DEPTH_FRAME = None


def _build_depth_owner_index(cellmap9, valid9):
    started9 = time.perf_counter_ns()
    flat_pixels9 = np.flatnonzero(valid9.reshape(-1)).astype(
        np.int32, copy=False)
    if not len(flat_pixels9):
        return {}, {
            "owner_index_build_count": 1,
            "owner_index_build_ms": round(
                (time.perf_counter_ns() - started9) / 1_000_000.0, 3),
            "owner_index_valid_pixel_count": 0,
            "owner_index_cell_count": 0,
        }
    cells9 = cellmap9.reshape(-1, 3)[flat_pixels9]
    order9 = np.lexsort((flat_pixels9, cells9[:, 2], cells9[:, 1], cells9[:, 0]))
    sorted_cells9 = cells9[order9]
    sorted_pixels9 = flat_pixels9[order9]
    starts9 = np.r_[0, 1 + np.flatnonzero(
        np.any(sorted_cells9[1:] != sorted_cells9[:-1], axis=1))]
    ends9 = np.r_[starts9[1:], len(sorted_pixels9)]
    owner_index9 = {
        tuple(int(value9) for value9 in sorted_cells9[start9]):
            sorted_pixels9[start9:end9]
        for start9, end9 in zip(starts9, ends9)
    }
    return owner_index9, {
        "owner_index_build_count": 1,
        "owner_index_build_ms": round(
            (time.perf_counter_ns() - started9) / 1_000_000.0, 3),
        "owner_index_valid_pixel_count": int(len(flat_pixels9)),
        "owner_index_cell_count": int(len(owner_index9)),
    }


def set_depth_frame(eye_f, yaw_deg, pitch_deg, cellmap, valid, *,
                    near, far, source="", build_owner_index=True,
                    audit_owner_index=False):
    global _DEPTH_FRAME
    cm9 = np.ascontiguousarray(cellmap, dtype=np.int64)
    vd9 = np.ascontiguousarray(valid, dtype=bool)
    if cm9.shape != (H_PX, W_PX, 3) or vd9.shape != (H_PX, W_PX):
        raise ValueError(
            f"depth frame requires cellmap{(H_PX, W_PX, 3)}+valid{(H_PX, W_PX)}")
    owner_index9 = None
    owner_stats9 = {
        "owner_index_build_count": 0,
        "owner_index_build_ms": 0.0,
        "owner_index_valid_pixel_count": int(vd9.sum()),
        "owner_index_cell_count": 0,
    }
    if bool(build_owner_index):
        owner_index9, owner_stats9 = _build_depth_owner_index(cm9, vd9)
    owner_stats9.update({
        "owner_index_lookup_count": 0,
        "owner_index_owner_miss_count": 0,
        "owner_index_compact_exclusion_filter_count": 0,
        "full_cellmap_equality_scan_count": 0,
        "owner_index_direct_ab_compare_count": 0,
        "owner_index_direct_ab_mismatch_count": 0,
    })
    _DEPTH_FRAME = {
        "eye": np.asarray(eye_f, dtype=np.float64),
        "yaw": float(yaw_deg), "pitch": float(pitch_deg),
        "cellmap": cm9, "valid": vd9,
        "owner_index": owner_index9,
        "owner_stats": owner_stats9,
        "audit_owner_index": bool(audit_owner_index),
        "near": float(near), "far": float(far), "source": str(source),
    }


def clear_depth_frame():
    global _DEPTH_FRAME
    _DEPTH_FRAME = None


def active_depth_frame():
    return _DEPTH_FRAME


def _depth_frame_match(eye9, yaw_deg, pitch_deg):
    frame9 = _DEPTH_FRAME
    if frame9 is None:
        return None
    eye_arr9 = np.asarray(eye9, dtype=np.float64)
    if (eye_arr9.shape == (3,)
            and abs(float(yaw_deg) - frame9["yaw"]) <= 1e-6
            and abs(float(pitch_deg) - frame9["pitch"]) <= 1e-6
            and float(np.max(np.abs(eye_arr9 - frame9["eye"]))) <= 1e-6):
        return frame9
    return None


def depth_owner_pixel_indices(eye_f, yaw_deg, pitch_deg, cell_xyz, *,
                              pixel_exclusion_mask=None):
    frame9 = _depth_frame_match(eye_f, yaw_deg, pitch_deg)
    if frame9 is None or frame9.get("owner_index") is None:
        return None
    stats9 = frame9["owner_stats"]
    stats9["owner_index_lookup_count"] += 1
    cell9 = tuple(int(value9) for value9 in cell_xyz)
    raw_pixels9 = frame9["owner_index"].get(cell9)
    if raw_pixels9 is None:
        stats9["owner_index_owner_miss_count"] += 1
        pixels9 = np.empty(0, dtype=np.int32)
    else:
        pixels9 = raw_pixels9
    exclusion9 = None
    if pixel_exclusion_mask is not None:
        exclusion9 = np.asarray(pixel_exclusion_mask)
        if exclusion9.shape != (H_PX, W_PX):
            raise ValueError(
                f"pixel_exclusion_mask must have shape {(H_PX, W_PX)}")
        if len(pixels9):
            stats9["owner_index_compact_exclusion_filter_count"] += 1
            pixels9 = pixels9[
                ~exclusion9.reshape(-1)[pixels9].astype(bool)]
    if frame9.get("audit_owner_index"):
        cx9, cy9, cz9 = cell9
        scan_member9 = (
            frame9["valid"]
            & (frame9["cellmap"][..., 0] == cx9)
            & (frame9["cellmap"][..., 1] == cy9)
            & (frame9["cellmap"][..., 2] == cz9))
        if exclusion9 is not None:
            scan_member9 &= ~exclusion9.astype(bool)
        scan_pixels9 = np.flatnonzero(scan_member9.reshape(-1)).astype(
            np.int32, copy=False)
        stats9["owner_index_direct_ab_compare_count"] += 1
        if not np.array_equal(pixels9, scan_pixels9):
            stats9["owner_index_direct_ab_mismatch_count"] += 1
            raise AssertionError(
                "depth owner index differs from the full cellmap equality scan "
                f"for cell {cell9}: {len(pixels9)} != {len(scan_pixels9)}")
    return pixels9


def backproject_depth_to_cells(depth_window, eye_f, yaw_deg, pitch_deg, *,
                               near, far, fy=None,
                               pixel_exclusion_mask=None):
    d9 = np.asarray(depth_window, dtype=np.float64)
    if d9.shape != (H_PX, W_PX):
        raise ValueError(f"depth must be {(H_PX, W_PX)}, got {d9.shape}")
    focal9 = float(FY if fy is None else fy)
    if not math.isfinite(focal9) or focal9 <= 0.0:
        raise ValueError("fy must be finite and positive")
    eye9 = np.asarray(eye_f, dtype=np.float64)
    if eye9.shape != (3,) or not np.isfinite(eye9).all():
        raise ValueError("eye_f must contain three finite XYZ coordinates")
    right9, up9, fwd9 = _basis(float(yaw_deg), float(pitch_deg))
    ax9 = (np.arange(W_PX, dtype=np.float64) + 0.5 - W_PX / 2.0) / focal9
    ay9 = (np.arange(H_PX, dtype=np.float64) + 0.5 - H_PX / 2.0) / focal9
    dir_unnorm9 = (fwd9[None, None, :]
                   + ax9[None, :, None] * right9[None, None, :]
                   - ay9[:, None, None] * up9[None, None, :])
    dir_norm9 = dir_unnorm9 / np.maximum(
        np.linalg.norm(dir_unnorm9, axis=-1, keepdims=True), 1e-12)
    n9, f9 = float(near), float(far)
    m10_9 = -(f9 + n9) / (f9 - n9)
    m14_9 = -2.0 * f9 * n9 / (f9 - n9)
    ndc9 = 2.0 * d9 - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z_view9 = m14_9 / (ndc9 + m10_9)
    valid9 = np.isfinite(d9) & (d9 < 1.0) & np.isfinite(z_view9) & (z_view9 > 0.0)
    if pixel_exclusion_mask is not None:
        ex9 = np.asarray(pixel_exclusion_mask)
        if ex9.shape != (H_PX, W_PX):
            raise ValueError(f"pixel_exclusion_mask must be {(H_PX, W_PX)}")
        valid9 = valid9 & ~ex9.astype(bool)
    z_safe9 = np.where(valid9, z_view9, 0.0)
    world_hit9 = eye9[None, None, :] + z_safe9[:, :, None] * dir_unnorm9
    cell9 = np.floor(world_hit9 + 1e-6 * dir_norm9).astype(np.int64)
    cell9[~valid9] = np.iinfo(np.int64).min
    return cell9, valid9


def backproject_depth_owner_pixels_for_blocks(
        depth_window, eye_f, yaw_deg, pitch_deg, block_cells, *,
        near, far, fy=None, pixel_exclusion_mask=None):
    d9 = np.asarray(depth_window, dtype=np.float64)
    if d9.shape != (H_PX, W_PX):
        raise ValueError(f"depth must be {(H_PX, W_PX)}, got {d9.shape}")
    eye9 = np.asarray(tuple(float(value9) for value9 in eye_f), dtype=np.float64)
    if eye9.shape != (3,) or not np.isfinite(eye9).all():
        raise ValueError("eye_f must contain three finite XYZ coordinates")
    focal9 = float(FY if fy is None else fy)
    if not math.isfinite(focal9) or focal9 <= 0.0:
        raise ValueError("fy must be finite and positive")
    cells9 = tuple(dict.fromkeys(
        tuple(int(value9) for value9 in raw9) for raw9 in block_cells))
    if any(len(cell9) != 3 for cell9 in cells9):
        raise ValueError("block_cells must contain xyz triples")
    if not cells9:
        return {}
    right9, up9, fwd9 = _basis(float(yaw_deg), float(pitch_deg))
    candidates9 = np.zeros((H_PX, W_PX), dtype=np.uint8)
    for cell9 in cells9:
        projected9 = _project_aabb_candidate_mask(
            eye9, right9, up9, fwd9, focal9,
            np.asarray(cell9, dtype=np.float64),
            np.asarray(tuple(value9 + 1 for value9 in cell9),
                       dtype=np.float64))
        if projected9 is not None:
            candidates9 |= projected9
    if pixel_exclusion_mask is not None:
        exclusion9 = np.asarray(pixel_exclusion_mask)
        if exclusion9.shape != (H_PX, W_PX):
            raise ValueError(
                f"pixel_exclusion_mask must be {(H_PX, W_PX)}")
        candidates9[np.asarray(exclusion9, dtype=bool)] = 0
    ys9, xs9 = np.nonzero(candidates9)
    empty9 = np.empty(0, dtype=np.int32)
    if not len(xs9):
        return {cell9: empty9.copy() for cell9 in cells9}

    ax9 = (xs9.astype(np.float64) + 0.5 - W_PX / 2.0) / focal9
    ay9 = (ys9.astype(np.float64) + 0.5 - H_PX / 2.0) / focal9
    directions9 = (fwd9[None, :]
                   + ax9[:, None] * right9[None, :]
                   - ay9[:, None] * up9[None, :])
    direction_norm9 = directions9 / np.maximum(
        np.linalg.norm(directions9, axis=-1, keepdims=True), 1e-12)
    selected_depth9 = d9[ys9, xs9]
    n9, f9 = float(near), float(far)
    m10_9 = -(f9 + n9) / (f9 - n9)
    m14_9 = -2.0 * f9 * n9 / (f9 - n9)
    ndc9 = 2.0 * selected_depth9 - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        z_view9 = m14_9 / (ndc9 + m10_9)
    valid9 = (np.isfinite(selected_depth9)
              & (selected_depth9 < 1.0)
              & np.isfinite(z_view9)
              & (z_view9 > 0.0))
    if not valid9.any():
        return {cell9: empty9.copy() for cell9 in cells9}
    valid_y9 = ys9[valid9]
    valid_x9 = xs9[valid9]
    hits9 = (eye9[None, :]
             + z_view9[valid9, None] * directions9[valid9])
    owners9 = np.floor(
        hits9 + 1e-6 * direction_norm9[valid9]).astype(np.int64)
    flat9 = (valid_y9.astype(np.int64) * W_PX
             + valid_x9.astype(np.int64)).astype(np.int32)
    return {
        cell9: flat9[np.all(
            owners9 == np.asarray(cell9, dtype=np.int64)[None, :], axis=1)]
        for cell9 in cells9
    }


def _visible_surface_mask_aabbs_from_depth(
        frame9, eye9, yaw_deg, pitch_deg, target_aabbs, target_cells,
        pixel_exclusion_mask, fy, unknown_is_solid):
    boxes9 = _normalise_visible_surface_aabbs(target_aabbs)
    focal9 = float(FY if fy is None else fy)
    owned9 = (_aabb_owned_cells(boxes9) if target_cells is None else
              frozenset(tuple(int(value9) for value9 in cell9)
                        for cell9 in target_cells))
    right9, up9, fwd9 = _basis(float(yaw_deg), float(pitch_deg))
    mask9 = np.zeros((H_PX, W_PX), dtype=np.uint8)
    owner_index9 = frame9.get("owner_index")
    if owner_index9 is not None:
        mask_flat9 = mask9.reshape(-1)
        for cell9 in owned9:
            pixels9 = depth_owner_pixel_indices(
                eye9, yaw_deg, pitch_deg, cell9,
                pixel_exclusion_mask=pixel_exclusion_mask)
            if pixels9 is not None and len(pixels9):
                mask_flat9[pixels9] = 1
    elif owned9:
        cellmap9 = frame9["cellmap"]
        valid9 = frame9["valid"]
        member9 = np.zeros((H_PX, W_PX), dtype=bool)
        for cx9, cy9, cz9 in owned9:
            frame9["owner_stats"][
                "full_cellmap_equality_scan_count"] += 1
            member9 |= ((cellmap9[..., 0] == cx9)
                        & (cellmap9[..., 1] == cy9)
                        & (cellmap9[..., 2] == cz9))
        mask9[member9 & valid9] = 1
    if pixel_exclusion_mask is not None and owner_index9 is None:
        exclusion9 = np.asarray(pixel_exclusion_mask)
        if exclusion9.shape != (H_PX, W_PX):
            raise ValueError(
                f"pixel_exclusion_mask must have shape {(H_PX, W_PX)}")
        mask9[np.asarray(exclusion9, dtype=bool)] = 0

    def nearest_target9(direction9):
        nearest9 = None
        for index9, (lo9, hi9) in enumerate(boxes9):
            hit9 = _ray_aabb_surface_entry(frame9["eye"], direction9, lo9, hi9)
            if hit9 is None:
                continue
            distance9, normal9 = hit9
            key9 = (float(distance9), int(index9))
            if nearest9 is None or key9 < nearest9[0]:
                nearest9 = (key9, int(index9), normal9)
        if nearest9 is None:
            return None
        return nearest9[0][0], nearest9[1], nearest9[2]

    ys9, xs9 = np.nonzero(mask9)
    bbox9 = point9 = representative_pixel9 = representative_normal9 = None
    if len(xs9):
        bbox9 = (int(xs9.min()), int(ys9.min()),
                 int(xs9.max()), int(ys9.max()))
        centroid_x9, centroid_y9 = float(xs9.mean()), float(ys9.mean())
        order9 = ((xs9 - centroid_x9) ** 2 + (ys9 - centroid_y9) ** 2)
        for rank9 in np.lexsort((xs9, ys9, order9)):
            rep_y9, rep_x9 = int(ys9[rank9]), int(xs9[rank9])
            direction9 = (fwd9
                          + ((rep_x9 + 0.5 - W_PX / 2.0) / focal9) * right9
                          - ((rep_y9 + 0.5 - H_PX / 2.0) / focal9) * up9)
            direction9 /= max(float(np.linalg.norm(direction9)), 1e-12)
            target_hit9 = nearest_target9(direction9)
            if target_hit9 is not None:
                distance9, _, representative_normal9 = target_hit9
                point9 = tuple(float(v9) for v9 in
                               (frame9["eye"] + direction9 * float(distance9)))
                representative_pixel9 = (rep_x9, rep_y9)
                break
        if representative_pixel9 is None:
            return np.zeros((H_PX, W_PX), dtype=np.uint8), None, None, {
                "mask_semantic": CHOSEN_VISIBLE_SURFACE_MASK_SEMANTIC,
                "mask_shape": [H_PX, W_PX], "mask_dtype": "uint8_binary_0_1",
                "pixel_count": 0, "bbox_xyxy": None,
                "representative_pixel_xy": None,
                "representative_visible_point": None,
                "representative_surface_normal": None,
                "occlusion_backend": "native_depth_backprojection",
                "zero_area_is_valid_absence": True,
            }

    meta9 = {
        "mask_semantic": CHOSEN_VISIBLE_SURFACE_MASK_SEMANTIC,
        "mask_shape": [H_PX, W_PX],
        "mask_dtype": "uint8_binary_0_1",
        "pixel_count": int(mask9.sum()),
        "bbox_xyxy": None if bbox9 is None else list(bbox9),
        "representative_pixel_xy": (
            None if representative_pixel9 is None
            else list(representative_pixel9)),
        "representative_visible_point": (
            None if point9 is None else list(point9)),
        "representative_surface_normal": (
            None if representative_normal9 is None
            else list(representative_normal9)),
        "target_aabb_count": len(boxes9),
        "target_owned_cell_count": len(owned9),
        "projected_aabb_count": len(boxes9),
        "candidate_pixel_count": int(mask9.sum()),
        "tested_ray_count": 0,
        "projection_contract": "per_aabb_hull_union_no_cross_aabb_fill",
        "occlusion_contract": "native_depth_first_hit_backprojection",
        "occlusion_backend": "native_depth_backprojection",
        "unknown_is_solid": bool(unknown_is_solid),
        "thin_layer_contract": "native_depth_exact_rasterised_geometry",
        "alpha_cutout_contract": "native_depth_exact_rasterised_alpha",
        "alpha_cutout_cell_count": 0,
        "zero_area_is_valid_absence": True,
    }
    return mask9, point9, bbox9, meta9


def visible_surface_mask_aabbs(
        eye_f, yaw_deg, pitch_deg, target_aabbs, occ: OccupancyMap, *,
        target_cells=None, fy=None, extra_visual_solid=None,
        alpha_cutout_visual=None, unknown_is_solid=True,
        pixel_exclusion_mask=None,
        thin_visual_solid=_AUTO_THIN_VISUAL_OCCLUDERS):
    boxes9 = _normalise_visible_surface_aabbs(target_aabbs)
    eye9 = np.asarray(eye_f, dtype=np.float64)
    if eye9.shape != (3,) or not np.isfinite(eye9).all():
        raise ValueError("eye_f must contain three finite XYZ coordinates")
    focal9 = float(FY if fy is None else fy)
    if not math.isfinite(focal9) or focal9 <= 0.0:
        raise ValueError("fy must be finite and positive")
    frame9 = _depth_frame_match(eye9, yaw_deg, pitch_deg)
    if frame9 is not None and (fy is None or float(fy) == float(FY)):
        return _visible_surface_mask_aabbs_from_depth(
            frame9, eye9, yaw_deg, pitch_deg, target_aabbs, target_cells,
            pixel_exclusion_mask, fy, unknown_is_solid)
    owned9 = (_aabb_owned_cells(boxes9) if target_cells is None else
              frozenset(tuple(int(value9) for value9 in cell9)
                        for cell9 in target_cells))
    right9, up9, fwd9 = _basis(float(yaw_deg), float(pitch_deg))

    candidate9 = np.zeros((H_PX, W_PX), dtype=np.uint8)
    projected_count9 = 0
    for lo9, hi9 in boxes9:
        primitive9 = _project_aabb_candidate_mask(
            eye9, right9, up9, fwd9, focal9, lo9, hi9)
        if primitive9 is not None:
            candidate9 |= primitive9
            projected_count9 += 1
    if pixel_exclusion_mask is not None:
        exclusion9 = np.asarray(pixel_exclusion_mask)
        if exclusion9.shape != (H_PX, W_PX):
            raise ValueError(
                f"pixel_exclusion_mask must have shape {(H_PX, W_PX)}")
        candidate9[np.asarray(exclusion9, dtype=bool)] = 0

    visual_extra9 = (visual_occluders(occ, eye9)
                     if extra_visual_solid is None
                     else frozenset(tuple(int(v9) for v9 in cell9)
                                    for cell9 in extra_visual_solid))
    visual_extra9 = visual_extra9 - owned9
    alpha_cutout9 = {
        tuple(int(value9) for value9 in cell9): _bare(str(kind9))
        for cell9, kind9 in dict(alpha_cutout_visual or {}).items()
        if tuple(int(value9) for value9 in cell9) not in owned9
    }
    if alpha_cutout9:
        visual_extra9 = visual_extra9 - frozenset(alpha_cutout9)
    raw_thin9 = (
        thin_visual_occluders(
            occ, eye9, layer_height=SNOW_ONE_LAYER_RENDER_HEIGHT)
        if thin_visual_solid is _AUTO_THIN_VISUAL_OCCLUDERS else
        dict(thin_visual_solid or {}))
    thin9 = {tuple(int(value9) for value9 in cell9): float(top9)
             for cell9, top9 in raw_thin9.items()
             if tuple(int(value9) for value9 in cell9) not in owned9}
    if thin9:
        visual_extra9 = visual_extra9 - frozenset(thin9)
    mask9 = np.zeros((H_PX, W_PX), dtype=np.uint8)
    candidate_ys9, candidate_xs9 = np.nonzero(candidate9)

    def nearest_target9(direction9):
        nearest9 = None
        for index9, (lo9, hi9) in enumerate(boxes9):
            hit9 = _ray_aabb_surface_entry(eye9, direction9, lo9, hi9)
            if hit9 is None:
                continue
            distance9, normal9 = hit9
            key9 = (float(distance9), int(index9))
            if nearest9 is None or key9 < nearest9[0]:
                nearest9 = (key9, int(index9), normal9)
        if nearest9 is None:
            return None
        return nearest9[0][0], nearest9[1], nearest9[2]

    for yy9, xx9 in zip(candidate_ys9, candidate_xs9):
        direction9 = (fwd9
                      + ((float(xx9) + 0.5 - W_PX / 2.0) / focal9) * right9
                      - ((float(yy9) + 0.5 - H_PX / 2.0) / focal9) * up9)
        direction9 /= max(float(np.linalg.norm(direction9)), 1e-12)
        target_hit9 = nearest_target9(direction9)
        if target_hit9 is None:
            continue
        distance9, _, _ = target_hit9
        blocker9, _ = raycast_visual_alpha_cutout(
            eye9, direction9, max(0.0, float(distance9) - 1e-7), occ,
            ignore=owned9, extra_solid=visual_extra9,
            alpha_cutout_cells=alpha_cutout9,
            unknown_is_solid=bool(unknown_is_solid), thin_solid=thin9)
        if blocker9 is None:
            mask9[int(yy9), int(xx9)] = 1

    ys9, xs9 = np.nonzero(mask9)
    bbox9 = point9 = representative_pixel9 = representative_normal9 = None
    if len(xs9):
        bbox9 = (int(xs9.min()), int(ys9.min()),
                 int(xs9.max()), int(ys9.max()))
        centroid_x9, centroid_y9 = float(xs9.mean()), float(ys9.mean())
        order9 = ((xs9 - centroid_x9) ** 2 + (ys9 - centroid_y9) ** 2)
        selected9 = int(np.lexsort((xs9, ys9, order9))[0])
        rep_y9, rep_x9 = int(ys9[selected9]), int(xs9[selected9])
        direction9 = (fwd9
                      + ((rep_x9 + 0.5 - W_PX / 2.0) / focal9) * right9
                      - ((rep_y9 + 0.5 - H_PX / 2.0) / focal9) * up9)
        direction9 /= max(float(np.linalg.norm(direction9)), 1e-12)
        distance9, _, representative_normal9 = nearest_target9(direction9)
        point_array9 = eye9 + direction9 * float(distance9)
        point9 = tuple(float(value9) for value9 in point_array9)
        representative_pixel9 = (rep_x9, rep_y9)

    meta9 = {
        "mask_semantic": CHOSEN_VISIBLE_SURFACE_MASK_SEMANTIC,
        "mask_shape": [H_PX, W_PX],
        "mask_dtype": "uint8_binary_0_1",
        "pixel_count": int(mask9.sum()),
        "bbox_xyxy": None if bbox9 is None else list(bbox9),
        "representative_pixel_xy": (
            None if representative_pixel9 is None
            else list(representative_pixel9)),
        "representative_visible_point": (
            None if point9 is None else list(point9)),
        "representative_surface_normal": (
            None if representative_normal9 is None
            else list(representative_normal9)),
        "target_aabb_count": len(boxes9),
        "target_owned_cell_count": len(owned9),
        "projected_aabb_count": int(projected_count9),
        "candidate_pixel_count": int(candidate9.sum()),
        "tested_ray_count": int(candidate9.sum()),
        "projection_contract": "per_aabb_hull_union_no_cross_aabb_fill",
        "occlusion_contract": (
            "nearest_target_aabb_vs_occupancy_dda_first_entry"),
        "unknown_is_solid": bool(unknown_is_solid),
        "thin_layer_contract": "natural_snow_one_layer_height_2_16",
        "alpha_cutout_contract": (
            "vanilla_1_16_crossed_texture_alpha_first_hit"
            if alpha_cutout9 else "no_modeled_alpha_cutout_cells"),
        "alpha_cutout_cell_count": int(len(alpha_cutout9)),
        "zero_area_is_valid_absence": True,
    }
    return mask9, point9, bbox9, meta9


def visible_surface_mask_blocks(
        eye_f, yaw_deg, pitch_deg, target_blocks, occ: OccupancyMap, **kwargs):
    blocks9 = tuple(tuple(int(value9) for value9 in block9)
                    for block9 in target_blocks)
    if not blocks9:
        raise ValueError("target_blocks must contain at least one cell")
    aabbs9 = tuple((block9, tuple(value9 + 1 for value9 in block9))
                   for block9 in blocks9)
    if "target_cells" in kwargs:
        raise TypeError("visible_surface_mask_blocks owns target_cells")
    return visible_surface_mask_aabbs(
        eye_f, yaw_deg, pitch_deg, aabbs9, occ,
        target_cells=blocks9, **kwargs)


def visible_face_within_reach(eye_f, block_xyz, occ: OccupancyMap, reach=5.2):
    target = tuple(int(v) for v in block_xyz)
    face = visible_face_aim(eye_f, target, occ)
    if face is None:
        return None
    point, normal = face
    direction = tuple(float(point[i]) - float(eye_f[i]) for i in range(3))
    hit, entry = raycast(eye_f, direction, float(reach), occ,
                         unknown_is_solid=True)
    if hit != target or float(entry) > float(reach) + 1e-9:
        return None
    return point, normal, float(entry)


def mine_stance_known_clear(occ: OccupancyMap, state) -> bool:
    x9, feet_y9, z9 = (int(v9) for v9 in state)
    support9 = (x9, feet_y9 - 1, z9)
    if not occ.known(support9):
        return False
    raw_support9 = occ.grid.get(support9)
    if raw_support9 is None:
        return False
    bare_support9 = _bare(raw_support9)
    if (bare_support9 in PASSTHROUGH
            or bare_support9 in ("water", "lava", "bubble_column")
            or bare_support9.endswith("_leaves")
            or bare_support9.endswith("_log")):
        return False
    for y9 in (feet_y9, feet_y9 + 1):
        cell9 = (x9, y9, z9)
        if not occ.known(cell9):
            return False
        raw9 = occ.grid.get(cell9)
        if raw9 is not None and _bare(raw9) not in PASSTHROUGH:
            return False
    return True


def reachable_mine_stance(occ: OccupancyMap, start_feet, target_xyz,
                           reach=5.2, max_nodes=6000, excluded_feet=(),
                           preferred_feet=None,
                           max_surface_distance=None,
                           prefer_outer_shell=False,
                           min_standoff_surface_distance=None):
    import heapq

    target = tuple(int(v) for v in target_xyz)
    excluded_feet = frozenset(tuple(int(v) for v in q) for q in excluded_feet)
    preferred9 = None
    if preferred_feet is not None:
        try:
            candidate9 = tuple(int(v9) for v9 in preferred_feet)
            exact9 = (len(candidate9) == 3
                      and all(float(preferred_feet[i9]) == float(candidate9[i9])
                              for i9 in range(3)))
        except (TypeError, ValueError, OverflowError):
            candidate9, exact9 = None, False
        if exact9:
            preferred9 = candidate9
    if not occ.known(target) or not occ.solid_at(target):
        return None
    reach = float(reach)
    if reach <= 0.0:
        return None
    max_surface9 = (None if max_surface_distance is None else
                    float(max_surface_distance))
    if (max_surface9 is not None
            and (not math.isfinite(max_surface9) or max_surface9 <= 0.0
                 or max_surface9 > reach)):
        raise ValueError("max_surface_distance must lie in (0, reach]")
    min_standoff9 = (
        None if min_standoff_surface_distance is None else
        float(min_standoff_surface_distance))
    if (min_standoff9 is not None
            and (not math.isfinite(min_standoff9) or min_standoff9 <= 0.0)):
        raise ValueError(
            "min_standoff_surface_distance must be finite and positive")
    if (min_standoff9 is not None
            and (max_surface9 is not None or bool(prefer_outer_shell))):
        raise ValueError(
            "standoff mode cannot be combined with a mine-distance bound")

    def surface_distance9(point9):
        delta9 = tuple(
            max(float(target[index9]) - float(point9[index9]), 0.0,
                float(point9[index9]) - (float(target[index9]) + 1.0))
            for index9 in range(3))
        return math.sqrt(sum(value9 * value9 for value9 in delta9))

    def state_known_clear(state):
        return mine_stance_known_clear(occ, state)

    sx, sy, sz = (int(math.floor(float(v))) for v in start_feet)
    start_options = [(sx, fy, sz) for fy in (sy, sy - 1, sy + 1)
                     if state_known_clear((sx, fy, sz))]
    if not start_options:
        return None
    start = min(start_options, key=lambda q: (abs(q[1] - float(start_feet[1])), q[1]))

    radius = int(math.ceil(reach)) + 1
    goals = {}
    for (x, support_y, z), _raw in occ.grid.items():
        if abs(x - target[0]) > radius or abs(z - target[2]) > radius:
            continue
        state = (int(x), int(support_y) + 1, int(z))
        if state in excluded_feet or state in goals or not state_known_clear(state):
            continue
        eye = (state[0] + 0.5, state[1] + EYE, state[2] + 0.5)
        distance9 = float(surface_distance9(eye))
        if min_standoff9 is not None:
            if min_standoff9 <= distance9 <= min_standoff9 + 1.5:
                goals[state] = (None, None, math.inf, distance9)
        else:
            strict = visible_face_within_reach(eye, target, occ, reach=reach)
            if strict is not None:
                goals[state] = strict + (distance9,)
    if not goals:
        return None

    def search9(goal_states9):
        goal_states9 = tuple(goal_states9)
        if not goal_states9:
            return None

        def heuristic(state):
            x, y, z = state
            return min(math.hypot(x - gx, z - gz) + 0.3 * abs(y - gy)
                       for gx, gy, gz in goal_states9)

        openq = [(heuristic(start), 0.0, start, None)]
        came = {}
        best_cost = {start: 0.0}
        seen = 0
        while openq and seen < int(max_nodes):
            _f, cost, cur, parent = heapq.heappop(openq)
            if cur in came:
                continue
            came[cur] = parent
            seen += 1
            if cur in goal_states9:
                path = [cur]
                while came[path[-1]] is not None:
                    path.append(came[path[-1]])
                path.reverse()
                point, normal, entry, surface_distance = goals[cur]
                result9 = {
                    "feet": cur,
                    "path": path,
                    "aim_point": point,
                    "face_normal": normal,
                    "entry_distance": float(entry),
                    "surface_distance": float(surface_distance),
                    "requested_max_surface_distance": max_surface9,
                    "surface_distance_fallback": bool(
                        max_surface9 is not None
                        and float(surface_distance) > max_surface9 + 1e-9),
                }
                if min_standoff9 is not None:
                    result9.update(
                        standoff_only=True,
                        requested_min_standoff_surface_distance=min_standoff9,
                    )
                if preferred9 is not None:
                    result9.update(
                        preferred_feet=preferred9,
                        preferred_feet_used=bool(cur == preferred9))
                return result9

            cx, cy, cz = cur
            for dx, dz in ((1, 0), (-1, 0), (0, 1), (0, -1),
                           (1, 1), (1, -1), (-1, 1), (-1, -1)):
                nx, nz = cx + dx, cz + dz
                ny = stand_y(nx, nz, cy)
                if ny is None:
                    continue
                nxt = (nx, ny, nz)
                if dx and dz and not mine_diagonal_swept_envelope_clear(
                        occ, cur, nxt):
                    continue
                new_cost = cost + math.hypot(dx, dz) + (0.3 if ny != cy else 0.0)
                if new_cost < best_cost.get(nxt, math.inf):
                    best_cost[nxt] = new_cost
                    heapq.heappush(
                        openq,
                        (new_cost + heuristic(nxt), new_cost, nxt, cur))
        return None

    def stand_y(x, z, from_y):
        for feet_y in (from_y, from_y + 1, from_y - 1):
            state = (x, feet_y, z)
            if state_known_clear(state):
                return feet_y
        return None

    distance_goals9 = tuple(
        state9 for state9, proof9 in goals.items()
        if max_surface9 is None or float(proof9[3]) <= max_surface9 + 1e-9)
    if min_standoff9 is not None:
        return search9(distance_goals9)
    if bool(prefer_outer_shell) and max_surface9 is not None:
        bands9 = {}
        for state9 in distance_goals9:
            distance9 = float(goals[state9][3])
            bucket9 = int(math.floor(distance9 / 0.25 + 1e-9))
            bands9.setdefault(bucket9, []).append(state9)
        for bucket9 in sorted(bands9, reverse=True):
            outer_result9 = search9(tuple(bands9[bucket9]))
            if outer_result9 is None:
                continue
            outer_result9.update(
                outer_shell_preferred=True,
                outer_shell_bucket_min=round(0.25 * bucket9, 3),
                outer_shell_bucket_max=round(0.25 * (bucket9 + 1), 3),
            )
            return outer_result9
    if (preferred9 is not None and preferred9 not in excluded_feet
            and preferred9 in goals and preferred9 in distance_goals9):
        preferred_result9 = search9((preferred9,))
        if preferred_result9 is not None:
            return preferred_result9
    bounded_result9 = search9(distance_goals9)
    if bounded_result9 is not None:
        bounded_result9["outer_shell_preferred"] = False
        return bounded_result9
    fallback9 = search9(tuple(goals))
    if fallback9 is not None:
        fallback9["outer_shell_preferred"] = False
    return fallback9


def _basis(yaw_deg, pitch_deg):
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([-math.sin(y) * math.cos(p), -math.sin(p), math.cos(y) * math.cos(p)])
    right = np.array([-math.cos(y), 0.0, -math.sin(y)])
    up = np.cross(right, fwd)
    return right, up, fwd


def group_instances(cells):
    todo = set(tuple(c) for c in cells)
    comps = []
    while todo:
        seed = todo.pop()
        comp, front = {seed}, [seed]
        while front:
            x, y, z = front.pop()
            for dx, dy, dz in ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                               (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                n = (x + dx, y + dy, z + dz)
                if n in todo:
                    todo.remove(n)
                    comp.add(n)
                    front.append(n)
        comps.append(frozenset(comp))
    return comps


