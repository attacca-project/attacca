from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple


Position = Tuple[float, float, float]


_KIND_KEYS: Tuple[str, ...] = ("name", "type", "entity", "kind")


def _finite_float(value: Any, field_name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mob candidate has invalid {field_name}={value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"mob candidate has non-finite {field_name}={value!r}")
    return result


def canonical_mob_kind(value: Any) -> str:

    text = str(value or "").strip().lower().split(":")[-1]
    text = text.replace(" ", "").replace("-", "_")
    if text.endswith("entity"):
        text = text[:-6]
    if text.startswith("entity_"):
        text = text[7:]
    return text


def _candidate_kind(raw: Mapping[str, Any]) -> str:
    for key in _KIND_KEYS:
        if raw.get(key) not in (None, ""):
            return canonical_mob_kind(raw[key])
    return "mob"


@dataclass(frozen=True)
class MobCandidate:

    observation_index: int
    kind: str
    position: Position


def normalize_mob_candidates(
    raw_mobs: Optional[Iterable[Mapping[str, Any]]],
    player_position: Sequence[float],
    *,
    coordinates: str = "relative",
) -> Tuple[MobCandidate, ...]:

    if coordinates not in {"relative", "absolute"}:
        raise ValueError("coordinates must be 'relative' or 'absolute'")
    if len(player_position) != 3:
        raise ValueError("player_position must contain exactly x/y/z")
    player = tuple(_finite_float(value, f"player_position[{idx}]")
                   for idx, value in enumerate(player_position))
    normalized = []
    for index, raw in enumerate(raw_mobs or ()):
        if not isinstance(raw, Mapping):
            raise ValueError(f"mob candidate {index} is not a mapping")
        kind = _candidate_kind(raw)
        relative = tuple(_finite_float(raw.get(axis), axis) for axis in ("x", "y", "z"))
        position = (tuple(player[i] + relative[i] for i in range(3))
                    if coordinates == "relative" else relative)
        normalized.append(MobCandidate(
            observation_index=index,
            kind=kind,
            position=position,
        ))
    return tuple(normalized)


