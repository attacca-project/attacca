"""Simulator-free contracts for Mine human-play sessions: world-seed range, decision-event validation and derivation of the full-session episode pack."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
from typing import Any, Iterable, Mapping, Sequence


HUMAN_SESSION_CONTRACT = "xbench_v2_human_multicommit_session/v1"
HUMAN_SEGMENT_CONTRACT = "xbench_v2_human_commit_segment/v1"
HUMAN_FULL_EPISODE_CONTRACT = "xbench_v2_human_full_quota3_episode/v1"
HUMAN_MOTION_PROFILE = "human_v1"
HUMAN_WORLD_SEED_RANGE = (300_000, 310_000)
HUMAN_AUTO_COMMIT_VISIBLE_FRAMES = 3
HUMAN_AUTO_COMMIT_VISIBILITY_CONTRACT = (
    "human_uncommitted_exact_target_visible_streak/v1")
FREEZE_REASONS = frozenset((
    "c_press", "visible_streak", "break_auto", "quota", "manual_pause"))
DECISION_KINDS = frozenset((
    "freeze_enter", "selection_rejected", "selection_accepted", "resume",
    "break_detected", "abort", "timeout", "curation_accept",
))


class HumanSessionError(ValueError):
    pass


def advance_uncommitted_visible_streak(
        previous: int, *, recognizable_visible: bool,
        has_active_selection: bool) -> tuple[int, bool]:
    if (isinstance(previous, bool) or not isinstance(previous, int)
            or int(previous) < 0):
        raise HumanSessionError(
            "previous visible streak must be a nonnegative integer")
    if not isinstance(recognizable_visible, bool):
        raise HumanSessionError("recognizable_visible must be boolean")
    if not isinstance(has_active_selection, bool):
        raise HumanSessionError("has_active_selection must be boolean")
    if has_active_selection or not recognizable_visible:
        return 0, False
    streak = min(
        int(previous) + 1, int(HUMAN_AUTO_COMMIT_VISIBLE_FRAMES))
    return streak, bool(streak == HUMAN_AUTO_COMMIT_VISIBLE_FRAMES)


def validate_human_world_seed(seed: int) -> int:
    value = int(seed)
    lo, hi = HUMAN_WORLD_SEED_RANGE
    if not lo <= value < hi:
        raise HumanSessionError(
            f"human world seed {value} is outside reserved range [{lo},{hi})")
    return value


def nearest_face_distance(
        eye_xyz: Sequence[float], cells: Iterable[Sequence[int]]) -> float:
    try:
        eye = tuple(float(value) for value in eye_xyz)
    except (TypeError, ValueError, OverflowError) as exc:
        raise HumanSessionError("eye position is malformed") from exc
    if len(eye) != 3 or not all(math.isfinite(value) for value in eye):
        raise HumanSessionError("eye position must be a finite xyz triple")
    distances = []
    for raw_cell in cells:
        if (not isinstance(raw_cell, (list, tuple)) or len(raw_cell) != 3
                or any(isinstance(value, bool) for value in raw_cell)):
            raise HumanSessionError("target cell is malformed")
        cell = tuple(int(value) for value in raw_cell)
        if any(float(raw_cell[i]) != float(cell[i]) for i in range(3)):
            raise HumanSessionError("target cell is not integral")
        delta = tuple(max(float(cell[i]) - eye[i], 0.0,
                          eye[i] - (float(cell[i]) + 1.0))
                      for i in range(3))
        distances.append(math.sqrt(sum(value * value for value in delta)))
    if not distances:
        raise HumanSessionError("target instance has no cells")
    return round(float(min(distances)), 4)


class DecisionLedger:

    def __init__(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._next_index = 0

    def append(self, kind: str, *, source_traj_t: int, source_frame: int,
               freeze_reason: str | None = None, **fields: Any) -> dict[str, Any]:
        event_kind = str(kind)
        if event_kind not in DECISION_KINDS:
            raise HumanSessionError(f"unknown decision event {event_kind!r}")
        reason = None if freeze_reason is None else str(freeze_reason)
        if reason is not None and reason not in FREEZE_REASONS:
            raise HumanSessionError(f"unknown freeze reason {reason!r}")
        event = {
            "version": 1,
            "index": int(self._next_index),
            "kind": event_kind,
            "source_traj_t": int(source_traj_t),
            "source_frame": int(source_frame),
            "freeze_reason": reason,
            **copy.deepcopy(fields),
        }
        self._next_index += 1
        self._events.append(event)
        return copy.deepcopy(event)

    def document(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._events)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip():
            continue
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise HumanSessionError(f"{path}:{line_number} is not an object")
        rows.append(value)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


def derive_full_session_pack(
        raw_session_dir: Path | str,
        pack_root: Path | str,
        *, cell: str = "mine", setting: str,
        terminal_outcome: str = "success") -> dict[str, Any]:
    raw_dir = Path(raw_session_dir)
    root = Path(pack_root)
    if root.exists():
        raise HumanSessionError(f"refusing to reuse full-session root {root}")
    story_path = raw_dir / f"story_{cell}_{setting}.jsonl"
    traj_path = raw_dir / f"traj_{cell}_{setting}.jsonl"
    video_path = raw_dir / f"demo_{cell}_{setting}.mp4"
    story_all = _read_jsonl(story_path)
    if len(story_all) < 2:
        raise HumanSessionError("raw full-session story has no frames")
    header, story = copy.deepcopy(story_all[0]), copy.deepcopy(story_all[1:])
    traj = copy.deepcopy(_read_jsonl(traj_path))
    if [int(row.get("f", -1)) for row in story] != list(range(len(story))):
        raise HumanSessionError("raw full-session story frames are not contiguous")
    source_ts = [int(row["traj_t"]) for row in story]
    lo_t, hi_t = min(source_ts), max(source_ts)
    traj = [row for row in traj if lo_t <= int(row.get("t", -1)) <= hi_t]
    if int(header.get("quota", -1)) != 3:
        raise HumanSessionError(
            f"full HumanPlay episode must retain quota=3, got {header.get('quota')!r}")
    if str(terminal_outcome) != "success":
        raise HumanSessionError(
            f"full HumanPlay training episode requires success, got {terminal_outcome!r}")
    runtime_path = raw_dir / "runtime_timing.json"
    if not runtime_path.is_file():
        raise HumanSessionError(f"raw full session lacks {runtime_path}")
    runtime_document = json.loads(runtime_path.read_text(encoding="utf-8"))
    runtime_episodes = runtime_document.get("episodes")
    runtime_episode = (runtime_episodes.get(f"{cell}/{setting}")
                       if isinstance(runtime_episodes, Mapping) else None)
    runtime_summary = (runtime_episode.get("human_deferred_labels")
                       if isinstance(runtime_episode, Mapping) else None)
    if isinstance(runtime_summary, Mapping):
        runtime_summary = copy.deepcopy(runtime_summary)
        counts = runtime_summary.get("human_marker_event_counts")
        if isinstance(counts, Mapping):
            valid_markers = sum(int(counts.get(key, 0)) for key in (
                "human_commit", "human_switch", "human_reaffirm"))
            evidence_complete = bool(
                runtime_summary.get("human_end_reason") == "success"
                and int(runtime_summary.get("human_completed", -1)) >= 3
                and valid_markers >= 3
                and int(counts.get("human_marker_rejected", -1)) == 0
                and int(counts.get("human_implicit_commit", -1)) == 0
                and int(runtime_summary.get(
                    "human_successful_active_chosen_interaction_count", -1)) >= 3)
            runtime_summary["human_marker_evidence_complete"] = evidence_complete
            runtime_summary["human_marker_contract_complete"] = bool(
                evidence_complete
                and runtime_summary.get("scripted_selfcheck") is False)
            runtime_episode["human_deferred_labels"] = runtime_summary
            runtime_episode["human_marker_contract_complete"] = bool(
                runtime_summary["human_marker_contract_complete"])
    terminal_summary = story[-1].get("human_episode_summary")
    if isinstance(runtime_summary, Mapping):
        if isinstance(terminal_summary, Mapping):
            disagreements = [
                key for key in terminal_summary
                if key in runtime_summary
                and key not in (
                    "human_marker_evidence_complete",
                    "human_marker_contract_complete")
                and terminal_summary[key] != runtime_summary[key]
            ]
            if disagreements:
                raise HumanSessionError(
                    "raw story/runtime human summaries disagree: "
                    + ", ".join(disagreements))
        terminal_summary = copy.deepcopy(runtime_summary)
        story[-1]["human_episode_summary"] = terminal_summary
    if (not isinstance(terminal_summary, Mapping)
            or terminal_summary.get("human_end_reason") != "success"
            or int(terminal_summary.get("human_completed", -1)) != 3):
        raise HumanSessionError(
            "raw full-session terminal does not certify quota-3 success")

    for row in [*story, *traj]:
        exclusion = row.pop("human_bc_exclusion", None)
        if isinstance(exclusion, Mapping):
            row["bc_valid"] = 1
            row["source_human_interaction_context_audit"] = {
                **copy.deepcopy(exclusion),
                "bc_exclusion_not_applied": True,
                "bc_valid_preserved": True,
            }

    header.update(
        episode_source="human_play",
        oracle_motion_profile=HUMAN_MOTION_PROFILE,
        census_enabled=True,
        human_full_episode={
            "contract": HUMAN_FULL_EPISODE_CONTRACT,
            "source_session": str(raw_dir.resolve()),
            "quota": 3,
            "continuous": True,
        })
    world_seed = int(header["world_seed"])
    target_kind = str(header["target_kind"])
    episode_id = (
        f"human_full_{cell}_{setting}_{target_kind}_w{world_seed}")
    episode_dir = root / "episodes" / episode_id
    episode_dir.mkdir(parents=True)
    _write_jsonl(
        episode_dir / f"story_{cell}_{setting}.jsonl", [header, *story])
    _write_jsonl(episode_dir / f"traj_{cell}_{setting}.jsonl", traj)
    destination_video = episode_dir / f"demo_{cell}_{setting}.mp4"
    try:
        destination_video.hardlink_to(video_path)
    except OSError:
        shutil.copy2(video_path, destination_video)
    demo_source = raw_dir / "demo.json"
    if not demo_source.is_file():
        raise HumanSessionError(f"raw full session lacks {demo_source}")
    shutil.copy2(demo_source, episode_dir / "demo.json")
    (episode_dir / "runtime_timing.json").write_text(
        json.dumps(runtime_document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    (episode_dir / "meta.json").write_text(json.dumps({
        "contract": HUMAN_FULL_EPISODE_CONTRACT,
        "episode_source": "human_play",
        "oracle_motion_profile": HUMAN_MOTION_PROFILE,
        "source_session": str(raw_dir.resolve()),
        "world_seed": world_seed,
        "quota": 3,
        "frames": len(story),
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    record = {
        "index": 0,
        "episode_id": episode_id,
        "cell": cell,
        "setting": setting,
        "world_seed": world_seed,
        "action_seed": int(header["action_seed"]),
        "biome": str(header["biome"]),
        "result": 1,
        "frames": len(story),
        "kept": True,
        "hard_case": False,
        "final_phase": "complete",
        "path": str(episode_dir.relative_to(root)),
        "episode_source": "human_play",
        "oracle_motion_profile": HUMAN_MOTION_PROFILE,
        "target_kind": target_kind,
        "quota": 3,
    }
    _write_jsonl(root / "manifest.jsonl", [record])
    summary = {
        "contract": HUMAN_FULL_EPISODE_CONTRACT,
        "raw_session": str(raw_dir.resolve()),
        "episodes": 1,
        "segments": 1,
        "kept": 1,
        "records": [record],
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    return summary


def materialize_raw_session(
        raw_session_dir: Path | str, *, cell: str, setting: str) -> dict[str, Any]:
    raw_dir = Path(raw_session_dir)
    canonical_story = raw_dir / f"story_{cell}_{setting}.jsonl"
    canonical_traj = raw_dir / f"traj_{cell}_{setting}.jsonl"
    canonical_video = raw_dir / f"demo_{cell}_{setting}.mp4"
    canonical_meta = raw_dir / f"human_session_{cell}_{setting}.json"
    required = (canonical_story, canonical_traj, canonical_video, canonical_meta)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise HumanSessionError(f"raw session inputs missing: {missing}")
    story_all = _read_jsonl(canonical_story)
    if len(story_all) < 2:
        raise HumanSessionError("raw session story has no frames")
    header, story = story_all[0], story_all[1:]
    if [int(row.get("f", -1)) for row in story] != list(range(len(story))):
        raise HumanSessionError("raw session story frames are not contiguous")
    traj = _read_jsonl(canonical_traj)
    traj_by_t = {int(row["t"]): row for row in traj}
    if len(traj_by_t) != len(traj):
        raise HumanSessionError("raw session canonical traj has duplicate t")
    aligned = []
    for story_row in story:
        source_t = int(story_row["traj_t"])
        source = traj_by_t.get(source_t)
        if source is None:
            raise HumanSessionError(
                f"raw frame {story_row['f']} has no trajectory source t={source_t}")
        row = copy.deepcopy(source)
        row["f"] = int(story_row["f"])
        aligned.append(row)

    import cv2
    capture = cv2.VideoCapture(str(canonical_video))
    if not capture.isOpened():
        raise HumanSessionError(f"cannot decode human POV {canonical_video}")
    video_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    if video_frames != len(aligned) or (width, height) != (640, 360):
        raise HumanSessionError(
            "raw POV/traj mismatch: "
            f"video={video_frames}@{width}x{height} traj={len(aligned)}")

    outputs = (raw_dir / "pov.mp4", raw_dir / "traj.jsonl",
               raw_dir / "story.jsonl", raw_dir / "meta.json")
    if any(path.exists() for path in outputs):
        raise HumanSessionError(
            f"refusing to overwrite raw-session aliases under {raw_dir}")
    (raw_dir / "pov.mp4").hardlink_to(canonical_video)
    _write_jsonl(raw_dir / "traj.jsonl", aligned)
    _write_jsonl(raw_dir / "story.jsonl", [header, *story])
    meta = json.loads(canonical_meta.read_text(encoding="utf-8"))
    meta.update({
        "raw_pov": "pov.mp4",
        "raw_traj": "traj.jsonl",
        "raw_story": "story.jsonl",
        "frame_count": len(aligned),
        "traj_row_count": len(aligned),
        "pov_sha256": hashlib.sha256(
            canonical_video.read_bytes()).hexdigest(),
    })
    (raw_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "contract": HUMAN_SESSION_CONTRACT,
        "frames": len(aligned),
        "traj_rows": len(aligned),
        "pov": str((raw_dir / "pov.mp4").resolve()),
    }
