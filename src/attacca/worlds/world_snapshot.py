"""Loading and validation of immutable saved-world snapshot bundles."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


TREE_HASH_SCHEMA = b"xbench-world-snapshot-tree/v1"
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_TASK_MARKER = "_xbench_immutable_world_snapshot"
SNAPSHOT_BUNDLE_CONTRACT = "xbench-staged-world-snapshot/v3"
_VALUE_TAG = "__xbench_snapshot_value__"
_SAVE_REQUEST = ".xbench_snapshot_save_request"
_SAVE_COMPLETE = ".xbench_snapshot_save_complete"
_ARCHIVE_WORLD_NAME = "xbench_snapshot_world"
_ARCHIVE_ROOT = f"./saves/{_ARCHIVE_WORLD_NAME}"


@dataclass(frozen=True)
class WorldSnapshot:

    directory: Path
    sha256: str


@dataclass(frozen=True)
class WorldSnapshotBundle:

    directory: Path
    world: WorldSnapshot
    archive: Path
    payload: Any
    metadata: dict


def _file_sha256(path: Path) -> str:
    digest9 = hashlib.sha256()
    with path.open("rb") as stream9:
        while True:
            chunk9 = stream9.read(1024 * 1024)
            if not chunk9:
                break
            digest9.update(chunk9)
    return digest9.hexdigest()


def _write_deterministic_world_archive(world9: Path, archive9: Path) -> str:
    directory_info9 = zipfile.ZipInfo(_ARCHIVE_ROOT + "/")
    directory_info9.date_time = (1980, 1, 1, 0, 0, 0)
    directory_info9.external_attr = (0o40755 << 16) | 0x10
    with zipfile.ZipFile(
            archive9, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=6) as zip9:
        zip9.writestr(directory_info9, b"")
        for source9 in sorted(
                (item9 for item9 in world9.rglob("*") if item9.is_file()),
                key=lambda item9: item9.relative_to(world9).as_posix()):
            relative9 = source9.relative_to(world9).as_posix()
            info9 = zipfile.ZipInfo(f"{_ARCHIVE_ROOT}/{relative9}")
            info9.date_time = (1980, 1, 1, 0, 0, 0)
            info9.compress_type = zipfile.ZIP_DEFLATED
            info9.external_attr = 0o100644 << 16
            zip9.writestr(info9, source9.read_bytes(), compresslevel=6)
    return _file_sha256(archive9)


def validate_world_snapshot_archive(
        archive: os.PathLike | str, snapshot: WorldSnapshot,
        expected_sha256: Optional[str] = None) -> Path:
    raw9 = Path(archive)
    if not raw9.is_absolute():
        raise ValueError("world snapshot archive must be an absolute path")
    if raw9.is_symlink() or not raw9.is_file():
        raise ValueError(f"world snapshot archive is not a regular file: {raw9}")
    observed9 = _file_sha256(raw9)
    if expected_sha256 is not None and observed9 != str(expected_sha256).lower():
        raise ValueError(
            "world snapshot archive SHA256 mismatch: "
            f"expected={expected_sha256} observed={observed9}")
    expected_files9 = {
        f"{_ARCHIVE_ROOT}/{item9.relative_to(snapshot.directory).as_posix()}":
        item9
        for item9 in snapshot.directory.rglob("*") if item9.is_file()
    }
    with zipfile.ZipFile(raw9, "r") as zip9:
        names9 = zip9.namelist()
        if not names9 or names9[0] != _ARCHIVE_ROOT + "/":
            raise ValueError("world snapshot archive has no canonical first entry")
        file_names9 = [name9 for name9 in names9 if not name9.endswith("/")]
        if set(file_names9) != set(expected_files9) or len(file_names9) != len(expected_files9):
            raise ValueError("world snapshot archive file roster differs from world tree")
        for name9 in file_names9:
            if zip9.read(name9) != expected_files9[name9].read_bytes():
                raise ValueError(
                    f"world snapshot archive content differs from world tree: {name9}")
    return raw9.resolve(strict=True)


def encode_snapshot_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, str, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("snapshot state contains a non-finite float")
        return value
    module9 = type(value).__module__.split(".", 1)[0]
    if module9 == "numpy":
        if hasattr(value, "tolist"):
            converted9 = value.tolist()
            return {
                _VALUE_TAG: "numpy",
                "dtype": str(getattr(value, "dtype", "")),
                "value": encode_snapshot_value(converted9),
            }
        if hasattr(value, "item"):
            return encode_snapshot_value(value.item())
    if isinstance(value, tuple):
        return {_VALUE_TAG: "tuple", "items": [
            encode_snapshot_value(item9) for item9 in value]}
    if isinstance(value, list):
        return [encode_snapshot_value(item9) for item9 in value]
    if isinstance(value, (set, frozenset)):
        items9 = [encode_snapshot_value(item9) for item9 in value]
        items9.sort(key=lambda item9: json.dumps(
            item9, sort_keys=True, separators=(",", ":")))
        return {_VALUE_TAG: "set", "items": items9}
    if isinstance(value, dict):
        items9 = [
            [encode_snapshot_value(key9), encode_snapshot_value(item9)]
            for key9, item9 in value.items()
        ]
        return {_VALUE_TAG: "dict", "items": items9}
    raise TypeError(
        "snapshot state is not JSON-reversible: "
        f"{type(value).__module__}.{type(value).__qualname__}")


def decode_snapshot_value(value: Any) -> Any:
    if isinstance(value, list):
        return [decode_snapshot_value(item9) for item9 in value]
    if not isinstance(value, dict) or _VALUE_TAG not in value:
        return value
    kind9 = value.get(_VALUE_TAG)
    if kind9 == "tuple":
        return tuple(decode_snapshot_value(item9) for item9 in value["items"])
    if kind9 == "set":
        return set(decode_snapshot_value(item9) for item9 in value["items"])
    if kind9 == "dict":
        return {
            decode_snapshot_value(pair9[0]): decode_snapshot_value(pair9[1])
            for pair9 in value["items"]
        }
    if kind9 == "numpy":
        return decode_snapshot_value(value["value"])
    raise ValueError(f"unknown snapshot value encoding {kind9!r}")


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False).encode("utf-8")


def _active_saved_world(world) -> Path:
    try:
        instances9 = list(world.sim.env.instances)
    except AttributeError as exc:
        raise RuntimeError(
            "snapshot producer cannot locate MineStudio instances") from exc
    if len(instances9) != 1:
        raise RuntimeError(
            f"snapshot producer requires exactly one instance, got {len(instances9)}")
    saves9 = Path(instances9[0].working_dir) / "saves"
    candidates9 = sorted({
        level9.parent.resolve()
        for level9 in saves9.glob("*/level.dat") if level9.is_file()
    })
    if len(candidates9) != 1:
        raise RuntimeError(
            "snapshot producer requires exactly one active saved world under "
            f"{saves9}, got {candidates9}")
    return candidates9[0]


def _flush_integrated_server_save(world, *, timeout_s: float = 30.0) -> str:
    instance9 = world.sim.env.instances[0]
    working9 = Path(instance9.working_dir)
    request9 = working9 / _SAVE_REQUEST
    complete9 = working9 / _SAVE_COMPLETE
    request9.unlink(missing_ok=True)
    complete9.unlink(missing_ok=True)
    nonce9 = uuid.uuid4().hex
    temporary9 = working9 / f"{_SAVE_REQUEST}.tmp.{os.getpid()}"
    temporary9.write_text(nonce9 + "\n", encoding="utf-8")
    os.replace(temporary9, request9)
    deadline9 = time.monotonic() + float(timeout_s)
    try:
        while time.monotonic() < deadline9:
            if complete9.is_file():
                parts9 = complete9.read_text(
                    encoding="utf-8", errors="replace").splitlines()
                if len(parts9) >= 2 and parts9[1].strip() == nonce9:
                    if parts9[0].strip() != "OK":
                        detail9 = parts9[2].strip() if len(parts9) >= 3 else ""
                        raise RuntimeError(
                            "integrated-server snapshot flush failed: " + detail9)
                    return parts9[2].strip() if len(parts9) >= 3 else ""
            world.step_noop()
            time.sleep(0.01)
    finally:
        request9.unlink(missing_ok=True)
        complete9.unlink(missing_ok=True)
        temporary9.unlink(missing_ok=True)
    raise TimeoutError(
        "timed out waiting for integrated-server snapshot flush acknowledgement")


def publish_world_snapshot_bundle(
        world, destination: os.PathLike | str, *, payload: Any) -> WorldSnapshotBundle:
    destination9 = Path(destination)
    if not destination9.is_absolute():
        raise ValueError("world snapshot bundle destination must be absolute")
    if destination9.exists():
        raise FileExistsError(
            f"refusing to replace world snapshot bundle: {destination9}")
    destination9.parent.mkdir(parents=True, exist_ok=True)

    source9 = _active_saved_world(world)
    before_level_mtime9 = (source9 / "level.dat").stat().st_mtime_ns
    flush_started9 = time.monotonic()
    save_detail9 = _flush_integrated_server_save(world)
    flush_wall9 = time.monotonic() - flush_started9
    if not (source9 / "level.dat").is_file():
        raise RuntimeError(
            "integrated-server snapshot flush removed the save before copy")
    after_level_mtime9 = (source9 / "level.dat").stat().st_mtime_ns
    print(
        "[snapshot-flush] "
        f"detail={save_detail9} wall_s={flush_wall9:.3f} "
        f"level_dat_advanced={int(after_level_mtime9 > before_level_mtime9)}",
        flush=True)
    if after_level_mtime9 <= before_level_mtime9:
        raise RuntimeError(
            "integrated-server snapshot flush did not advance level.dat; "
            "refusing an unflushed snapshot")
    encoded9 = encode_snapshot_value(payload)
    payload_bytes9 = _canonical_json_bytes(encoded9)
    payload_sha9 = hashlib.sha256(payload_bytes9).hexdigest()

    temp9 = Path(tempfile.mkdtemp(
        prefix=f".{destination9.name}.tmp.", dir=destination9.parent))
    try:
        copied9 = temp9 / "world"
        copy_started9 = time.monotonic()
        shutil.copytree(source9, copied9)
        copy_wall9 = time.monotonic() - copy_started9
        snapshot9 = validate_world_snapshot(copied9)
        archive9 = temp9 / "world.zip"
        archive_started9 = time.monotonic()
        archive_sha9 = _write_deterministic_world_archive(copied9, archive9)
        archive_wall9 = time.monotonic() - archive_started9
        validate_world_snapshot_archive(archive9, snapshot9, archive_sha9)
        metadata9 = {
            "contract": SNAPSHOT_BUNDLE_CONTRACT,
            "created_unix_s": time.time(),
            "world_dir": "world",
            "world_sha256": snapshot9.sha256,
            "world_archive": "world.zip",
            "world_archive_sha256": archive_sha9,
            "world_archive_contract": "minestudio_replay_sender_dot_saves_zip/v1",
            "world_size_bytes": sum(
                entry9.stat().st_size for entry9 in copied9.rglob("*")
                if entry9.is_file()),
            "payload": encoded9,
            "payload_sha256": payload_sha9,
            "producer_timing": {
                "integrated_server_flush_wall_s": round(flush_wall9, 6),
                "copy_wall_s": round(copy_wall9, 6),
                "archive_wall_s": round(archive_wall9, 6),
            },
            "save_method": "renderer_nonce_integrated_server_flush/v1",
            "save_detail": save_detail9,
        }
        metadata_path9 = temp9 / "snapshot.json"
        with metadata_path9.open("wb") as stream9:
            stream9.write(_canonical_json_bytes(metadata9) + b"\n")
            stream9.flush()
            os.fsync(stream9.fileno())
        os.replace(temp9, destination9)
    except BaseException:
        shutil.rmtree(temp9, ignore_errors=True)
        raise
    return load_world_snapshot_bundle(destination9)


def load_world_snapshot_bundle(
        directory: os.PathLike | str) -> WorldSnapshotBundle:
    root9 = Path(directory)
    if not root9.is_absolute():
        raise ValueError("world snapshot bundle directory must be absolute")
    metadata_path9 = root9 / "snapshot.json"
    if not metadata_path9.is_file():
        raise FileNotFoundError(
            f"world snapshot bundle lacks snapshot.json: {root9}")
    metadata9 = json.loads(metadata_path9.read_text(encoding="utf-8"))
    if metadata9.get("contract") != SNAPSHOT_BUNDLE_CONTRACT:
        raise ValueError(
            f"unsupported world snapshot bundle contract: {metadata9.get('contract')!r}")
    encoded9 = metadata9.get("payload")
    observed_payload_sha9 = hashlib.sha256(
        _canonical_json_bytes(encoded9)).hexdigest()
    if observed_payload_sha9 != metadata9.get("payload_sha256"):
        raise ValueError(
            "world snapshot sidecar SHA256 mismatch: "
            f"expected={metadata9.get('payload_sha256')} "
            f"observed={observed_payload_sha9}")
    relative_world9 = metadata9.get("world_dir")
    if relative_world9 != "world":
        raise ValueError(
            f"world snapshot bundle has invalid world_dir {relative_world9!r}")
    snapshot9 = validate_world_snapshot(
        root9 / relative_world9, metadata9.get("world_sha256"))
    relative_archive9 = metadata9.get("world_archive")
    if relative_archive9 != "world.zip":
        raise ValueError(
            f"world snapshot bundle has invalid world_archive {relative_archive9!r}")
    archive9 = validate_world_snapshot_archive(
        root9 / relative_archive9, snapshot9,
        metadata9.get("world_archive_sha256"))
    return WorldSnapshotBundle(
        directory=root9.resolve(), world=snapshot9,
        archive=archive9,
        payload=decode_snapshot_value(encoded9), metadata=metadata9)


def _hash_field(digest, tag: bytes, payload: bytes) -> None:
    digest.update(tag)
    digest.update(len(payload).to_bytes(8, "big"))
    digest.update(payload)


def _saved_world_root(directory: os.PathLike | str) -> Path:
    if not isinstance(directory, (str, os.PathLike)):
        raise TypeError("world_snapshot_dir must be a path string")
    raw = Path(directory)
    if not raw.is_absolute():
        raise ValueError("world_snapshot_dir must be an absolute path")
    if raw.is_symlink():
        raise ValueError("world_snapshot_dir must not be a symbolic link")
    if not raw.is_dir():
        raise FileNotFoundError(
            f"world_snapshot_dir is not a saved-world directory: {raw}")
    level_dat = raw / "level.dat"
    if level_dat.is_symlink() or not level_dat.is_file():
        raise ValueError(
            f"world_snapshot_dir must contain a regular level.dat: {raw}")
    if level_dat.stat().st_size <= 0:
        raise ValueError(f"saved-world level.dat is empty: {level_dat}")
    return raw.resolve(strict=True)


def canonical_tree_sha256(directory: os.PathLike | str) -> str:
    root = _saved_world_root(directory)
    entries = sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix())
    digest = hashlib.sha256()
    _hash_field(digest, b"S", TREE_HASH_SCHEMA)
    for entry in entries:
        relative = entry.relative_to(root).as_posix().encode("utf-8")
        mode = entry.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ValueError(f"world snapshot contains a symbolic link: {entry}")
        if stat.S_ISDIR(mode):
            _hash_field(digest, b"D", relative)
            continue
        if not stat.S_ISREG(mode):
            raise ValueError(f"world snapshot contains a special file: {entry}")
        _hash_field(digest, b"F", relative)
        before = entry.stat()
        _hash_field(digest, b"N", int(before.st_size).to_bytes(8, "big"))
        with entry.open("rb") as stream:
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
            after_fd = os.fstat(stream.fileno())
        after = entry.stat()
        before_identity = (before.st_dev, before.st_ino, before.st_size,
                           before.st_mtime_ns)
        after_identity = (after.st_dev, after.st_ino, after.st_size,
                          after.st_mtime_ns)
        fd_identity = (after_fd.st_dev, after_fd.st_ino, after_fd.st_size,
                       after_fd.st_mtime_ns)
        if before_identity != after_identity or before_identity != fd_identity:
            raise RuntimeError(
                f"world snapshot changed while it was being hashed: {entry}")
    return digest.hexdigest()


def validate_world_snapshot(
        directory: os.PathLike | str,
        expected_sha256: Optional[str] = None) -> WorldSnapshot:
    root = _saved_world_root(directory)
    if expected_sha256 is not None:
        if (not isinstance(expected_sha256, str)
                or _SHA256_RE.fullmatch(expected_sha256) is None):
            raise ValueError("world_snapshot_sha256 must be 64 hexadecimal characters")
        expected_sha256 = expected_sha256.lower()
    observed = canonical_tree_sha256(root)
    if expected_sha256 is not None and observed != expected_sha256:
        raise ValueError(
            "world snapshot SHA256 mismatch: "
            f"expected={expected_sha256} observed={observed}")
    return WorldSnapshot(directory=root, sha256=observed)


def install_snapshot_world_generator(
        task,
        snapshot: WorldSnapshot,
        *,
        archive: os.PathLike | str,
        archive_sha256: Optional[str] = None) -> WorldSnapshot:
    if not isinstance(snapshot, WorldSnapshot):
        raise TypeError("snapshot must be a validated WorldSnapshot")
    if getattr(task, _TASK_MARKER, None) is not None:
        raise RuntimeError("a world snapshot generator is already installed on this task")
    archive9 = validate_world_snapshot_archive(
        archive, snapshot, archive_sha256)
    source9 = str(archive9)
    task.load_filename = source9
    task.agent_start = [task.create_agent_start()]
    setattr(task, _TASK_MARKER, {
        "directory": str(snapshot.directory),
        "sha256": snapshot.sha256,
        "archive": source9,
        "archive_sha256": _file_sha256(archive9),
        "fresh_process_only": True,
    })
    return snapshot
