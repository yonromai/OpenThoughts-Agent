import os
from pathlib import Path

from hpc.hf_utils import (
    DAYTONA_DATA_COPY_MARKER,
    materialize_raw_tasks_for_daytona_source_build,
    needs_daytona_source_staging,
)


def _relative_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(os.path.relpath(target, link.parent))


def _write_raw_task_snapshot(tmp_path: Path) -> Path:
    snapshot = tmp_path / "datasets--DCAgent2--terminal_bench_2" / "snapshots" / "abc123"
    blobs = tmp_path / "datasets--DCAgent2--terminal_bench_2" / "blobs"
    blobs.mkdir(parents=True)

    users_json = blobs / "users.json"
    users_json.write_text('{"users": [{"id": 1, "name": "A"}]}\n', encoding="utf-8")

    task = snapshot / "multi-source-data-merger"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text(
        "FROM scratch\nCOPY data /data\nCOPY task-deps/ /app/\n",
        encoding="utf-8",
    )
    (task / "environment" / "task-deps").mkdir()
    (task / "environment" / "task-deps" / "helper.txt").write_text("helper\n")
    (task / "instruction.md").write_text("Merge the input files.\n", encoding="utf-8")
    _relative_symlink(users_json, task / "environment" / "data" / "source_a" / "users.json")
    return snapshot


def test_materialize_raw_tasks_dereferences_hf_snapshot_symlinks(tmp_path: Path) -> None:
    snapshot = _write_raw_task_snapshot(tmp_path)
    source_data = (
        snapshot
        / "multi-source-data-merger"
        / "environment"
        / "data"
        / "source_a"
        / "users.json"
    )
    assert source_data.is_symlink()
    assert needs_daytona_source_staging(snapshot)

    result_path = materialize_raw_tasks_for_daytona_source_build(
        snapshot,
        verbose=False,
    )

    staged = Path(result_path)
    staged_data = (
        staged
        / "multi-source-data-merger"
        / "environment"
        / "data"
        / "source_a"
        / "users.json"
    )
    staged_dockerfile = staged / "multi-source-data-merger" / "environment" / "Dockerfile"

    assert staged_data.read_text(encoding="utf-8") == '{"users": [{"id": 1, "name": "A"}]}\n'
    assert not staged_data.is_symlink()
    assert not any(path.is_symlink() for path in staged.rglob("*"))
    dockerfile_text = staged_dockerfile.read_text(encoding="utf-8")
    assert DAYTONA_DATA_COPY_MARKER in dockerfile_text
    assert "COPY data /data" not in dockerfile_text
    assert "COPY data/source_a/users.json /data/source_a/users.json" in dockerfile_text
    assert "COPY task-deps/ /app/" in dockerfile_text
    assert not needs_daytona_source_staging(staged)
