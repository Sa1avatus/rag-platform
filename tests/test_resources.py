from collections import namedtuple

import pytest

from rag_platform.services import resources


def test_system_resources_are_container_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    usage = namedtuple("usage", "total used free")(1000, 400, 600)
    monkeypatch.setattr(resources.shutil, "disk_usage", lambda path: usage)
    monkeypatch.setattr(resources.os, "cpu_count", lambda: 4)
    monkeypatch.setattr(resources, "_load_average", lambda: (0.5, 0.25, 0.1))
    monkeypatch.setattr(
        resources,
        "_memory",
        lambda: {
            "status": "available",
            "total_bytes": 2000,
            "used_bytes": 500,
            "available_bytes": 1500,
            "used_percent": 25.0,
        },
    )

    result = resources.system_resources()
    assert result["scope"] == "rag-api-container"
    assert result["cpu"]["count"] == 4
    assert result["disk"]["used_percent"] == 40.0
    assert result["gpu"] == {"status": "not_detected"}


def test_memory_probe_is_safe_off_linux(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resources.Path, "exists", lambda path: False)
    assert resources._memory() == {"status": "unavailable"}
    assert resources._load_average() == (None, None, None)
