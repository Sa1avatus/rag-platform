import os
import shutil
from pathlib import Path
from typing import Any


def _memory() -> dict[str, int | float | str]:
    source = Path("/proc/meminfo")
    if not source.exists():
        return {"status": "unavailable"}
    values: dict[str, int] = {}
    for line in source.read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    total = values.get("MemTotal", 0)
    available = values.get("MemAvailable", 0)
    used = max(0, total - available)
    return {
        "status": "available",
        "total_bytes": total,
        "used_bytes": used,
        "available_bytes": available,
        "used_percent": round(used / total * 100, 2) if total else 0.0,
    }


def _load_average() -> tuple[float | None, float | None, float | None]:
    probe = getattr(os, "getloadavg", None)
    if not callable(probe):
        return None, None, None
    try:
        raw = probe()
        return float(raw[0]), float(raw[1]), float(raw[2])
    except OSError:
        return None, None, None


def system_resources() -> dict[str, Any]:
    disk = shutil.disk_usage("/")
    load_1m, load_5m, load_15m = _load_average()
    return {
        "cpu": {
            "count": os.cpu_count() or 1,
            "load_1m": load_1m,
            "load_5m": load_5m,
            "load_15m": load_15m,
        },
        "memory": _memory(),
        "disk": {
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
            "used_percent": round(disk.used / disk.total * 100, 2) if disk.total else 0.0,
        },
        "gpu": {"status": "not_detected"},
        "scope": "rag-api-container",
    }
