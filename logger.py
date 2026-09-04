"""
logger.py — Structured run logger (Section 1).

Every training run writes one JSON-lines record to logs/run_log.jsonl.
Each record is fully self-describing: config hash, seed, fold, all metrics,
checkpoint path, and wall-clock time — so every reported number is traceable.
"""

import json
import hashlib
import time
import os
from typing import Any, Dict, Optional


class RunLogger:
    """Append-only structured logger producing JSON Lines output."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    def _config_hash(self, config: Dict[str, Any]) -> str:
        """Deterministic 8-char hash of the config dict."""
        serialized = json.dumps(config, sort_keys=True, default=str)
        return hashlib.md5(serialized.encode()).hexdigest()[:8]

    def log(
        self,
        model_name: str,
        fold: int,
        config: Dict[str, Any],
        seed: int,
        metrics: Dict[str, float],
        checkpoint_path: Optional[str] = None,
        wall_time_sec: Optional[float] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model_name,
            "fold": fold,
            "seed": seed,
            "config_hash": self._config_hash(config),
            "config": config,
            "metrics": metrics,
            "checkpoint": checkpoint_path,
            "wall_time_sec": wall_time_sec,
        }
        if extra:
            record["extra"] = extra

        with open(self.log_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def read_all(self):
        """Return list of all logged records."""
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as f:
            return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# Module-level convenience instance (import and use directly)
# ---------------------------------------------------------------------------

from config import RUN_LOG_PATH
run_logger = RunLogger(RUN_LOG_PATH)
