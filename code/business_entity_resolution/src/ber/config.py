"""YAML config loading: `inherit:` chains, ${ENV} expansion, --set overrides."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import yaml

PKG_ROOT = Path(__file__).resolve().parents[2]        # code/business_entity_resolution
REPO_ROOT = PKG_ROOT.parents[1]                      # repo root (contains planning/)

ENV_DEFAULTS = {
    "BER_DATA_DIR": str(REPO_ROOT / "data" / "dataset"),
    "BER_ARTIFACTS": str(REPO_ROOT / "artifacts"),
    "BER_MEMBER": "anon",
}


def env(name: str) -> str:
    return os.environ.get(name) or ENV_DEFAULTS[name]


def _deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _expand(obj):
    if isinstance(obj, str):
        for k in ENV_DEFAULTS:
            os.environ.setdefault(k, ENV_DEFAULTS[k])
        return os.path.expandvars(obj)
    if isinstance(obj, dict):
        return {k: _expand(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_expand(v) for v in obj]
    return obj


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        base = load_config((path.parent / parent).resolve())
        base.pop("_meta", None)
        cfg = _deep_merge(base, cfg)
    for ov in overrides or []:
        key, _, raw = ov.partition("=")
        val = yaml.safe_load(raw)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if isinstance(node, list):
                node = node[int(p)]
            else:
                node = node.setdefault(p, {})
        if isinstance(node, list):
            node[int(parts[-1])] = val
        else:
            node[parts[-1]] = val
    cfg = _expand(cfg)
    try:
        rel = path.relative_to(PKG_ROOT).as_posix()
    except ValueError:
        rel = str(path)
    cfg["_meta"] = {"config_path": rel, "overrides": list(overrides or [])}
    return cfg


def config_hash(obj) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]
