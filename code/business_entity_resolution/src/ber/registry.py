"""Component registry.

Every stage implementation registers itself under a unique name:

    from ber.registry import register

    @register("blocker", "p3_addr_tfidf")
    class AddrTfidf(Blocker): ...

Configs refer to components as {"name": "...", "params": {...}}. Adding a new
file under ber/<stage>/ is enough: `autodiscover()` imports every module in the
stage packages, so nobody has to edit a shared __init__.py (no merge conflicts).
"""
from __future__ import annotations

import importlib
import pkgutil
from typing import Any

KINDS = ("normalizer", "blocker", "features", "scorer", "decider")
_STAGE_PACKAGES = ("ber.normalize", "ber.blocking", "ber.features", "ber.scorers", "ber.deciders")
_REGISTRY: dict[str, dict[str, type]] = {k: {} for k in KINDS}
_DISCOVERED = False


def register(kind: str, name: str):
    """Class decorator: register `cls` as component `name` of stage `kind`."""
    if kind not in KINDS:
        raise ValueError(f"unknown component kind {kind!r}; expected one of {KINDS}")

    def deco(cls):
        existing = _REGISTRY[kind].get(name)
        if existing is not None and existing.__qualname__ != cls.__qualname__:
            raise ValueError(
                f"{kind} name {name!r} already registered by {existing.__module__}.{existing.__qualname__}. "
                "Prefix experimental components with your handle (e.g. 'p3_...')."
            )
        _REGISTRY[kind][name] = cls
        cls.registry_name = name
        return cls

    return deco


def autodiscover() -> None:
    """Import every module in the stage packages so their @register calls run."""
    global _DISCOVERED
    if _DISCOVERED:
        return
    for pkg_name in _STAGE_PACKAGES:
        pkg = importlib.import_module(pkg_name)
        for mod in pkgutil.iter_modules(pkg.__path__):
            if not mod.name.startswith("_"):
                importlib.import_module(f"{pkg_name}.{mod.name}")
    _DISCOVERED = True


def build(kind: str, spec: dict[str, Any] | str):
    """Instantiate a component from a config spec {"name": ..., "params": {...}}."""
    autodiscover()
    if isinstance(spec, str):
        spec = {"name": spec}
    name = spec["name"]
    params = spec.get("params") or {}
    try:
        cls = _REGISTRY[kind][name]
    except KeyError:
        raise KeyError(f"no {kind} named {name!r}. Available: {sorted(_REGISTRY[kind])}") from None
    obj = cls(**params)
    obj.spec = {"name": name, "params": params}
    return obj


def available() -> dict[str, list[str]]:
    autodiscover()
    return {k: sorted(v) for k, v in _REGISTRY.items()}
