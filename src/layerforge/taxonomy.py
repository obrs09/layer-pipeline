from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from layerforge.config import ROOT


@dataclass(frozen=True)
class RoleSpec:
    name: str
    order: int
    complete: bool
    overlay: bool
    expand_px: int
    occluded_by: tuple[str, ...]
    aliases: tuple[str, ...]


@dataclass
class Taxonomy:
    roles: dict[str, RoleSpec]
    pair_roles: dict[str, tuple[str, str]]
    pair_aliases: dict[str, str]
    alias_to_role: dict[str, str]

    def spec(self, role: str) -> RoleSpec:
        if role not in self.roles:
            return RoleSpec(
                name=role,
                order=50,
                complete=False,
                overlay=True,
                expand_px=0,
                occluded_by=(),
                aliases=(),
            )
        return self.roles[role]

    def resolve_label(self, label: str) -> str | None:
        key = _norm(label)
        if key in self.roles:
            return key
        return self.alias_to_role.get(key)

    def pair_family(self, label: str) -> str | None:
        key = _norm(label)
        if key in self.pair_aliases:
            return self.pair_aliases[key]
        if key in self.pair_roles:
            return key
        role = self.resolve_label(label)
        if role is None:
            return None
        for family, pair in self.pair_roles.items():
            if role in pair:
                return family
        return None


def _norm(text: str) -> str:
    return text.strip().lower().replace("_", "-")


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    tax_path = Path(path) if path else ROOT / "taxonomy.yaml"
    raw: dict[str, Any] = yaml.safe_load(tax_path.read_text(encoding="utf-8")) or {}
    roles: dict[str, RoleSpec] = {}
    alias_to_role: dict[str, str] = {}
    for name, spec in (raw.get("roles") or {}).items():
        role = RoleSpec(
            name=name,
            order=int(spec["order"]),
            complete=bool(spec.get("complete", False)),
            overlay=bool(spec.get("overlay", False)),
            expand_px=int(spec.get("expand_px", 0)),
            occluded_by=tuple(spec.get("occluded_by") or ()),
            aliases=tuple(spec.get("aliases") or ()),
        )
        roles[name] = role
        alias_to_role[_norm(name)] = name
        for alias in role.aliases:
            alias_to_role.setdefault(_norm(alias), name)
    pair_roles = {
        family: (pair[0], pair[1])
        for family, pair in (raw.get("pair_roles") or {}).items()
    }
    pair_aliases = {
        _norm(alias): family
        for alias, family in (raw.get("pair_aliases") or {}).items()
    }
    return Taxonomy(
        roles=roles,
        pair_roles=pair_roles,
        pair_aliases=pair_aliases,
        alias_to_role=alias_to_role,
    )
