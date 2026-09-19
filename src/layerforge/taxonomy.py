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
    queries: tuple[str, ...] = ()
    required: bool = False
    required_if_tags: tuple[str, ...] = ()
    skip_if_tags: tuple[str, ...] = ()
    tag_names: tuple[str, ...] = ()
    exclude_roles: tuple[str, ...] = ()
    cut_priority: int = 0
    tag_threshold: float = 0.35
    min_area_frac: float = 0.0
    max_area_frac: float = 1.0
    max_components: int = 8
    min_box_cover: float = 0.35
    max_out_of_box: float = 0.5
    max_overlap_frac: float = 0.45
    crumb_frac: float = 0.02


@dataclass
class Taxonomy:
    roles: dict[str, RoleSpec]
    pair_roles: dict[str, tuple[str, str]]
    pair_aliases: dict[str, str]
    alias_to_role: dict[str, str]
    pair_queries: dict[str, tuple[str, ...]]

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

    def family_of_role(self, role: str) -> str | None:
        for family, pair in self.pair_roles.items():
            if role in pair:
                return family
        return None


def _norm(text: str) -> str:
    return text.strip().lower().replace("_", "-")


def _norm_tag(text: str) -> str:
    return text.strip().lower().replace("-", "_").replace(" ", "_")


def load_taxonomy(path: str | Path | None = None) -> Taxonomy:
    tax_path = Path(path) if path else ROOT / "taxonomy.yaml"
    raw: dict[str, Any] = yaml.safe_load(tax_path.read_text(encoding="utf-8")) or {}
    roles: dict[str, RoleSpec] = {}
    alias_to_role: dict[str, str] = {}
    for name, spec in (raw.get("roles") or {}).items():
        usable = spec.get("usable") or {}
        role = RoleSpec(
            name=name,
            order=int(spec["order"]),
            complete=bool(spec.get("complete", False)),
            overlay=bool(spec.get("overlay", False)),
            expand_px=int(spec.get("expand_px", 0)),
            occluded_by=tuple(spec.get("occluded_by") or ()),
            aliases=tuple(spec.get("aliases") or ()),
            queries=tuple(spec.get("queries") or ()),
            required=bool(spec.get("required", False)),
            required_if_tags=tuple(_norm_tag(t) for t in (spec.get("required_if_tags") or ())),
            skip_if_tags=tuple(_norm_tag(t) for t in (spec.get("skip_if_tags") or ())),
            tag_names=tuple(_norm_tag(t) for t in (spec.get("tag_names") or ())),
            exclude_roles=tuple(spec.get("exclude_roles") or ()),
            cut_priority=int(spec.get("cut_priority", 0)),
            tag_threshold=float(spec.get("tag_threshold", 0.35)),
            min_area_frac=float(usable.get("min_area_frac", spec.get("min_area_frac", 0.0))),
            max_area_frac=float(usable.get("max_area_frac", spec.get("max_area_frac", 1.0))),
            max_components=int(usable.get("max_components", spec.get("max_components", 8))),
            min_box_cover=float(usable.get("min_box_cover", spec.get("min_box_cover", 0.35))),
            max_out_of_box=float(usable.get("max_out_of_box", spec.get("max_out_of_box", 0.5))),
            max_overlap_frac=float(usable.get("max_overlap_frac", spec.get("max_overlap_frac", 0.45))),
            crumb_frac=float(usable.get("crumb_frac", spec.get("crumb_frac", 0.02))),
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
    pair_queries = {
        family: tuple(queries)
        for family, queries in (raw.get("pair_queries") or {}).items()
    }
    return Taxonomy(
        roles=roles,
        pair_roles=pair_roles,
        pair_aliases=pair_aliases,
        alias_to_role=alias_to_role,
        pair_queries=pair_queries,
    )
