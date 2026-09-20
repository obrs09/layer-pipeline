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
    # box_cover against box ∩ character instead of the whole box (props that hang past the silhouette).
    cover_in_character: bool = False
    # Hair boxes include face/clothes; cover is judged on the leftover area.
    cover_minus_exclude: bool = False
    # ((gating tags), (extra detector queries)) pairs; queries only run when a tag fired.
    tag_queries: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = ()

    def gate_tags(self) -> tuple[str, ...]:
        """Tags that put this role in the inventory."""
        base = self.required_if_tags or self.tag_names
        extra = tuple(tag for tags, _queries in self.tag_queries for tag in tags)
        return tuple(dict.fromkeys(base + extra))

    def fired_queries(self, tags: dict[str, float], threshold: float) -> list[str]:
        """Detector queries whose gating tag scored at or above threshold."""
        scores = {_norm_tag(name): float(score) for name, score in tags.items()}
        out: list[str] = []
        for gate, queries in self.tag_queries:
            if any(scores.get(tag, 0.0) >= threshold for tag in gate):
                out.extend(q for q in queries if q not in out)
        return out


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


def _parse_tag_queries(raw) -> tuple[tuple[tuple[str, ...], tuple[str, ...]], ...]:
    out: list[tuple[tuple[str, ...], tuple[str, ...]]] = []
    for item in raw or ():
        tags = tuple(_norm_tag(t) for t in (item.get("tags") or ()))
        queries = tuple(str(q) for q in (item.get("queries") or ()))
        if tags and queries:
            out.append((tags, queries))
    return tuple(out)


def _check_occluders_sit_above(roles: dict[str, RoleSpec]) -> None:
    """A layer can only be completed under layers that composite above it."""
    for role in roles.values():
        for other in role.occluded_by:
            occluder = roles.get(other)
            if occluder is None:
                raise ValueError(f"taxonomy: {role.name}.occluded_by lists unknown role {other!r}")
            if occluder.order <= role.order:
                raise ValueError(
                    f"taxonomy: {role.name} (order {role.order}) cannot be occluded_by "
                    f"{other} (order {occluder.order}); occluders must have a higher order"
                )


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
            cover_in_character=bool(usable.get("box_cover_in_character", False)),
            cover_minus_exclude=bool(usable.get("box_cover_minus_exclude", False)),
            tag_queries=_parse_tag_queries(spec.get("tag_queries")),
        )
        roles[name] = role
        alias_to_role[_norm(name)] = name
        for alias in role.aliases:
            alias_to_role.setdefault(_norm(alias), name)
    _check_occluders_sit_above(roles)
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
