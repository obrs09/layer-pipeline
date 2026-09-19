from __future__ import annotations

from layerforge.taxonomy import Taxonomy, _norm_tag


def build_inventory(
    tags: dict[str, float],
    taxonomy: Taxonomy,
    default_threshold: float = 0.35,
) -> list[str]:
    """Roles to cut: required, or tag-gated. Bald skips hair. No pose prior."""
    scores = {_norm_tag(name): float(score) for name, score in tags.items()}
    wanted: list[str] = []
    for name, spec in taxonomy.roles.items():
        if name == "bg":
            continue
        thresh = spec.tag_threshold if spec.tag_threshold > 0 else default_threshold
        if spec.skip_if_tags and any(scores.get(tag, 0.0) >= thresh for tag in spec.skip_if_tags):
            continue
        if spec.required:
            wanted.append(name)
            continue
        gate = spec.required_if_tags or spec.tag_names
        if gate and any(scores.get(tag, 0.0) >= thresh for tag in gate):
            wanted.append(name)
    return wanted
