from __future__ import annotations

from pathlib import Path

import pytest

from layerforge.taxonomy import load_taxonomy


def test_occluders_sit_above_the_layer_they_cover():
    taxonomy = load_taxonomy()
    for role in taxonomy.roles.values():
        for other in role.occluded_by:
            assert other in taxonomy.roles, f"{role.name}.occluded_by has unknown {other}"
            assert taxonomy.roles[other].order > role.order, (
                f"{role.name} (order {role.order}) is occluded_by {other} "
                f"(order {taxonomy.roles[other].order}) which composites below it"
            )


def test_arms_do_not_occlude_clothes():
    taxonomy = load_taxonomy()
    clothes = taxonomy.spec("clothes")
    assert "arm_l" not in clothes.occluded_by
    assert "arm_r" not in clothes.occluded_by
    assert taxonomy.spec("arm_r").order < clothes.order


def test_load_taxonomy_rejects_lower_occluder(tmp_path: Path):
    bad = tmp_path / "taxonomy.yaml"
    bad.write_text(
        "\n".join(
            [
                "roles:",
                "  under:",
                "    order: 10",
                "    complete: true",
                "    expand_px: 4",
                "    occluded_by: []",
                "  over:",
                "    order: 20",
                "    complete: true",
                "    expand_px: 4",
                "    occluded_by: [under]",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="occluded_by"):
        load_taxonomy(bad)


def test_load_taxonomy_rejects_unknown_occluder(tmp_path: Path):
    bad = tmp_path / "taxonomy.yaml"
    bad.write_text(
        "\n".join(
            [
                "roles:",
                "  under:",
                "    order: 10",
                "    complete: true",
                "    expand_px: 4",
                "    occluded_by: [ghost]",
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown role"):
        load_taxonomy(bad)
