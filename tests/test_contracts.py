from __future__ import annotations

import json

from layerforge.contracts import SCHEMA, BackendInfo, Canvas, LayerRecord, Manifest, validate_manifest


def test_schema_is_v1():
    assert SCHEMA == "layerforge.manifest.v1"


def test_manifest_roundtrip():
    manifest = Manifest(
        schema=SCHEMA,
        job_id="job",
        source="source.png",
        canvas=Canvas(w=64, h=48),
        backend=BackendInfo(segment="noop_from_parts", inpaint="identity", compose="reproject_v1"),
        layers=[
            LayerRecord(
                id="20_body",
                role="body",
                order=20,
                file="layers/20_body.png",
                mask_visible="masks/20_body.png",
                mask_occluded=None,
                bbox=[1, 2, 3, 4],
                source="imagine_part",
                complete=True,
                notes="",
            )
        ],
    )
    data = manifest.to_dict()
    restored = validate_manifest(json.loads(json.dumps(data)))
    assert restored.layers[0].id == "20_body"
    assert restored.backend.segment == "noop_from_parts"
    assert restored.missing == []
    assert restored.layers[0].needs_click is False


def test_rejects_wrong_schema():
    data = {
        "schema": "layerforge.manifest.v2",
        "job_id": "x",
        "source": "source.png",
        "canvas": {"w": 1, "h": 1},
        "backend": {"segment": "a", "inpaint": "b", "compose": "c"},
        "layers": [],
    }
    try:
        validate_manifest(data)
    except ValueError as exc:
        assert "unsupported" in str(exc)
    else:
        raise AssertionError("expected ValueError")
