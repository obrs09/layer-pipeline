"""Stable v0 pack contract. Field meanings must not change without a new schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Literal

SCHEMA = "layerforge.manifest.v1"

LayerSource = Literal["imagine_part", "sam", "manual", "silhouette"]


@dataclass
class Canvas:
    w: int
    h: int


@dataclass
class BackendInfo:
    segment: str
    inpaint: str
    compose: str


@dataclass
class LayerRecord:
    id: str
    role: str
    order: int
    file: str
    mask_visible: str
    mask_occluded: str | None
    bbox: list[int]
    source: str
    complete: bool
    notes: str = ""
    needs_click: bool = False


@dataclass
class Manifest:
    schema: str
    job_id: str
    source: str
    canvas: Canvas
    backend: BackendInfo
    layers: list[LayerRecord] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Manifest:
        if data.get("schema") != SCHEMA:
            raise ValueError(f"unsupported manifest schema: {data.get('schema')}")
        canvas = Canvas(**data["canvas"])
        backend = BackendInfo(**data["backend"])
        layers = [_layer_from_dict(layer) for layer in data.get("layers", [])]
        return cls(
            schema=data["schema"],
            job_id=data["job_id"],
            source=data["source"],
            canvas=canvas,
            backend=backend,
            layers=layers,
            missing=list(data.get("missing") or []),
        )


def _layer_from_dict(data: dict[str, Any]) -> LayerRecord:
    allowed = {item.name for item in fields(LayerRecord)}
    return LayerRecord(**{key: value for key, value in data.items() if key in allowed})


def validate_manifest(data: dict[str, Any]) -> Manifest:
    required = {"schema", "job_id", "source", "canvas", "backend", "layers"}
    missing = required - set(data)
    if missing:
        raise ValueError(f"manifest missing fields: {sorted(missing)}")
    manifest = Manifest.from_dict(data)
    if manifest.canvas.w <= 0 or manifest.canvas.h <= 0:
        raise ValueError("canvas size must be positive")
    ids = [layer.id for layer in manifest.layers]
    if len(ids) != len(set(ids)):
        raise ValueError("layer ids must be unique")
    for layer in manifest.layers:
        if len(layer.bbox) != 4:
            raise ValueError(f"{layer.id} bbox must be [x, y, w, h]")
        if layer.source not in {"imagine_part", "sam", "manual", "silhouette"}:
            raise ValueError(f"{layer.id} has unknown source {layer.source}")
    return manifest
