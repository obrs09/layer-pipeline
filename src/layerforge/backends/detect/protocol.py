from __future__ import annotations

from typing import Protocol

import numpy as np


class CharacterCut(Protocol):
    name: str

    def cut(self, image: np.ndarray) -> np.ndarray:
        """Full-character mask, uint8 0/255."""
        ...


class Tagger(Protocol):
    name: str

    def tag(self, image: np.ndarray) -> dict[str, float]:
        ...


class BoxDetector(Protocol):
    name: str

    def detect(
        self,
        image: np.ndarray,
        queries: list[str],
        threshold: float,
    ) -> list[tuple[str, list[float], float]]:
        """(query, xyxy, score) boxes. One detector call per inventory role, not all-class."""
        ...


class TextMasker(Protocol):
    name: str

    def predict_text(
        self,
        image: np.ndarray,
        query: str,
        character: np.ndarray,
    ) -> np.ndarray | None:
        ...


class PoseEstimator(Protocol):
    name: str

    def estimate(self, image: np.ndarray):
        """Person skeleton / hull used as hints. None if weights or person missing."""
        ...
