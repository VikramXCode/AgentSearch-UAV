from typing import Protocol, List, Optional
from PIL import Image

class SemanticEmbeddingAdapter(Protocol):
    """
    Interface for semantic verification encoding and similarity.
    Implementations should avoid tying the verifier to a specific backend.
    """
    def score_image_against_text(self, image: Image.Image, text: str) -> float:
        ...

    def score_image_against_image(self, candidate_image: Image.Image, reference_image: Image.Image) -> float:
        ...


class CLIPEngineAdapter:
    """
    Adapter that wraps the legacy CLIPEngine for V2 verification.
    """
    def __init__(self, clip_engine=None):
        """
        Takes an optional clip_engine instance. If not provided,
        loads the shared instance (lazy loading).
        """
        self._clip_engine = clip_engine
    
    @property
    def engine(self):
        if self._clip_engine is None:
            # Lazy load CLIP engine to avoid loading during unit tests
            from models.clip_engine import CLIPEngine
            self._clip_engine = CLIPEngine.shared()
        return self._clip_engine

    def score_image_against_text(self, image: Image.Image, text: str) -> float:
        scores = self.engine.score_image_against_texts(image, [text])
        return scores.get(text, 0.0)

    def score_image_against_image(self, candidate_image: Image.Image, reference_image: Image.Image) -> float:
        scores = self.engine.score_images_against_image([candidate_image], reference_image)
        return scores[0] if scores else 0.0


class MockSemanticAdapter:
    """
    Mock adapter for unit tests to avoid loading models.
    """
    def __init__(self, text_score: float = 0.85, image_score: float = 0.88):
        self.text_score = text_score
        self.image_score = image_score

    def score_image_against_text(self, image: Image.Image, text: str) -> float:
        return self.text_score

    def score_image_against_image(self, candidate_image: Image.Image, reference_image: Image.Image) -> float:
        return self.image_score
