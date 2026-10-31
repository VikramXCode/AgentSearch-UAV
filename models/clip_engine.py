from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image

try:
    import clip
except ImportError as exc:
    clip = None
    _clip_import_error = exc
else:
    _clip_import_error = None


class CLIPEngine:
    """
    High-performance CLIP Engine with singleton instance caching,
    pre-computed normalized text embeddings, and batched image forward passes.
    Returns cosine similarities (instead of softmax probabilities) so that
    prompts do not artificially compete when none are a good match.
    """

    def __init__(self, model_name: str = "ViT-B/32"):
        if clip is None:
            raise RuntimeError(
                "CLIP is not installed. Install OpenAI CLIP before "
                "running verification."
            ) from _clip_import_error

        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"Loading CLIP {model_name} on {self.device}...")
        self.model, self.preprocess = clip.load(
            model_name,
            device=self.device,
            jit=False,
        )
        self.model.eval()
        self._text_features_cache: Dict[Tuple[str, ...], torch.Tensor] = {}
        print("CLIP loaded successfully.")

    @classmethod
    @lru_cache(maxsize=1)
    def shared(cls) -> "CLIPEngine":
        return cls()

    def get_text_features(self, texts: Sequence[str]) -> torch.Tensor:
        """
        Get cached, normalized text embeddings for a list of text prompts.
        Only encodes through transformer text-encoder once per unique prompt tuple.
        """
        texts_key = tuple(texts)
        if texts_key in self._text_features_cache:
            return self._text_features_cache[texts_key]

        if not texts:
            empty_tensor = torch.empty((0, self.model.visual.output_dim), device=self.device)
            return empty_tensor

        text_tokens = clip.tokenize(list(texts)).to(self.device)
        with torch.inference_mode():
            text_features = self.model.encode_text(text_tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        self._text_features_cache[texts_key] = text_features
        return text_features

    def get_image_features(self, images: list[Image.Image], batch_size: int = 32) -> torch.Tensor:
        """
        Get normalized image embeddings for a list of PIL images.
        """
        if not images:
            return torch.empty((0, self.model.visual.output_dim), device=self.device)

        all_features = []
        for i in range(0, len(images), batch_size):
            chunk_images = images[i : i + batch_size]
            image_tensors = torch.stack([
                self.preprocess(img.convert("RGB")) for img in chunk_images
            ]).to(self.device)

            with torch.inference_mode():
                image_features = self.model.encode_image(image_tensors)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                all_features.append(image_features)
        
        return torch.cat(all_features, dim=0)

    def score_images_against_image(
        self,
        candidate_images: list[Image.Image],
        reference_image: Image.Image,
        batch_size: int = 32,
    ) -> list[float]:
        """
        Batch CLIP verification returning cosine similarities of candidates against a single reference image.
        """
        if not candidate_images or reference_image is None:
            return []

        # Get reference embedding [1, embedding_dim]
        reference_features = self.get_image_features([reference_image])

        # Get candidate embeddings [num_candidates, embedding_dim]
        candidate_features = self.get_image_features(candidate_images, batch_size=batch_size)

        # Cosine similarities: [num_candidates, 1]
        with torch.inference_mode():
            similarities = candidate_features @ reference_features.T
            
        return similarities.detach().float().cpu().numpy().flatten().tolist()

    def score_image_against_texts(
        self,
        image: Image.Image,
        texts: list[str],
    ) -> dict[str, float]:
        """
        Score a single PIL image against text prompts.
        Returns cosine similarities.
        """
        if not texts:
            return {}

        batch_scores = self.score_images_against_texts([image], texts)
        return batch_scores[0] if batch_scores else {}

    def score_images_against_texts(
        self,
        images: list[Image.Image],
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """
        Batch CLIP verification returning cosine similarities.
        One GPU/CPU forward pass can verify multiple YOLO detections.
        """
        if not images or not texts:
            return [{} for _ in images] if images else []

        text_features = self.get_text_features(texts)  # [num_texts, embedding_dim]
        all_similarities: list[dict[str, float]] = []

        for i in range(0, len(images), batch_size):
            chunk_images = images[i : i + batch_size]
            image_tensors = torch.stack([
                self.preprocess(img.convert("RGB")) for img in chunk_images
            ]).to(self.device)

            with torch.inference_mode():
                image_features = self.model.encode_image(image_tensors)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)

                # Cosine similarities: [batch_size, num_texts]
                similarities = image_features @ text_features.T

            values = similarities.detach().float().cpu().numpy()

            for row in values:
                all_similarities.append({
                    text: float(score) for text, score in zip(texts, row)
                })

        return all_similarities

    def score_image_batch_against_texts(
        self,
        images: list[Image.Image],
        texts: list[str],
        batch_size: int = 32,
    ) -> list[dict[str, float]]:
        """Alias for score_images_against_texts for backward compatibility."""
        return self.score_images_against_texts(images, texts, batch_size=batch_size)
