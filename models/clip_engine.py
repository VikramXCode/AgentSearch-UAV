from __future__ import annotations

from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import torch
from PIL import Image

try:
	import clip
except ImportError as exc:  # pragma: no cover - handled at runtime
	clip = None
	_clip_import_error = exc
else:
	_clip_import_error = None


class CLIPEngine:
	"""
	High-performance CLIP Engine with singleton instance caching,
	pre-computed normalized text embeddings, and batched image forward passes.
	"""

	def __init__(self, model_name: str = "ViT-B/32"):
		if clip is None:
			raise RuntimeError(
				"CLIP is not installed. Install the `clip` package from the CLIP repository before running verification."
			) from _clip_import_error

		self.model_name = model_name
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
		self.model.eval()
		self._text_features_cache: Dict[Tuple[str, ...], torch.Tensor] = {}

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
		with torch.no_grad():
			text_features = self.model.encode_text(text_tokens)
			text_features = text_features / text_features.norm(dim=-1, keepdim=True)

		self._text_features_cache[texts_key] = text_features
		return text_features

	def score_image_against_texts(self, image: Image.Image, texts: list[str]) -> dict[str, float]:
		"""Score a single PIL image against text prompts."""
		if len(texts) == 0:
			return {}

		batch_scores = self.score_image_batch_against_texts([image], texts)
		return batch_scores[0] if batch_scores else {}

	def score_image_batch_against_texts(
		self,
		images: list[Image.Image],
		texts: list[str],
		batch_size: int = 32,
	) -> list[dict[str, float]]:
		"""
		Score a batch of PIL images against text prompts using a single vectorized forward pass.
		
		Args:
			images: List of PIL images (object crops)
			texts: List of text prompts
			batch_size: Sub-batch size for GPU/CPU memory efficiency
		
		Returns:
			List of dicts mapping each text prompt to its probability score for each image.
		"""
		if not images or not texts:
			return [{} for _ in images]

		text_features = self.get_text_features(texts)  # [num_texts, embedding_dim]
		all_probabilities: list[dict[str, float]] = []

		for i in range(0, len(images), batch_size):
			chunk_images = images[i : i + batch_size]
			image_tensors = torch.stack([
				self.preprocess(img.convert("RGB")) for img in chunk_images
			]).to(self.device)

			with torch.no_grad():
				image_features = self.model.encode_image(image_tensors)
				image_features = image_features / image_features.norm(dim=-1, keepdim=True)

				# Cosine similarity logits: [batch_size, num_texts]
				logits = image_features @ text_features.T
				probs = logits.softmax(dim=-1).detach().cpu().numpy()

			for row in probs:
				all_probabilities.append({
					text: float(score) for text, score in zip(texts, row)
				})

		return all_probabilities
