from __future__ import annotations

from functools import lru_cache

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

	# Cache the model once so the verification step can reuse it across many detections.
	def __init__(self, model_name: str = "ViT-B/32"):
		if clip is None:
			raise RuntimeError(
				"CLIP is not installed. Install the `clip` package from the CLIP repository before running verification."
			) from _clip_import_error

		self.model_name = model_name
		self.device = "cuda" if torch.cuda.is_available() else "cpu"
		self.model, self.preprocess = clip.load(model_name, device=self.device, jit=False)
		self.model.eval()

	@classmethod
	@lru_cache(maxsize=1)
	def shared(cls) -> "CLIPEngine":
		return cls()

	def score_image_against_texts(self, image: Image.Image, texts: list[str]) -> dict[str, float]:

		if len(texts) == 0:
			return {}

		pil_image = image.convert("RGB")
		image_tensor = self.preprocess(pil_image).unsqueeze(0).to(self.device)
		text_tokens = clip.tokenize(texts).to(self.device)

		with torch.no_grad():
			image_features = self.model.encode_image(image_tensor)
			text_features = self.model.encode_text(text_tokens)

			image_features = image_features / image_features.norm(dim=-1, keepdim=True)
			text_features = text_features / text_features.norm(dim=-1, keepdim=True)

			logits = image_features @ text_features.T
			probabilities = logits.softmax(dim=-1)[0].detach().cpu().tolist()

		return {text: float(score) for text, score in zip(texts, probabilities)}
