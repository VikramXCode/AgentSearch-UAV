from __future__ import annotations

from functools import lru_cache

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

        print("CLIP loaded successfully.")

    @classmethod
    @lru_cache(maxsize=1)
    def shared(cls) -> "CLIPEngine":
        return cls()

    def score_image_against_texts(
        self,
        image: Image.Image,
        texts: list[str],
    ) -> dict[str, float]:
        """
        Return cosine similarities instead of softmax probabilities.

        This is important because softmax always forces the prompts
        to compete and produces a winner even when every prompt is bad.
        """

        if not texts:
            return {}

        image_tensor = (
            self.preprocess(image.convert("RGB"))
            .unsqueeze(0)
            .to(self.device)
        )

        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.inference_mode():

            image_features = self.model.encode_image(image_tensor)
            text_features = self.model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )

            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            )

            similarities = (
                image_features @ text_features.T
            )[0]

        values = similarities.detach().float().cpu().tolist()

        return {
            text: float(score)
            for text, score in zip(texts, values)
        }

    def score_images_against_texts(
        self,
        images: list[Image.Image],
        texts: list[str],
    ) -> list[dict[str, float]]:
        """
        Batch CLIP verification.

        One GPU forward pass can verify multiple YOLO detections.
        """

        if not images or not texts:
            return []

        image_batch = torch.stack(
            [
                self.preprocess(image.convert("RGB"))
                for image in images
            ]
        ).to(self.device)

        text_tokens = clip.tokenize(texts).to(self.device)

        with torch.inference_mode():

            image_features = self.model.encode_image(image_batch)
            text_features = self.model.encode_text(text_tokens)

            image_features = image_features / image_features.norm(
                dim=-1,
                keepdim=True,
            )

            text_features = text_features / text_features.norm(
                dim=-1,
                keepdim=True,
            )

            similarities = image_features @ text_features.T

        similarities = similarities.detach().float().cpu()

        results = []

        for row in similarities:

            results.append(
                {
                    text: float(score)
                    for text, score in zip(
                        texts,
                        row.tolist(),
                    )
                }
            )

        return results