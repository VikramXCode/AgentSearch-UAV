from pathlib import Path


def save_pil_image(image, output_path: str) -> None:

    # JPEG cannot store alpha channels, so we strip transparency only when the
    # destination format is JPEG. PNG outputs keep their original mode.
    suffix = Path(output_path).suffix.lower()

    if suffix in {".jpg", ".jpeg"} and image.mode in {"RGBA", "LA"}:
        image = image.convert("RGB")
    elif suffix in {".jpg", ".jpeg"} and image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGB")

    image.save(output_path)