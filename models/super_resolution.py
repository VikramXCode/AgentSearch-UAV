import os
from PIL import Image

from utils.image_saving import save_pil_image


class SuperResolutionEngine:

    def upscale(self, image_path: str, output_dir: str = "outputs/preprocessed", scale: int = 2) -> str:

        image = Image.open(image_path)
        width, height = image.size
        upscaled = image.resize((width * scale, height * scale), Image.Resampling.LANCZOS)

        os.makedirs(output_dir, exist_ok=True)

        file_name = os.path.splitext(os.path.basename(image_path))[0]
        output_path = os.path.join(output_dir, f"{file_name}_srx{scale}.jpg")

        # Save through a shared helper so JPEG output never fails on alpha-based images.
        save_pil_image(upscaled, output_path)

        return output_path
