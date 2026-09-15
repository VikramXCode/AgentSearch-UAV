from PIL import Image

from utils.image_saving import save_pil_image

img = Image.open("sample_images/cars.jpg")

img = img.resize((1440, 960), Image.Resampling.LANCZOS)

# Use the shared helper so JPEG saves automatically drop alpha if needed.
save_pil_image(img, "sample_images/upscaled/cars_4x.jpg")

print("Saved!")