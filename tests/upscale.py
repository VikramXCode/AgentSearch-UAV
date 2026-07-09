from PIL import Image

img = Image.open("sample_images/cars.jpg")

img = img.resize((1440, 960), Image.Resampling.LANCZOS)

img.save("sample_images/upscaled/cars_4x.jpg")

print("Saved!")