import os
import random
from PIL import Image, ImageDraw

IMAGE_DIRECTORY = os.environ.get("IMAGE_DIRECTORY", "./images")
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)

def generate_image(filename):
    # Procedurally generate a random 256x256 image
    width, height = 256, 256
    image = Image.new('RGB', (width, height), color=(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
    draw = ImageDraw.Draw(image)

    # Draw some random shapes
    for _ in range(10):
        shape_type = random.choice(['rectangle', 'ellipse'])
        x0, y0 = random.randint(0, width), random.randint(0, height)
        x1, y1 = random.randint(x0, width), random.randint(y0, height)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        if shape_type == 'rectangle':
            draw.rectangle([x0, y0, x1, y1], fill=color)
        else:
            draw.ellipse([x0, y0, x1, y1], fill=color)

    filepath = os.path.join(IMAGE_DIRECTORY, filename)
    image.save(filepath)
    print(f"Generated {filepath}")

if __name__ == "__main__":
    print(f"Generating test images in {IMAGE_DIRECTORY}...")
    for i in range(1, 11):
        generate_image(f"test_image_{i}.jpg")
    print("Done!")
