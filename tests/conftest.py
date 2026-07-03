import pytest
import os
from PIL import Image

@pytest.fixture(scope="session")
def dummy_images_dir(tmp_path_factory):
    # Create a temporary directory for images
    img_dir = tmp_path_factory.mktemp("dummy_images")

    # Create a valid image
    valid_img_path = img_dir / "valid1.jpg"
    img = Image.new('RGB', (60, 30), color = 'red')
    img.save(valid_img_path)

    # Create a corrupted image (just a text file with a .jpg extension)
    corrupted_img_path = img_dir / "corrupted.jpg"
    corrupted_img_path.write_text("This is not an image.")

    return str(img_dir)
