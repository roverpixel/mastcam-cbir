import os
import glob
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient
from utils import extract_features
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from tqdm import tqdm

# --- CONFIGURATION ---
IMAGE_DIRECTORY = os.environ.get("IMAGE_DIRECTORY", "/path/to/your/mars/images")
THUMBNAIL_DIRECTORY = os.environ.get("THUMBNAIL_DIRECTORY", "./thumbnails")
DB_PATH = os.environ.get("DB_PATH", "./mars_qdrant_db")  # Database will be saved to this local folder
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "mars_mastcam")
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 64))  # Adjust based on your GPU/CPU RAM (32, 64, or 128)

def setup_database():
    """Initializes the Qdrant local database."""
    # Running Qdrant locally on disk (no Docker required)
    client = QdrantClient(path=DB_PATH)

    # Check if collection exists, if not, create it
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=512, # CLIP ViT-B/32 output dimension
                distance=Distance.COSINE
            )
        )
        print(f"Created new collection: {COLLECTION_NAME}")
    else:
        print(f"Collection {COLLECTION_NAME} already exists. Resuming/Appending...")

    return client

def main():
    # 1. Setup Device & Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device.upper()}")

    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    # 2. Setup Database
    qdrant = setup_database()

    # Create thumbnail directory if it doesn't exist
    os.makedirs(THUMBNAIL_DIRECTORY, exist_ok=True)

    # 3. Gather Image Paths
    print("Scanning directory for images...")
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    image_paths = []
    try:
        with os.scandir(IMAGE_DIRECTORY) as entries:
            for entry in entries:
                if entry.is_file() and os.path.splitext(entry.name)[1].lower() in valid_extensions:
                    image_paths.append(entry.path)
    except FileNotFoundError:
        pass

    total_images = len(image_paths)
    print(f"Found {total_images} images to process.")

    if total_images == 0:
        return

    # 4. Batch Processing Loop
    # We process in batches to maximize GPU/CPU efficiency and prevent memory crashes
    for i in tqdm(range(0, total_images, BATCH_SIZE), desc="Ingesting Images"):
        batch_paths = image_paths[i : i + BATCH_SIZE]
        valid_images = []
        valid_paths = []

        # Open images and filter out corrupted files
        for path in batch_paths:
            try:
                img = Image.open(path).convert("RGB")
                valid_images.append(img)
                valid_paths.append(path)

                # Save thumbnail
                img.thumbnail((256, 256))
                thumb_path = os.path.join(THUMBNAIL_DIRECTORY, os.path.basename(path))
                img.save(thumb_path)
            except Exception as e:
                print(f"\nSkipping corrupted image {path}: {e}")

        if not valid_images:
            continue

        # Process batch through CLIP
        try:
            vectors = extract_features(valid_images, model, processor, device)

            # 5. Prepare Qdrant Points
            points = [
                PointStruct(
                    id=i + j,
                    vector=vector,
                    payload={
                        "filepath": filepath,
                        "filename": os.path.basename(filepath)
                    }
                )
                for j, (vector, filepath) in enumerate(zip(vectors, valid_paths))
            ]

            # 6. Upload to Vector Database
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )

        except Exception as e:
            print(f"\nError processing batch starting at index {i}: {e}")

    print("\n✅ Ingestion Complete!")
    print(f"Vectors are stored safely in: {os.path.abspath(DB_PATH)}")

if __name__ == "__main__":
    main()
