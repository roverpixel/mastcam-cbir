import sys
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import os
from qdrant_client import QdrantClient

# --- CONFIGURATION ---
DB_PATH = os.environ.get("DB_PATH", "./mars_qdrant_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "mars_mastcam")

def get_image_vector(image_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_id).to(device)
    processor = CLIPProcessor.from_pretrained(model_id)

    try:
        img = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        sys.exit(1)

    inputs = processor(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    features = features / features.norm(dim=-1, keepdim=True)
    vectors = features.cpu().numpy().tolist()

    return vectors[0]

def main():
    if len(sys.argv) < 2:
        print("Usage: python search.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    # Load Qdrant client
    try:
        client = QdrantClient(path=DB_PATH)
    except Exception as e:
        print(f"Error connecting to Qdrant local database: {e}")
        sys.exit(1)

    print(f"Extracting features for image: {image_path}")
    query_vector = get_image_vector(image_path)

    print("Searching database...")
    try:
        search_result = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=query_vector,
            limit=5  # Return top 5 matches
        )

        for hit in search_result:
            print(f"Match Score: {hit.score:.4f} | File: {hit.payload['filepath']}")
    except Exception as e:
        print(f"Error during search: {e}")

if __name__ == "__main__":
    main()
