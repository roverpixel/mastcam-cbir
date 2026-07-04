import os
import io
from werkzeug.middleware.proxy_fix import ProxyFix
from flask import Flask, request, jsonify, render_template, send_from_directory
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
from qdrant_client import QdrantClient

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max
# Apply ProxyFix to support subpaths and reverse proxies
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configuration
DB_PATH = os.environ.get("DB_PATH", "./mars_qdrant_db")
COLLECTION_NAME = os.environ.get("COLLECTION_NAME", "mars_mastcam")
THUMBNAIL_DIRECTORY = os.environ.get("THUMBNAIL_DIRECTORY", "./thumbnails")

# Ensure thumbnail directory exists
os.makedirs(THUMBNAIL_DIRECTORY, exist_ok=True)

# Device setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model_id = "openai/clip-vit-base-patch32"

# Global variables for lazy loading
_model = None
_processor = None

def get_model():
    global _model, _processor
    if _model is None:
        print(f"Loading CLIP model to {device}...")
        _model = CLIPModel.from_pretrained(model_id).to(device)
        _processor = CLIPProcessor.from_pretrained(model_id)
    return _model, _processor

def get_qdrant():
    global _qdrant_client
    if _qdrant_client is None:
        try:
            _qdrant_client = QdrantClient(path=DB_PATH)
        except Exception as e:
            print(f"Error connecting to Qdrant local database: {e}")
            raise e
    return _qdrant_client

def get_image_vector_from_bytes(image_bytes):
    model, processor = get_model()
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Invalid image: {e}")

    inputs = processor(images=[img], return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    if hasattr(features, 'pooler_output'):
        features = features.pooler_output
    elif isinstance(features, tuple):
        features = features[0]

    features = features / features.norm(dim=-1, keepdim=True)
    vectors = features.cpu().numpy().tolist()

    return vectors[0]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    return send_from_directory(THUMBNAIL_DIRECTORY, filename)

@app.route('/search', methods=['POST'])
def search():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file:
        return jsonify({'error': 'Invalid file'}), 400

    try:
        image_bytes = file.read()
        query_vector = get_image_vector_from_bytes(image_bytes)

        client = get_qdrant()

        if hasattr(client, 'search'):
            search_result = client.search(
                collection_name=COLLECTION_NAME,
                query_vector=query_vector,
                limit=10
            )
        else:
            search_result = client.query_points(
                collection_name=COLLECTION_NAME,
                query=query_vector,
                limit=10
            ).points

        results = [
            {
                'score': round(hit.score, 4),
                'filename': hit.payload['filename']
            }
            for hit in search_result
        ]

        return jsonify({'matches': results})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)), debug=True)
