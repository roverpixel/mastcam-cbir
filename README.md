# Mastcam-CBIR

Mastcam-CBIR is a Content-Based Image Retrieval (CBIR) system specifically designed for Mars Mastcam images. It allows users to search a database of Mars images using an image query, retrieving the most visually similar images from the dataset.

## How the Database Works

The system utilizes a modern vector database approach for image retrieval:
1.  **Image Embeddings:** We use the OpenAI CLIP model (`openai/clip-vit-base-patch32`) via PyTorch and Transformers to process images and generate 512-dimensional vector representations (embeddings). These embeddings capture the semantic meaning and visual features of the images.
2.  **Vector Database (Qdrant):** The generated embeddings are stored in a local [Qdrant](https://qdrant.tech/) vector database. The database is configured to use **Cosine similarity** (as defined in `ingest_images.py`) to measure the distance between image vectors.
3.  **Search Process:** When a user submits a query image, the system generates its CLIP embedding and compares it against all embeddings stored in Qdrant. The database quickly returns the vectors with the highest Cosine similarity to the query, which correspond to the most visually similar images.

## Setup & Installation

You can run this project locally or using Docker.

### Local Setup

1.  Clone the repository and navigate to the project directory.
2.  Install the required dependencies using pip:
    ```bash
    pip install -r requirements.txt
    ```

### Docker Setup

The project includes a `Dockerfile` and `docker-compose.yml` for easy containerization.

The Docker Compose configuration uses two environment variables to configure volume bind mounts:
*   `IMAGE_DIRECTORY`: Defaults to `/images` inside the container, mapped to `./images` locally.
*   `DB_PATH`: Defaults to `/db` inside the container, mapped to `./db` locally.

## Ingesting Images ("Training")

To populate the database, you need to "ingest" your dataset of Mars images. This process generates the CLIP embeddings and stores them in Qdrant.

### Locally

By default, the script looks for images in `/path/to/your/mars/images` and saves the database to `./mars_qdrant_db`. You can override these using environment variables.

```bash
export IMAGE_DIRECTORY="./my_mars_images"
export DB_PATH="./my_qdrant_db"
export COLLECTION_NAME="mars_mastcam" # Optional
export BATCH_SIZE=64 # Optional, adjust based on RAM/VRAM

python ingest_images.py
```

### Using Docker Compose

1. Place your images in a local directory named `images` in the root of the project.
2. Run the ingestion process using Docker Compose:
```bash
docker-compose up app
```
This will run the `ingest_images.py` script inside the container, reading from the `./images` directory and saving the Qdrant database to the `./db` directory.

## Search Instructions

Once your database is populated, you can search for similar images using a query image.

The search script takes the path to your query image as an argument.

```bash
export DB_PATH="./my_qdrant_db" # Make sure this points to your populated database
export COLLECTION_NAME="mars_mastcam" # Optional, defaults to mars_mastcam
python search.py <path_to_query_image>
```

The script will output the top 5 matches with their similarity scores and file paths.
*(Note: If the `search` method is unavailable in your `qdrant-client` version, the script automatically falls back to `query_points`.)*