import pytest
import os
import sys
import torch
from unittest.mock import patch, MagicMock
from qdrant_client import QdrantClient

import ingest_images
import search

@pytest.fixture
def temp_db_dir(tmp_path):
    # Returns a temporary directory path for the Qdrant database
    return str(tmp_path / "test_qdrant_db")

@patch('ingest_images.CLIPModel.from_pretrained')
@patch('ingest_images.CLIPProcessor.from_pretrained')
@patch('search.CLIPModel.from_pretrained')
@patch('search.CLIPProcessor.from_pretrained')
def test_integration_ingest_and_search(mock_search_processor, mock_search_model,
                                       mock_ingest_processor, mock_ingest_model,
                                       dummy_images_dir, temp_db_dir, capsys):
    """
    Integration test:
    1. Sets up a real local Qdrant database in a temp directory.
    2. Runs the ingestion script to 'train' the database with dummy images.
    3. Runs the search script to identify the closest match.
    """

    # --- 1. Setup Environment Variables ---
    ingest_images.DB_PATH = temp_db_dir
    search.DB_PATH = temp_db_dir

    # We will test using 2 valid dummy images and 1 query image
    from PIL import Image
    img2_path = os.path.join(dummy_images_dir, "valid2.jpg")
    img2 = Image.new('RGB', (60, 30), color='blue')
    img2.save(img2_path)

    ingest_images.IMAGE_DIRECTORY = dummy_images_dir

    # --- 2. Mocking CLIP models for deterministic vector outputs ---
    mock_ingest_processor_instance = MagicMock()
    mock_ingest_processor.return_value = mock_ingest_processor_instance
    mock_inputs = MagicMock()
    mock_ingest_processor_instance.return_value = mock_inputs
    mock_inputs.to.return_value = mock_inputs

    mock_ingest_model_instance = MagicMock()
    mock_ingest_model.return_value = mock_ingest_model_instance
    mock_ingest_model_instance.to.return_value = mock_ingest_model_instance

    fake_ingest_features = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    mock_ingest_model_instance.get_image_features.return_value = fake_ingest_features

    # --- 3. Execute Ingestion ("Database Training") ---
    original_vector_params = ingest_images.VectorParams

    # Do NOT monkeypatch QdrantClient class globally, let's just let it be normally created and then closed.

    def mock_vector_params(*args, **kwargs):
        kwargs['size'] = 2
        return original_vector_params(*args, **kwargs)

    with patch('ingest_images.VectorParams', side_effect=mock_vector_params):
        ingest_images.main()

    # --- 4. Search Mocking ("Identification") ---
    mock_search_processor_instance = MagicMock()
    mock_search_processor.return_value = mock_search_processor_instance
    mock_search_inputs = MagicMock()
    mock_search_processor_instance.return_value = mock_search_inputs
    mock_search_inputs.to.return_value = mock_search_inputs

    mock_search_model_instance = MagicMock()
    mock_search_model.return_value = mock_search_model_instance
    mock_search_model_instance.to.return_value = mock_search_model_instance

    fake_search_features = torch.tensor([[0.99, 0.1]])
    mock_search_model_instance.get_image_features.return_value = fake_search_features

    # --- 5. Execute Search ---
    query_image = os.path.join(dummy_images_dir, "valid1.jpg")
    with patch('sys.argv', ['search.py', query_image]):
        search.main()

    # --- 6. Verify Results ---
    captured = capsys.readouterr()

    assert "Match Score:" in captured.out
    assert "valid" in captured.out
    assert ".jpg" in captured.out
