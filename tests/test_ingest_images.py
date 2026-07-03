import pytest
import os
import torch
from unittest.mock import patch, MagicMock

import ingest_images

@patch('ingest_images.QdrantClient')
def test_setup_database_collection_exists(mock_qdrant_client):
    mock_client_instance = mock_qdrant_client.return_value
    mock_client_instance.collection_exists.return_value = True

    client = ingest_images.setup_database()

    mock_client_instance.collection_exists.assert_called_once_with(collection_name=ingest_images.COLLECTION_NAME)
    mock_client_instance.create_collection.assert_not_called()
    assert client == mock_client_instance

@patch('ingest_images.QdrantClient')
def test_setup_database_collection_not_exists(mock_qdrant_client):
    mock_client_instance = mock_qdrant_client.return_value
    mock_client_instance.collection_exists.return_value = False

    client = ingest_images.setup_database()

    mock_client_instance.collection_exists.assert_called_once_with(collection_name=ingest_images.COLLECTION_NAME)
    mock_client_instance.create_collection.assert_called_once()
    assert client == mock_client_instance

@patch('ingest_images.CLIPModel.from_pretrained')
@patch('ingest_images.CLIPProcessor.from_pretrained')
@patch('ingest_images.setup_database')
def test_main_no_images(mock_setup_database, mock_processor, mock_model, capsys):
    # Set IMAGE_DIRECTORY to empty or non-existent
    ingest_images.IMAGE_DIRECTORY = "/tmp/non_existent_dir_for_test_123"

    mock_qdrant = MagicMock()
    mock_setup_database.return_value = mock_qdrant

    ingest_images.main()

    captured = capsys.readouterr()
    assert "Found 0 images to process." in captured.out
    mock_qdrant.upsert.assert_not_called()

@patch('ingest_images.CLIPModel.from_pretrained')
@patch('ingest_images.CLIPProcessor.from_pretrained')
@patch('ingest_images.setup_database')
def test_main_with_images(mock_setup_database, mock_processor, mock_model, dummy_images_dir, tmp_path, capsys):
    # Set IMAGE_DIRECTORY to dummy_images_dir
    ingest_images.IMAGE_DIRECTORY = dummy_images_dir

    # Use temporary directory for thumbnails
    ingest_images.THUMBNAIL_DIRECTORY = str(tmp_path / "thumbnails")

    # Mocking Qdrant
    mock_qdrant = MagicMock()
    mock_setup_database.return_value = mock_qdrant

    # Mocking Processor
    mock_processor_instance = MagicMock()
    mock_processor.return_value = mock_processor_instance

    # Create a mock for the inputs that has a .to() method
    mock_inputs = MagicMock()
    mock_processor_instance.return_value = mock_inputs
    mock_inputs.to.return_value = mock_inputs

    # Mocking Model
    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    # Fake features
    fake_features = torch.tensor([[1.0, 0.0]])
    mock_model_instance.get_image_features.return_value = fake_features

    ingest_images.main()

    captured = capsys.readouterr()
    assert "Skipping corrupted image" in captured.out

    # We should have found 2 images, 1 corrupted, 1 processed
    assert "Found 2 images to process." in captured.out
    mock_qdrant.upsert.assert_called_once()
