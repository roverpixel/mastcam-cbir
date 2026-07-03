import pytest
import sys
import torch
from unittest.mock import patch, MagicMock

import search

@patch('search.CLIPModel.from_pretrained')
@patch('search.CLIPProcessor.from_pretrained')
def test_get_image_vector(mock_processor, mock_model, dummy_images_dir):
    # Setup mocks
    mock_processor_instance = MagicMock()
    mock_processor.return_value = mock_processor_instance
    mock_inputs = MagicMock()
    mock_processor_instance.return_value = mock_inputs
    mock_inputs.to.return_value = mock_inputs

    mock_model_instance = MagicMock()
    mock_model.return_value = mock_model_instance
    mock_model_instance.to.return_value = mock_model_instance

    # Fake output feature vector for 1 image
    fake_features = torch.tensor([[0.6, 0.8]])
    mock_model_instance.get_image_features.return_value = fake_features

    valid_image_path = f"{dummy_images_dir}/valid1.jpg"
    vector = search.get_image_vector(valid_image_path)

    # Check that vector is normalized
    assert len(vector) == 2
    assert abs(vector[0]**2 + vector[1]**2 - 1.0) < 1e-5

@patch('search.CLIPModel.from_pretrained')
@patch('search.CLIPProcessor.from_pretrained')
def test_get_image_vector_corrupted_image(mock_processor, mock_model, dummy_images_dir, capsys):
    corrupted_image_path = f"{dummy_images_dir}/corrupted.jpg"

    with pytest.raises(SystemExit) as pytest_wrapped_e:
        search.get_image_vector(corrupted_image_path)

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1

@patch('search.QdrantClient')
@patch('search.get_image_vector')
@patch('sys.argv', ['search.py', 'dummy_image.jpg'])
def test_main_search_success(mock_get_image_vector, mock_qdrant_client, capsys):
    mock_get_image_vector.return_value = [0.1, 0.2]

    mock_client_instance = mock_qdrant_client.return_value

    # Create fake hit objects
    hit1 = MagicMock()
    hit1.score = 0.95
    hit1.payload = {'filepath': '/path/to/img1.jpg'}

    hit2 = MagicMock()
    hit2.score = 0.85
    hit2.payload = {'filepath': '/path/to/img2.jpg'}

    mock_client_instance.search.return_value = [hit1, hit2]

    search.main()

    mock_client_instance.search.assert_called_once_with(
        collection_name=search.COLLECTION_NAME,
        query_vector=[0.1, 0.2],
        limit=5
    )

    captured = capsys.readouterr()
    assert "Match Score: 0.9500 | File: /path/to/img1.jpg" in captured.out
    assert "Match Score: 0.8500 | File: /path/to/img2.jpg" in captured.out

@patch('sys.argv', ['search.py'])
def test_main_no_args(capsys):
    with pytest.raises(SystemExit) as pytest_wrapped_e:
        search.main()

    assert pytest_wrapped_e.type == SystemExit
    assert pytest_wrapped_e.value.code == 1

    captured = capsys.readouterr()
    assert "Usage: python search.py <image_path>" in captured.out
