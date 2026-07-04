import pytest
import io
import os
from unittest.mock import patch, MagicMock
from app import app, get_image_vector_from_bytes
from PIL import Image

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'Mars Mastcam Image Search' in response.data

@patch('app.get_image_vector_from_bytes')
@patch('app.get_qdrant')
def test_search_success(mock_get_qdrant, mock_get_image_vector, client):
    mock_get_image_vector.return_value = [0.1, 0.2]

    mock_qdrant_client = MagicMock()
    mock_get_qdrant.return_value = mock_qdrant_client

    hit1 = MagicMock()
    hit1.score = 0.95
    hit1.payload = {'filename': 'img1.jpg'}
    mock_qdrant_client.search.return_value = [hit1]

    # Create dummy image file
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    response = client.post('/search', data={
        'file': (img_byte_arr, 'test.jpg')
    })

    assert response.status_code == 200
    data = response.get_json()
    assert 'matches' in data
    assert len(data['matches']) == 1
    assert data['matches'][0]['filename'] == 'img1.jpg'

def test_search_no_file(client):
    response = client.post('/search')
    assert response.status_code == 400
    assert b'No file uploaded' in response.data

def test_get_image_vector_from_bytes_invalid_image():
    with pytest.raises(ValueError, match="Invalid image:"):
        get_image_vector_from_bytes(b"not an image")
def test_search_empty_filename(client):
    # Create dummy image file
    img = Image.new('RGB', (10, 10), color='red')
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG')
    img_byte_arr.seek(0)

    response = client.post('/search', data={
        'file': (img_byte_arr, '')
    })

    assert response.status_code == 400
    assert b'No selected file' in response.data
def test_serve_thumbnail(client, tmp_path, monkeypatch):
    # Patch the THUMBNAIL_DIRECTORY in the app module
    monkeypatch.setattr('app.THUMBNAIL_DIRECTORY', str(tmp_path))

    # Create a dummy thumbnail file
    thumb_path = tmp_path / 'test_thumb.jpg'
    thumb_content = b'dummy image content'
    thumb_path.write_bytes(thumb_content)

    response = client.get('/thumbnails/test_thumb.jpg')
    assert response.status_code == 200
    assert response.data == thumb_content

def test_serve_thumbnail_not_found(client, tmp_path, monkeypatch):
    monkeypatch.setattr('app.THUMBNAIL_DIRECTORY', str(tmp_path))
    response = client.get('/thumbnails/non_existent.jpg')
    assert response.status_code == 404
@patch('app.Image.open')
@patch('app.get_model')
def test_decompression_bomb_prevention(mock_get_model, mock_image_open):
    # Mock get_model so we don't load CLIP
    mock_get_model.return_value = (MagicMock(), MagicMock())

    # Simulate a decompression bomb error
    mock_image_open.side_effect = Image.DecompressionBombError("Image size exceeds limit")

    with pytest.raises(ValueError) as excinfo:
        get_image_vector_from_bytes(b"dummy image data")

    assert "Invalid image: Image size exceeds limit" in str(excinfo.value)
