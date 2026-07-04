import torch

def extract_features(images, model, processor, device):
    """
    Extracts normalized image features using the provided CLIP model and processor.

    Args:
        images: A list of PIL Images.
        model: The CLIP model.
        processor: The CLIP processor.
        device: The device to run the model on ('cuda' or 'cpu').

    Returns:
        A list of feature vectors as Python lists.
    """
    inputs = processor(images=images, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)

    if hasattr(features, 'pooler_output'):
        features = features.pooler_output
    elif isinstance(features, tuple):
        features = features[0]

    # Normalize vectors (best practice for Cosine similarity)
    features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().tolist()
