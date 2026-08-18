import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from src.config import (
    EMBEDDINGS_DIR,
    CLASS_NAMES,
    MAX_DATABASE_IMAGES,
)
from src.dataset import CIFAR10EmbeddingDataset
from src.model import load_model


def extract_embeddings():
    dataset = CIFAR10EmbeddingDataset()

    model, device = load_model()

    embeddings = []
    labels = []
    indices = []

    print(f"Using device: {device}")
    print(f"Images available: {len(dataset)}")

    max_images = MAX_DATABASE_IMAGES

    if max_images is None:
        max_images = len(dataset)

    for i in tqdm(range(max_images), desc="Extracting embeddings"):

        sample = dataset[i]

        image = sample["image"].unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model(image)

        embedding = embedding.squeeze(0).cpu().numpy()

        embeddings.append(embedding)
        labels.append(sample["label"])
        indices.append(sample["index"])

    embeddings = np.asarray(embeddings)

    metadata = pd.DataFrame(
        {
            "index": indices,
            "label": labels,
            "class_name": [
                CLASS_NAMES[label]
                for label in labels
            ],
        }
    )

    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    metadata_path = EMBEDDINGS_DIR / "metadata.csv"

    np.save(embeddings_path, embeddings)

    metadata.to_csv(
        metadata_path,
        index=False,
    )

    print()
    print("Embeddings saved.")
    print(f"Shape: {embeddings.shape}")
    print(f"Embeddings: {embeddings_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    extract_embeddings()
