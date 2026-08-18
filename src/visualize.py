import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from src.config import (
    EMBEDDINGS_DIR,
    RESULTS_DIR,
    CLASS_NAMES,
)


def plot_embeddings_2d():

    embeddings = np.load(EMBEDDINGS_DIR / "embeddings.npy")

    metadata = np.genfromtxt(
        EMBEDDINGS_DIR
        / "metadata.csv",
        delimiter=",",
        names=True,
        dtype=None,
        encoding="utf-8",
    )

    labels = metadata["label"]

    # For visualisation, use a subset
    # to keep the plot readable.
    max_points = 3000

    embeddings = embeddings[:max_points]
    labels = labels[:max_points]

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(embeddings)

    plt.figure(figsize=(10, 8))

    for class_id, class_name in enumerate(CLASS_NAMES):

        mask = labels == class_id

        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            label=class_name,
            alpha=0.6,
        )

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.title(
        "CIFAR-10 Image Embeddings "
        "Projected with PCA"
    )

    plt.legend()

    plt.tight_layout()

    output_path = (RESULTS_DIR / "pca_2d.png")

    plt.savefig(output_path, dpi=200)

    plt.close()

    print(f"Plot saved to: {output_path}")

if __name__ == "__main__":
    plot_embeddings_2d()
