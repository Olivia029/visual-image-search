import json

import joblib
import numpy as np
from sklearn.decomposition import PCA

from src.config import (
    EMBEDDINGS_DIR,
    PCA_COMPONENTS,
    RANDOM_SEED,
)


def build_pca_embeddings():
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"

    embeddings = np.load(embeddings_path)

    print(f"Original embedding shape: {embeddings.shape}")

    max_components = min(
        embeddings.shape[0],
        embeddings.shape[1],
    )

    valid_components = [
        n
        for n in PCA_COMPONENTS
        if n <= max_components
    ]

    explained_variance = {}

    for n_components in valid_components:

        print(
            f"\nComputing PCA with "
            f"{n_components} components..."
        )

        pca = PCA(
            n_components=n_components,
            random_state=RANDOM_SEED,
        )

        reduced = pca.fit_transform(embeddings)

        output_path = (
            EMBEDDINGS_DIR
            / f"embeddings_pca_{n_components}.npy"
        )

        np.save(output_path, reduced)

        variance = float(
            pca.explained_variance_ratio_.sum()
        )

        explained_variance[str(n_components)] = variance

        # Save the fitted PCA model.
        # This is needed to transform new query images
        # into the same feature space as the database.
        if n_components == 512:
            pca_path = (
                EMBEDDINGS_DIR
                / "pca_512.joblib"
            )

            joblib.dump(
                pca,
                pca_path,
            )

            print(
                f"Saved PCA transformer to: "
                f"{pca_path}"
            )

        print(
            f"Reduced shape: {reduced.shape}"
        )

        print(
            f"Explained variance: "
            f"{variance:.4f}"
        )

    with open(
        EMBEDDINGS_DIR / "pca_variance.json",
        "w",
    ) as file:

        json.dump(
            explained_variance,
            file,
            indent=4,
        )

    print("\nPCA completed.")


if __name__ == "__main__":
    build_pca_embeddings()