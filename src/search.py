import numpy as np
import pandas as pd
import joblib
from pathlib import Path


class ImageSearchEngine:

    def __init__(
        self,
        embeddings_path: Path,
        metadata_path: Path,
        pca_path: Path | None = None,
    ):

        self.embeddings = np.load(
            embeddings_path
        )

        self.metadata = pd.read_csv(
            metadata_path
        )

        # Load PCA transformer if provided.
        # The same fitted PCA must be used for
        # both database and query embeddings.
        self.pca = None

        if pca_path is not None:
            self.pca = joblib.load(
                pca_path
            )

        # Normalize embeddings so that
        # dot product becomes cosine similarity.
        norms = np.linalg.norm(
            self.embeddings,
            axis=1,
            keepdims=True,
        )

        self.normalized_embeddings = (
            self.embeddings / norms
        )

    def transform_query(
        self,
        query_embedding,
    ):
        """
        Transform a query embedding into the
        same vector space used by the database.
        """

        query_embedding = np.asarray(
            query_embedding
        )

        # If PCA is being used, transform the
        # original 1280-dimensional embedding
        # into the database dimensionality.
        if self.pca is not None:

            query_embedding = (
                self.pca.transform(
                    query_embedding.reshape(1, -1)
                )[0]
            )

        return query_embedding

    # Search for the most similar images
    # Parameters
    # query_embedding: Embedding of the query image.
    # top_k: Number of images to retrieve.
    # exclude_index: Optional index to exclude from results.
    def search(
        self,
        query_embedding,
        top_k=5,
        exclude_index=None,
    ):

        query_embedding = np.asarray(
            query_embedding
        )

        # Normalize query embedding.
        query_norm = np.linalg.norm(
            query_embedding
        )

        if query_norm == 0:
            raise ValueError(
                "Query embedding has zero norm."
            )

        query_embedding = (
            query_embedding / query_norm
        )

        # Cosine similarity because both
        # database and query vectors are normalized.
        similarities = (
            self.normalized_embeddings
            @ query_embedding
        )

        # Prevent retrieving the query itself.
        if exclude_index is not None:
            similarities[
                exclude_index
            ] = -np.inf

        # Get indices of highest similarities.
        top_indices = np.argsort(
            similarities
        )[::-1][:top_k]

        # Retrieve metadata.
        results = self.metadata.iloc[
            top_indices
        ].copy()

        # Add similarity score.
        results["similarity"] = similarities[
            top_indices
        ]

        return results.reset_index(
            drop=True
        )


def load_search_engine(
    pca_components=512,
):

    from src.config import EMBEDDINGS_DIR

    if pca_components is None:

        embeddings_path = (
            EMBEDDINGS_DIR
            / "embeddings.npy"
        )

        pca_path = None

    else:

        embeddings_path = (
            EMBEDDINGS_DIR
            / f"embeddings_pca_{pca_components}.npy"
        )

        # At the moment we use the fitted
        # 512-dimensional PCA transformer.
        if pca_components == 512:

            pca_path = (
                EMBEDDINGS_DIR
                / "pca_512.joblib"
            )

        else:

            pca_path = (
                EMBEDDINGS_DIR
                / f"pca_{pca_components}.joblib"
            )

    metadata_path = (
        EMBEDDINGS_DIR
        / "metadata.csv"
    )

    return ImageSearchEngine(
        embeddings_path=embeddings_path,
        metadata_path=metadata_path,
        pca_path=pca_path,
    )
