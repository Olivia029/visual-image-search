import json

import numpy as np

from src.config import (
    EMBEDDINGS_DIR,
    RESULTS_DIR,
)

# Evaluate whether retrieved neighbours belong to the same semantic class.
def evaluate_retrieval(embeddings, labels, top_k=5):
   
    # Normalize embeddings.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    normalized = embeddings / norms

    similarities = normalized @ normalized.T

    # Ignore the query image itself.
    np.fill_diagonal(similarities, -np.inf)

    top_indices = np.argsort(similarities, axis=1)[:, ::-1][:, :top_k]

    top_labels = labels[top_indices]

    query_labels = labels[:, None]

    correct = (top_labels == query_labels)

    # Recall@K:
    # Does at least one retrieved image
    # belong to the same class?
    recall_at_k = correct.any(axis=1).mean()

    # Mean class agreement among top-k.
    class_agreement = correct.mean()

    return {f"recall@{top_k}": float(recall_at_k), f"class_agreement@{top_k}": float(class_agreement)}


def evaluate_multiple_k(embeddings, labels, ks=(1, 5, 10)):

    results = {}

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)

    normalized = embeddings / norms

    similarities = normalized @ normalized.T

    np.fill_diagonal(similarities, -np.inf)

    max_k = max(ks)

    top_indices = np.argsort(similarities,axis=1)[:, ::-1][:, :max_k]

    for k in ks:

        neighbours = top_indices[:, :k]

        neighbour_labels = labels[neighbours]

        matches = (neighbour_labels == labels[:, None])

        recall = matches.any(axis=1).mean()

        precision = matches.mean()

        results[f"recall@{k}"] = float(recall)

        results[f"precision@{k}"] = float(precision)

    return results


def main():

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

    results = evaluate_multiple_k(embeddings, labels, ks=(1, 5, 10))   
        
    print("\nRetrieval evaluation")
    print("====================")

    for metric, value in results.items():

        print(f"{metric}: " f"{value * 100:.2f}%")

    output_path = (RESULTS_DIR / "evaluation.json")

    with open(
        output_path, 
        "w",
    ) as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print()
    print(f"Results saved to: " f"{output_path}")

if __name__ == "__main__":
    main()
