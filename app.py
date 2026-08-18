import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import datasets, transforms

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATA_DIR
from src.model import load_model
from src.search import load_search_engine

# Streamlit configuration
st.set_page_config(
    page_title="Visual Image Search",
    page_icon="🔎",
    layout="wide",
)

# Load model
@st.cache_resource
def load_embedding_model():

    model, device = load_model()

    return model, device


# Load search database
@st.cache_resource
def load_database():

    return load_search_engine()


# Load original CIFAR-10 dataset
@st.cache_resource
def load_cifar10_images():

    dataset = datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=True,
        transform=None,
    )

    return dataset


# Image preprocessing
def preprocess_image(image):

    transform = transforms.Compose(
        [
            transforms.Resize(
                (384, 384)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(
                    0.485,
                    0.456,
                    0.406,
                ),
                std=(
                    0.229,
                    0.224,
                    0.225,
                ),
            ),
        ]
    )

    return transform(image).unsqueeze(0)


# Generate embedding
def get_embedding(
    image,
    model,
    device,
):

    tensor = preprocess_image(
        image
    ).to(device)

    with torch.no_grad():

        embedding = model(
            tensor
        )

    return (
        embedding
        .squeeze(0)
        .cpu()
        .numpy()
    )


# User interface
st.title(
    "Visual Image Search"
)

st.markdown(
    """
    Upload an image and retrieve the most
    visually similar images from the CIFAR-10
    database using deep embeddings.

    **Pipeline:**

    `Image → EfficientNetV2-S → Embedding → Cosine Similarity → Similar Images`
    """
)

# Upload image
uploaded_file = st.file_uploader(
    "Upload an image",
    type=[
        "jpg",
        "jpeg",
        "png",
    ],
)


if uploaded_file is not None:

    # Open uploaded image
    image = Image.open(
        uploaded_file
    ).convert("RGB")

    # Display query
    st.subheader(
        "Query image"
    )

    st.image(
        image,
        width=300,
    )

    # Load model and database
    model, device = (
        load_embedding_model()
    )

    search_engine = (
        load_database()
    )

    # Generate query embedding
    with st.spinner(
        "Computing image embedding..."
    ):

        query_embedding = get_embedding(
            image,
            model,
            device,
        )

    st.success(
        f"Embedding generated "
        f"({len(query_embedding)} dimensions)"
    )

    # Search parameters
    top_k = st.slider(
        "Number of similar images",
        min_value=1,
        max_value=10,
        value=5,
    )

    # Perform search
    with st.spinner(
        "Searching for similar images..."
    ):

        query_embedding = search_engine.transform_query(
            query_embedding
        )

        results = search_engine.search(
            query_embedding,
            top_k=top_k,
        )

    # Display results
    st.subheader(
        "Most similar images"
    )

    # Load original CIFAR-10 images
    cifar_dataset = (
        load_cifar10_images()
    )

    columns = st.columns(
        min(top_k, 5)
    )

    for i, (_, row) in enumerate(
        results.iterrows()
    ):

        column = columns[
            i % len(columns)
        ]

        image_index = int(
            row["index"]
        )

        # Retrieve original PIL image
        retrieved_image, _ = (
            cifar_dataset[
                image_index
            ]
        )

        similarity = float(
            row["similarity"]
        )

        class_name = row[
            "class_name"
        ]

        # Display result
        column.image(
            retrieved_image,
            caption=(
                f"{class_name}\n"
                f"Cosine similarity: "
                f"{similarity:.4f}"
            ),
        )