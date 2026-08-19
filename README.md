# Visual Image Search using Deep Embeddings

A deep learning project that builds a visual image search engine using **EfficientNetV2-S**, deep image embeddings, **Principal Component Analysis (PCA)**, and **cosine similarity**.

Instead of training a model to answer:

> "What class does this image belong to?"

the system asks:

> "Which images in my database look most similar to this image?"

The project demonstrates an important idea in modern deep learning:

**Neural networks do not only learn classifiers. They also learn useful representations of data.**

---

## **Project Overview**

Traditional image classification maps an image to a class:

image --> CNN --> class prediction

For example: `image -> "cat"`

Visual search uses the neural network differently:

query image -> CNN -> embedding -> compare against image database -> most similar images


The central representation is a high-dimensional vector called an embedding.
Images that have similar visual characteristics should ideally produce embeddings that are close to each other in the learned representation space.

This project therefore explores the complete pipeline:

*Image -> EfficientNetV2-S -> 1280-D embedding -> PCA -> 512-D embedding -> Cosine similarity -> Top-K visually similar images*


The project uses CIFAR-10 as the image database.

---

## **Objectives**

The main objectives are:

- Extract meaningful visual representations from images using a pretrained CNN.
- Build an image database from deep embeddings.
- Reduce embedding dimensionality using PCA.
- Perform similarity search using cosine similarity.
- Build an interactive Streamlit application for visual retrieval.
- Evaluate how dimensionality reduction affects the representation space.
- Keep the pipeline reproducible and clearly separated into data preparation, embedding extraction, dimensionality reduction, evaluation, and search.

---

## **Dataset**

The project uses the CIFAR-10 dataset.
CIFAR-10 contains:

- 60,000 colour images
- 10 classes
- 32 × 32 pixel images
- 50,000 training images and 10,000 test images
- The ten classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.

For this project, the 10,000-image CIFAR-10 test set is used as the retrieval database.
The dataset is downloaded automatically through torchvision.
No dataset files need to be committed to the repository.

---

## **Why Image Embeddings?**

A classifier produces a probability distribution over predefined classes.
For example:
cat 0.92,
dog 0.04,
bird 0.02,
...


However, classification does not directly answer: *"Which images are visually similar to this image?"*

Embeddings provide a different representation.
Instead of mapping an image directly to a class, the CNN maps it to a vector:

image -> [0.12, -0.34, 0.87, ..., 0.21]

In this project, EfficientNetV2-S produces a 1280-dimensional representation.
This representation can then be compared mathematically.
The key idea is that the semantic information learned by the neural network is encoded in the geometry of the embedding space.

---

## **Model: EfficientNetV2-S**

The project uses a pretrained EfficientNetV2-S model from torchvision.
The network is used as a feature extractor rather than as a conventional classifier.

Conceptually:
`Input image -> EfficientNetV2-S -> feature representation -> 1280-dimensional embedding`


The pretrained network provides a rich visual representation without requiring the project to train a CNN from scratch.
This is another example of transfer learning.
Instead of learning visual features from zero, the project reuses representations learned by a large pretrained model.

---

## **Embedding Extraction**

Each CIFAR-10 image is passed through EfficientNetV2-S.
The classifier output is not the representation used for retrieval.
Instead, the network's feature representation is extracted before the final classification layer.

The resulting embedding matrix has the shape: `(10000, 1280)`


This means:

- 10,000 images
- × 1,280 features per image

The embeddings are stored in:
```text 
embeddings/embeddings.npy
```

This file is approximately 49 MB and is therefore excluded from the Git repository.
The smaller PCA-reduced representations are included instead.

---

## **Why PCA?**

The original embedding has 1,280 dimensions.
Although this representation is useful, high-dimensional vectors can be expensive to store and compare.

Principal Component Analysis (PCA) provides a way to reduce dimensionality while preserving as much variance as possible.
The project evaluates several representations:

`1280 dimensions -> 512 -> 256 -> 128 -> 64 -> 32 -> 2`


For example: `1280-D -> 512-D` reduces the dimensionality by 60% while attempting to preserve the most important structure of the original representation.

The PCA transformation is fitted on the database embeddings.

---

## **An Important Detail: The Same PCA Must Be Used for the Query**

A critical part of the implementation is that the query image must be transformed using the same fitted PCA model as the database embeddings.

The database pipeline is:
database image -> EfficientNetV2-S -> 1280-D embedding -> fitted PCA -> 512-D embedding


A new query image follows exactly the same representation pipeline:
query image -> EfficientNetV2-S -> 1280-D embedding -> same fitted PCA -> 512-D embedding


Only then can the two representations be compared.

The fitted PCA transformation is stored in:
```text
embeddings/pca_512.joblib
```

This avoids fitting a new PCA model for every query.
It also guarantees that the database and query embeddings live in the same vector space.
This is essential for meaningful similarity search.

---

## **Cosine Similarity**

Once both the database and query embeddings are represented in the same space, the system calculates their similarity.
The project uses cosine similarity:

`cosine_similarity(q, x) = (q · x) / (||q|| ||x||)`


where:

- `q` is the query embedding;
- `x` is a database embedding;
- `·` is the dot product;
- `||x||` is the vector magnitude.

Cosine similarity measures the angle between two vectors rather than their absolute magnitude.
A value closer to 1 means the vectors point in similar directions; a value closer to 0 indicates little directional similarity.

---

## **Efficient Similarity Computation**

The database embeddings are normalized when the search engine is initialized.
For normalized vectors:
**cosine similarity = dot product**

Therefore, retrieval can be performed efficiently using matrix multiplication:
**database embeddings × query embedding -> similarity scores**


The system then selects the highest-scoring images.
The default number of retrieved images is `Top-K = 5`.

---

## **Complete Retrieval Pipeline**

The complete system can be summarized as:
QUERY IMAGE<br>
↓<br>
EfficientNetV2-S<br>
↓<br>
1280-D embedding<br>
↓<br>
fitted PCA (512-D)<br>
↓<br>
Normalize vector<br>
↓<br>
Cosine similarity with CIFAR-10 embedding DB (10,000 × 512)<br>
↓<br>
Rank similarities<br>
↓<br>
Top-K results<br>

---

## **Project Structure**
visual-image-search/<br>
|<br>
├── data/<br>
│ └── CIFAR-10 dataset<br>
│<br>
├── embeddings/<br>
│ ├── embeddings.npy<br>
│ ├── embeddings_pca_512.npy<br>
│ ├── embeddings_pca_256.npy<br>
│ ├── embeddings_pca_128.npy<br>
│ ├── embeddings_pca_64.npy<br>
│ ├── embeddings_pca_32.npy<br>
│ ├── embeddings_pca_2.npy<br>
│ ├── metadata.csv<br>
│ ├── pca_512.joblib<br>
│ └── pca_variance.json<br>
│<br>
├── results/<br>
│ ├── evaluation.json<br>
│ └── pca_2d.png<br>
│<br>
├── src/<br>
│ ├── init.py<br>
│ ├── build_pca.py<br>
│ ├── config.py<br>
│ ├── dataset.py<br>
│ ├── evaluate.py<br>
│ ├── extract_embeddings.py<br>
│ ├── model.py<br>
│ ├── search.py<br>
│ └── visualize.py<br>
│<br>
├── app.py<br>
├── .gitignore<br>
├── requirements.txt<br>
└── README.md<br>


---

## **Code Organization**

The project is intentionally divided into separate components.

### `dataset.py`
Responsible for:

- downloading/loading CIFAR-10;
- preparing images;
- creating the dataset used by the embedding pipeline.

### `model.py`
Defines the pretrained EfficientNetV2-S feature extractor.

### `extract_embeddings.py`
Runs images through EfficientNetV2-S and creates the original 1280-dimensional embeddings.

### `build_pca.py`
Fits PCA models and generates reduced representations:

- 512, 256, 128, 64, 32, 2 dimensions

For the 512-dimensional representation, the fitted PCA transformer is also saved so that new query images can be transformed consistently.

### `search.py`
Implements the retrieval engine:

- loads embeddings;
- normalizes them;
- transforms query embeddings;
- calculates cosine similarity;
- ranks results;
- returns the most similar images.

### `evaluate.py`
Evaluates the retrieval system and stores quantitative results.

### `visualize.py`
Creates visualizations of the embedding space.

### `app.py`
Provides the interactive Streamlit interface.

---

## **PCA Experiments**

The project does not only create one PCA representation.
Several dimensionalities are generated:

- 512, 256, 128, 64, 32, 2

This allows the effect of dimensionality reduction to be investigated experimentally.

The amount of variance explained by each representation is stored in:
```text
embeddings/pca_variance.json
```

The 2-dimensional representation is also visualized in:
```text
results/pca_2d.png
```


The 2D representation is particularly useful for understanding the geometry of the learned embedding space.
However, reducing a representation to 2 dimensions is mainly useful for visualization. It is not necessarily the best representation for accurate retrieval.

---

## **Why 512 Dimensions for the Application?**

The interactive application uses the 512-dimensional PCA representation.
The reasoning is:
Original representation (1280 dimensions) -> PCA -> 512 dimensions -> Similarity search


This provides a significantly more compact representation while retaining substantially more information than very aggressive reductions such as 32 or 2 dimensions.
The other PCA dimensions are retained for experimentation and comparison.

---

## **Reproducibility**

The project uses a fixed random seed:
```text
RANDOM_SEED = 42
```

This is used for reproducible PCA configuration and experimental behaviour where randomness is involved.

The project also separates: raw data -> embeddings -> PCA representations -> search -> evaluation
This makes the pipeline easier to reproduce and debug.

---

## **Environment**

The project was developed and tested on:

macOS
Apple Silicon
Python 3.12
PyTorch uses Apple's Metal Performance Shaders backend when available.
The application can therefore run using: MPS -> CUDA -> CPU
depending on the available hardware and configuration.

---

## **Compatible Python Version**

Recommended: **Python 3.12.x**

The project was developed using Python 3.12 and should be run with Python 3.12 for the most reproducible environment.

---

## **Main Dependencies**
```text
The project uses:
torch==2.13.0
torchvision==0.28.0
numpy==2.5.2
pandas==3.0.5
scikit-learn==1.9.0
joblib==1.5.3
matplotlib==3.11.1
Pillow==12.3.0
streamlit==1.61.1
tqdm==4.70.0
```

All direct dependencies are listed in requirements.txt

---

## **Installation**

### 1. Clone the repository
```text
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd visual-image-search
```

### 2. Create a Python 3.12 virtual environment
```text
python3.12 -m venv .venv
```

### 3. Activate the virtual environment
for macOS / Linux
```text
source .venv/bin/activate
```

for Windows
```text
.venv\Scripts\activate
```

### 4. Install dependencies
```text
pip install -r requirements.txt
```

### 5. Verify Python
```text
which python
python --version
```

The Python version should report Python 3.12.x.

---

## **Dataset**

CIFAR-10 is downloaded automatically by torchvision.
No manual dataset download is required.
The dataset is stored locally under data/.
The data/ directory is excluded from Git.

---

## **Running the pipeline**
The project can be executed in several stages.

**Step 1 — Prepare the dataset**
```text
python src/dataset.py
```

This verifies that CIFAR-10 can be loaded correctly.

**Step 2 — Extract image embeddings**
```text
python src/extract_embeddings.py
```

This generates embeddings/embeddings.npy with shape (10000, 1280).
The resulting matrix contains one 1280-dimensional embedding for each database image.

**Step 3 — Build PCA representations**
```text
python src/build_pca.py
```
This generates: embeddings_pca_512.npy, embeddings_pca_256.npy, embeddings_pca_128.npy, embeddings_pca_64.npy, embeddings_pca_32.npy, embeddings_pca_2.npy and and pca_variance.json.

For the 512-dimensional representation, it also saves pca_512.joblib.
This fitted PCA model is required to transform new query images into the same 512-dimensional space.

**Step 4 — Evaluate the retrieval system**
```text
python src/evaluate.py
```
The evaluation results are saved under results/evaluation.json.

**Step 5 — Generate visualizations**
```text
python src/visualize.py
```
The PCA visualization is saved under results/pca_2d.png.

---

## **Running the Web Application**

The project includes an interactive Streamlit interface.
Start it with:
```text
streamlit run app.py
```

Streamlit will provide a local URL such as http://localhost:8501.

The application allows a user to:

Upload an image.
Extract its EfficientNetV2-S embedding.
Transform the embedding using the fitted 512-dimensional PCA.
Compare it against the CIFAR-10 embedding database.
Retrieve the most visually similar images.
Inspect similarity scores and predicted classes.

---

## **Evaluation Philosophy**

A key distinction in this project is between:

- **Classification**: image -> class
- **Retrieval**: image -> ranked list of similar images

The second task requires evaluating whether visually or semantically related images appear near the top of the ranking.
Overall similarity scores alone are therefore not sufficient to fully understand retrieval quality.
The project combines:

+ retrieval evaluation;
+ embedding-space visualization;
+ PCA variance analysis;
+ ualitative inspection of retrieved examples.

---

## **Results and Interpretation**

The project produces several forms of evidence.

**Embedding dimensionality**
The original EfficientNetV2-S representation has 1280 dimensions.
PCA produces: 512, 256, 128, 64, 32, 2 dimensional representations.

**PCA variance**
The explained variance for each PCA representation is stored in embeddings/pca_variance.json.
This allows the information retained by different dimensionalities to be compared quantitatively.

**Embedding visualization**
The 2D PCA representation provides a visual approximation of the geometry of the embedding space.
Classes that form relatively coherent regions indicate that the pretrained representation captures meaningful visual structure.
However, the 2D projection should not be interpreted as the complete structure of the original 1280-dimensional space.

---

## **Important Engineering Considerations**

### **Same representation space**

The database and query must be represented in the same feature space.
This means that if the database uses 512-D PCA embeddings, the query must also be transformed into the same 512-D PCA space before calculating similarity.

### **Normalization**

Embeddings are L2-normalized before cosine similarity is computed.
This allows cosine similarity to be implemented efficiently as a dot product.

### **Pretrained model vs. training from scratch**

This project does not train EfficientNetV2-S from scratch.
Instead, it uses the pretrained network as a visual feature extractor.
This significantly reduces training requirements and demonstrates a practical transfer-learning workflow.

---

## **Limitations**

There are several limitations to the current implementation.

### **Dataset size**
The retrieval database contains 10,000 CIFAR-10 images.
A production visual search system would typically operate over much larger collections.

### **Search complexity**
The current implementation compares the query against all database embeddings.
For a database of 10,000 images this is completely reasonable.
For millions of images, a brute-force matrix comparison would become increasingly expensive.
A production-scale system could use approximate nearest-neighbour methods such as FAISS, Annoy, HNSW, or vector databases.

### **Domain mismatch**
EfficientNetV2-S was pretrained on a large natural-image dataset, while CIFAR-10 contains very small 32 × 32 images.
The resulting embedding space is therefore not specifically optimized for CIFAR-10 retrieval.

### **Retrieval quality**
The current system relies on the representation learned by the pretrained model.
A retrieval model trained specifically with metric-learning objectives could potentially produce a better embedding space.

---

## **Conclusion**

This project implements a complete visual search pipeline using a pretrained deep neural network.
The system:
extracts deep visual features
    -> represents images as embeddings
    -> reduces dimensionality with PCA
    -> normalizes the representations
    -> computes cosine similarity
    -> returns the most similar images

The project demonstrates how a pretrained CNN can be repurposed from classification into a content-based retrieval system.
More importantly, it explores the idea that modern neural networks can be understood not only as predictors, but as representation learning systems.

That representation-learning perspective is fundamental to many modern applications, including:
- image search
- recommendation systems
- semantic search
- face recognition
- retrieval-augmented systems
- multimodal models
- image-text search
- and modern vector databases

---

## **Author**
Olivia Méndez Blanco. This project was developed as a practical study of: deep learning, transfer learning, representation learning, image embeddings, dimensionality reduction, vector similarity, information retrieval, reproducible machine learning and scalable search architectures.
