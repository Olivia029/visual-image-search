Visual Image Search using Deep Embeddings

A deep learning project that builds a visual image search engine using EfficientNetV2-S, deep image embeddings, Principal Component Analysis (PCA), and cosine similarity.

Instead of training a model to answer:

"What class does this image belong to?"
the system asks:

"Which images in my database look most similar to this image?"
The project demonstrates an important idea in modern deep learning:

Neural networks do not only learn classifiers. They also learn useful representations of data.

Project Overview

Traditional image classification maps an image to a class:

text
image
  ↓
CNN
  ↓
class prediction
For example: image → "cat"

Visual search uses the neural network differently:

text
query image
     ↓
CNN
     ↓
embedding
     ↓
compare against image database
     ↓
most similar images
The central representation is a high-dimensional vector called an embedding.
Images that have similar visual characteristics should ideally produce embeddings that are close to each other in the learned representation space.

This project therefore explores the complete pipeline:

text
Image
  ↓
EfficientNetV2-S
  ↓
1280-dimensional embedding
  ↓
PCA
  ↓
512-dimensional embedding
  ↓
Cosine similarity
  ↓
Top-K visually similar images
The project uses CIFAR-10 as the image database.

Objectives

The main objectives are:

Extract meaningful visual representations from images using a pretrained CNN.
Build an image database from deep embeddings.
Reduce embedding dimensionality using PCA.
Perform similarity search using cosine similarity.
Build an interactive Streamlit application for visual retrieval.
Evaluate how dimensionality reduction affects the representation space.
Keep the pipeline reproducible and clearly separated into data preparation, embedding extraction, dimensionality reduction, evaluation, and search.
Dataset

The project uses the CIFAR-10 dataset.
CIFAR-10 contains:

60,000 colour images
10 classes
32 × 32 pixel images
50,000 training images and 10,000 test images
The ten classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
For this project, the 10,000-image CIFAR-10 test set is used as the retrieval database.
The dataset is downloaded automatically through torchvision.
No dataset files need to be committed to the repository.

Why Image Embeddings?

A classifier produces a probability distribution over predefined classes.
For example:

text
cat       0.92
dog       0.04
bird      0.02
...
However, classification does not directly answer:
"Which images are visually similar to this image?"

Embeddings provide a different representation.
Instead of mapping an image directly to a class, the CNN maps it to a vector:

text
image → [0.12, -0.34, 0.87, ..., 0.21]
In this project, EfficientNetV2-S produces a 1280-dimensional representation.
This representation can then be compared mathematically.
The key idea is that the semantic information learned by the neural network is encoded in the geometry of the embedding space.

Model: EfficientNetV2-S

The project uses a pretrained EfficientNetV2-S model from torchvision.
The network is used as a feature extractor rather than as a conventional classifier.

Conceptually:

text
Input image
     ↓
EfficientNetV2-S
     ↓
feature representation
     ↓
1280-dimensional embedding
The pretrained network provides a rich visual representation without requiring the project to train a CNN from scratch.
This is another example of transfer learning.
Instead of learning visual features from zero, the project reuses representations learned by a large pretrained model.

Embedding Extraction

Each CIFAR-10 image is passed through EfficientNetV2-S.
The classifier output is not the representation used for retrieval.
Instead, the network's feature representation is extracted before the final classification layer.

The resulting embedding matrix has the shape:

text
(10000, 1280)
This means:

10,000 images
× 1,280 features per image
The embeddings are stored in:

text
embeddings/embeddings.npy
This file is approximately 49 MB and is therefore excluded from the Git repository.
The smaller PCA-reduced representations are included instead.

Why PCA?

The original embedding has 1,280 dimensions.
Although this representation is useful, high-dimensional vectors can be expensive to store and compare.

Principal Component Analysis (PCA) provides a way to reduce dimensionality while preserving as much variance as possible.
The project evaluates several representations:

text
1280 dimensions
      ↓
512
      ↓
256
      ↓
128
      ↓
64
      ↓
32
      ↓
2
For example:

text
1280-D → 512-D
reduces the dimensionality by 60% while attempting to preserve the most important structure of the original representation.

The PCA transformation is fitted on the database embeddings.

An Important Detail: The Same PCA Must Be Used for the Query

A critical part of the implementation is that the query image must be transformed using the same fitted PCA model as the database embeddings.

The database pipeline is:

text
database image
      ↓
EfficientNetV2-S
      ↓
1280-D embedding
      ↓
fitted PCA
      ↓
512-D embedding
A new query image follows exactly the same representation pipeline:

text
query image
      ↓
EfficientNetV2-S
      ↓
1280-D embedding
      ↓
same fitted PCA
      ↓
512-D embedding
Only then can the two representations be compared.

The fitted PCA transformation is stored in:

text
embeddings/pca_512.joblib
This avoids fitting a new PCA model for every query.
It also guarantees that the database and query embeddings live in the same vector space.
This is essential for meaningful similarity search.

Cosine Similarity

Once both the database and query embeddings are represented in the same space, the system calculates their similarity.
The project uses cosine similarity:

text
cosine_similarity(q, x) = (q · x) / (||q|| ||x||)
where:

q is the query embedding;
x is a database embedding;
· is the dot product;
||x|| is the vector magnitude.
Cosine similarity measures the angle between two vectors rather than their absolute magnitude.
A value closer to:

1 means the vectors point in similar directions.
0 indicates little directional similarity.
Efficient Similarity Computation

The database embeddings are normalized when the search engine is initialized.
For normalized vectors:

text
cosine similarity = dot product
Therefore, retrieval can be performed efficiently using matrix multiplication:

text
database embeddings
        ×
query embedding
        ↓
similarity scores
The system then selects the highest-scoring images.
The default number of retrieved images is:

text
Top-K = 5
Complete Retrieval Pipeline

The complete system can be summarized as:

text
                         QUERY IMAGE
                              │
                              ▼
                       EfficientNetV2-S
                              │
                              ▼
                       1280-D embedding
                              │
                              ▼
                         fitted PCA
                              │
                              ▼
                       512-D embedding
                              │
                              ▼
                       Normalize vector
                              │
                              ▼
                    Cosine similarity
                              │
                              ▼
                ┌─────────────────────────┐
                │ CIFAR-10 embedding DB   │
                │ 10,000 × 512            │
                └─────────────────────────┘
                              │
                              ▼
                     Rank similarities
                              │
                              ▼
                       Top-K results
Project Structure

text
visual-image-search/
│
├── data/
│   └── CIFAR-10 dataset
│
├── embeddings/
│   ├── embeddings.npy
│   ├── embeddings_pca_512.npy
│   ├── embeddings_pca_256.npy
│   ├── embeddings_pca_128.npy
│   ├── embeddings_pca_64.npy
│   ├── embeddings_pca_32.npy
│   ├── embeddings_pca_2.npy
│   ├── metadata.csv
│   ├── pca_512.joblib
│   └── pca_variance.json
│
├── results/
│   ├── evaluation.json
│   └── pca_2d.png
│
├── src/
│   ├── __init__.py
│   ├── build_pca.py
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── extract_embeddings.py
│   ├── model.py
│   ├── search.py
│   └── visualize.py
│
├── app.py
├── .gitignore
├── requirements.txt
└── README.md
Code Organization

The project is intentionally divided into separate components.

dataset.py

Responsible for:

downloading/loading CIFAR-10;
preparing images;
creating the dataset used by the embedding pipeline.
model.py

Defines the pretrained EfficientNetV2-S feature extractor.

extract_embeddings.py

Runs images through EfficientNetV2-S and creates the original 1280-dimensional embeddings.

build_pca.py

Fits PCA models and generates reduced representations:

512
256
128
64
32
2
For the 512-dimensional representation, the fitted PCA transformer is also saved so that new query images can be transformed consistently.

search.py

Implements the retrieval engine:

loads embeddings;
normalizes them;
transforms query embeddings;
calculates cosine similarity;
ranks results;
returns the most similar images.
evaluate.py

Evaluates the retrieval system and stores quantitative results.

visualize.py

Creates visualizations of the embedding space.

app.py

Provides the interactive Streamlit interface.

PCA Experiments

The project does not only create one PCA representation.
Several dimensionalities are generated:

512
256
128
64
32
2
This allows the effect of dimensionality reduction to be investigated experimentally.

The amount of variance explained by each representation is stored in:

text
embeddings/pca_variance.json
The 2-dimensional representation is also visualized in:

text
results/pca_2d.png
The 2D representation is particularly useful for understanding the geometry of the learned embedding space.
However, reducing a representation to 2 dimensions is mainly useful for visualization. It is not necessarily the best representation for accurate retrieval.

Why 512 Dimensions for the Application?

The interactive application uses the 512-dimensional PCA representation.
The reasoning is:

text
Original representation
1280 dimensions
        ↓
       PCA
        ↓
512 dimensions
        ↓
Similarity search
This provides a significantly more compact representation while retaining substantially more information than very aggressive reductions such as 32 or 2 dimensions.
The other PCA dimensions are retained for experimentation and comparison.

Reproducibility

The project uses a fixed random seed:

python
RANDOM_SEED = 42
This is used for reproducible PCA configuration and experimental behaviour where randomness is involved.

The project also separates:

text
raw data
↓
embeddings
↓
PCA representations
↓
search
↓
evaluation
This makes the pipeline easier to reproduce and debug.

Environment

The project was developed and tested on:

macOS
Apple Silicon
Python 3.12
PyTorch uses Apple's Metal Performance Shaders backend when available.
The application can therefore run using:

text
MPS → CUDA → CPU
depending on the available hardware and configuration.

Compatible Python Version

Recommended:

Python 3.12.x
The project was developed using Python 3.12 and should be run with Python 3.12 for the most reproducible environment.

Main Dependencies

The project uses:

text
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
All direct dependencies are listed in requirements.txt.

Installation

1. Clone the repository

bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd visual-image-search
2. Create a Python 3.12 virtual environment

bash
python3.12 -m venv .venv
3. Activate the virtual environment

macOS / Linux

bash
source .venv/bin/activate
Windows

bash
.venv\Scripts\activate
4. Install dependencies

bash
pip install -r requirements.txt
5. Verify Python

bash
which python
python --version
The Python version should report Python 3.12.x.

Dataset

CIFAR-10 is downloaded automatically by torchvision.
No manual dataset download is required.
The dataset is stored locally under data/.
The data/ directory is excluded from Git.

Running the Pipeline

The project can be executed in several stages.

Step 1 — Prepare the dataset

bash
python src/dataset.py
This verifies that CIFAR-10 can be loaded correctly.

Step 2 — Extract image embeddings

bash
python src/extract_embeddings.py
This generates:

text
embeddings/embeddings.npy
with shape (10000, 1280).
The resulting matrix contains one 1280-dimensional embedding for each database image.

Step 3 — Build PCA representations

bash
python src/build_pca.py
This generates:

text
embeddings_pca_512.npy
embeddings_pca_256.npy
embeddings_pca_128.npy
embeddings_pca_64.npy
embeddings_pca_32.npy
embeddings_pca_2.npy
and:

text
pca_variance.json
For the 512-dimensional representation, it also saves:

text
pca_512.joblib
This fitted PCA model is required to transform new query images into the same 512-dimensional space.

Step 4 — Evaluate the retrieval system

bash
python src/evaluate.py
The evaluation results are saved under:

text
results/evaluation.json
Step 5 — Generate visualizations

bash
python src/visualize.py
The PCA visualization is saved under:

text
results/pca_2d.png
Running the Web Application

The project includes an interactive Streamlit interface.
Start it with:

bash
streamlit run app.py
Streamlit will provide a local URL such as:

text
http://localhost:8501
The application allows a user to:

Upload an image.
Extract its EfficientNetV2-S embedding.
Transform the embedding using the fitted 512-dimensional PCA.
Compare it against the CIFAR-10 embedding database.
Retrieve the most visually similar images.
Inspect similarity scores and predicted classes.
Example Query Pipeline

Suppose the user uploads an image.
The application performs:

text
Uploaded image
      ↓
Image preprocessing
      ↓
EfficientNetV2-S
      ↓
1280-D embedding
      ↓
pca_512.joblib
      ↓
512-D embedding
      ↓
L2 normalization
      ↓
Cosine similarity
      ↓
10,000 database images
      ↓
Ranking
      ↓
Top 5 results
The application therefore performs genuine content-based image retrieval rather than simply returning a classification label.

Evaluation Philosophy

A key distinction in this project is between:

Classification: image → class
Retrieval: image → ranked list of similar images
The second task requires evaluating whether visually or semantically related images appear near the top of the ranking.
Overall similarity scores alone are therefore not sufficient to fully understand retrieval quality.

The project combines:

retrieval evaluation;
embedding-space visualization;
PCA variance analysis;
qualitative inspection of retrieved examples.
Results and Interpretation

The project produces several forms of evidence.

Embedding dimensionality
The original EfficientNetV2-S representation has 1280 dimensions.
PCA produces:

512
256
128
64
32
2
dimensional representations.

PCA variance
The explained variance for each PCA representation is stored in embeddings/pca_variance.json.
This allows the information retained by different dimensionalities to be compared quantitatively.

Embedding visualization
The 2D PCA representation provides a visual approximation of the geometry of the embedding space.
Classes that form relatively coherent regions indicate that the pretrained representation captures meaningful visual structure.
However, the 2D projection should not be interpreted as the complete structure of the original 1280-dimensional space.

Important Engineering Considerations

Same representation space

The database and query must be represented in the same feature space.
This means that if the database uses 512-D PCA embeddings, the query must also be transformed into the same 512-D PCA space before calculating similarity.

Normalization

Embeddings are L2-normalized before cosine similarity is computed.
This allows cosine similarity to be implemented efficiently as a dot product.

Pretrained model vs. training from scratch

This project does not train EfficientNetV2-S from scratch.
Instead, it uses the pretrained network as a visual feature extractor.
This significantly reduces training requirements and demonstrates a practical transfer-learning workflow.

Limitations

There are several limitations to the current implementation.

Dataset size
The retrieval database contains 10,000 CIFAR-10 images.
A production visual search system would typically operate over much larger collections.

Search complexity
The current implementation compares the query against all database embeddings.
For a database of 10,000 images this is completely reasonable.
For millions of images, a brute-force matrix comparison would become increasingly expensive.
A production-scale system could use approximate nearest-neighbour methods such as:

FAISS
Annoy
HNSW
vector databases
Domain mismatch
EfficientNetV2-S was pretrained on a large natural-image dataset, while CIFAR-10 contains very small 32 × 32 images.
The resulting embedding space is therefore not specifically optimized for CIFAR-10 retrieval.

Retrieval quality
The current system relies on the representation learned by the pretrained model.
A retrieval model trained specifically with metric-learning objectives could potentially produce a better embedding space.

Future Improvements

Several extensions would make the project more sophisticated.

1. Compare different embedding dimensionalities

Measure retrieval quality for:

1280
512
256
128
64
32
This would allow the trade-off between representation size and retrieval quality to be quantified.

2. Compare pretrained vs. fine-tuned embeddings

A future experiment could compare:

Pretrained EfficientNet
vs. Fine-tuned EfficientNet
to determine whether adapting the CNN specifically to CIFAR-10 improves retrieval quality.

3. Train with metric learning

Instead of optimizing only for classification, the network could be trained using objectives such as:

contrastive loss;
triplet loss;
supervised contrastive loss.
The goal would be to explicitly structure the embedding space so that similar images are closer and dissimilar images are farther apart.

4. Use approximate nearest-neighbour search

For a larger database, the current brute-force search could be replaced by:

FAISS
HNSW
Annoy
Vector database
This would make the system more scalable.

5. Improve the evaluation methodology

A stronger retrieval benchmark could include:

Recall@K
Precision@K
Mean Average Precision (mAP)
class-aware retrieval metrics
retrieval latency
memory usage
This would make the comparison between different embedding spaces more rigorous.

Key Takeaways

This project demonstrates several important concepts in modern machine learning.

1. Neural networks learn representations

A CNN can be used not only for classification but also as a feature extractor.
image → embedding

2. Embeddings turn visual similarity into a mathematical problem

Once images are represented as vectors, similarity can be measured geometrically.
image → vector → distance/similarity

3. PCA provides dimensionality reduction

The original 1280-D representation can be compressed into smaller spaces while retaining much of its structure.

4. The query and database must share the same representation space

The same fitted PCA transformation must be applied to both.
This is essential for mathematically meaningful similarity search.

5. Cosine similarity is a simple but effective retrieval metric

For normalized embeddings, cosine similarity can be efficiently computed using a dot product.

6. Representation quality matters more than the classifier

The central idea of this project is not classification accuracy.
The important question is:

Does the learned representation organize images in a meaningful way?
This is a fundamental concept behind modern computer vision, recommendation systems, semantic search, and multimodal AI.

Connection to NLP

The same representation principle appears in Natural Language Processing.

In computer vision:
image → embedding

In NLP:
word / sentence → embedding

In both cases, the data is transformed into a vector representation that allows relationships to be studied geometrically.

This creates a conceptual bridge between:

Computer Vision

Image embeddings
Vector similarity
and

NLP

Text embeddings
Semantic similarity
The underlying idea is the same:
represent complex objects in a vector space where meaningful relationships can be measured.

Conclusion

This project implements a complete visual search pipeline using a pretrained deep neural network.
The system:

text
extracts deep visual features
        ↓
represents images as embeddings
        ↓
reduces dimensionality with PCA
        ↓
normalizes the representations
        ↓
computes cosine similarity
        ↓
returns the most similar images
The project demonstrates how a pretrained CNN can be repurposed from classification into a content-based retrieval system.
More importantly, it explores the idea that modern neural networks can be understood not only as predictors, but as representation learning systems.

That representation-learning perspective is fundamental to many modern applications, including:

image search;
recommendation systems;
semantic search;
face recognition;
retrieval-augmented systems;
multimodal models;
image-text search;
and modern vector databases.
Author

Olivia

This project was developed as a practical study of:

deep learning;
transfer learning;
representation learning;
image embeddings;
dimensionality reduction;
vector similarity;
information retrieval;
reproducible machine learning;
and scalable search architectures.