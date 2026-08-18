from pathlib import Path 

# Project directories 
PROJECT_ROOT = Path(__file__).resolve().parent.parent 
DATA_DIR = PROJECT_ROOT / "data" 
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings" 
RESULTS_DIR = PROJECT_ROOT / "results" 
RETRIEVAL_RESULTS_DIR = RESULTS_DIR / "retrieval_examples" 

# Dataset 
NUM_CLASSES = 10 
CLASS_NAMES = [ "airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck", ] 

# Model 
MODEL_NAME = "efficientnet_v2_s" 

# Number of images used to build the retrieval database. 
# Set to None to use the complete dataset. 
MAX_DATABASE_IMAGES = None 

# Number of nearest neighbours to retrieve. 
DEFAULT_TOP_K = 5 

# PCA 
PCA_COMPONENTS = [512, 256, 128, 64, 32, 2] 

# Reproducibility 
RANDOM_SEED = 42 

# Create required directories 
DATA_DIR.mkdir(parents=True, exist_ok=True) 
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True) 
RESULTS_DIR.mkdir(parents=True, exist_ok=True) 
RETRIEVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)