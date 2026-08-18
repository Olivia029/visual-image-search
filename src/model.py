import torch 
import torch.nn as nn 

from torchvision.models import ( 
    EfficientNet_V2_S_Weights, 
    efficientnet_v2_s, 
)

class EfficientNetEmbeddingModel(nn.Module):

    def __init__(self):
        super().__init__()

        weights = EfficientNet_V2_S_Weights.DEFAULT 
        
        model = efficientnet_v2_s(weights=weights) 
        
        # Keep the convolutional feature extractor. 
        self.features = model.features

        # EfficientNetV2-S classifier input dimension
        self.embedding_dim = model.classifier[1].in_features
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
    
    def forward(self, x):
        x = self.features(x) 
        x = self.pool(x) 
        x = torch.flatten(x, 1) 
        return x

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# Load EfficientNetV2-S as an embedding extractor
def load_model():
    device = get_device() 
    model = EfficientNetEmbeddingModel() 
    model = model.to(device) 
    model.eval() 
    return model, device

if __name__ == "__main__":
    model, device = load_model()

    print(f"Using device: {device}")
    print(f"Embedding dimension: {model.embedding_dim}")

    dummy = torch.randn(2, 3, 384, 384).to(device)

    with torch.no_grad():
        embeddings = model(dummy)

    print(f"Embedding shape: {embeddings.shape}")
