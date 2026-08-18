from pathlib import Path 
import torch 
from torch.utils.data import Dataset 
from torchvision import datasets, transforms 
from src.config import DATA_DIR, RANDOM_SEED

class CIFAR10EmbeddingDataset(Dataset):

    def __init__(self, root: Path = DATA_DIR):
        self.transform = transforms.Compose(
            [ 
                transforms.Resize((384, 384)), 
                transforms.ToTensor(), 
                transforms.Normalize( 
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225), 
                ), 
            ]
        )

        self.dataset = datasets.CIFAR10( 
            root=root, 
            train=False, 
            download=True, 
            transform=self.transform, 
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return { 
            "image": image, 
            "label": label, 
            "index": idx,  # <-- CORREGIDO: cambiado 'index' por 'idx'
        }

    def get_dataloader(self, batch_size=64, num_workers=2): # <-- NOTA: Agregado 'self' que suele hacer falta aquí
        dataset = CIFAR10EmbeddingDataset() 

        generator = torch.Generator() 
        generator.manual_seed(RANDOM_SEED)
        
        return torch.utils.data.DataLoader( 
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            generator=generator,
        )

if __name__ == "__main__":
    dataset = CIFAR10EmbeddingDataset()
    
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of classes: {len(dataset.dataset.classes)}")
    print(f"Classes: {dataset.dataset.classes}")
    
    sample = dataset[0]

    print(f"Image tensor shape: {sample['image'].shape}")
    print(f"Label: {sample['label']}")
