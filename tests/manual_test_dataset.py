import os
import sys
from PIL import Image
import torch
from torchvision import transforms

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_retriever import DatasetRetriever
from src.data.dataset_loader import DatasetLoader

def test_pipeline():
    print("=== Test de la pipeline de données ===")
    
    # 1. Préparation du dataset
    gdrive_url = "https://drive.google.com/file/d/1F9YwjozTfhTLugxX-GKQc7hnkhy7JusG/view?usp=share_link"
    dataset_dir = "tests/test_data/test_dataset"
    
    # Créer et préparer le dataset
    retriever = DatasetRetriever(gdrive_url=gdrive_url, target_images=10, dataset_dir=dataset_dir)
    total_images = retriever.prepare_dataset()
    print(f"\nNombre d'images préparées: {total_images}")
    
    # 2. Test du chargeur de données
    if total_images > 0:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        dataset = DatasetLoader(
            bw_dir=os.path.join(dataset_dir, "source", "bw"),
            color_dir=os.path.join(dataset_dir, "source", "color"),
            metadata_dir=os.path.join(dataset_dir, "metadata"),
            transform=transform
        )
        
        print(f"\nTaille du dataset: {len(dataset)}")
        
        # Test de chargement d'une image
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nDimensions des images:")
            print(f"- N&B: {sample['bw_image'].shape}")
            print(f"- Couleur: {sample['color_image'].shape}")

if __name__ == "__main__":
    test_pipeline() 