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
    
    # Configuration minimale pour le test
    config = {
        'data': {
            'training_dataset': {
                'gdrive_url': "https://drive.google.com/file/d/1RWt9kawzXyIvvZWDmEv9e4QrXpqbgac0/view?usp=share_link",
                'target_images': 5000
            },
            'inference_dataset': {
                'gdrive_url': "https://drive.google.com/file/d/1F9YwjozTfhTLugxX-GKQc7hnkhy7JusG/view?usp=share_link",
                'target_images': 100
            },
            'validation_split': 0.2
        }
    }
    
    # Créer et préparer le dataset d'inférence pour le test
    print("\n📥 Test avec le dataset d'inférence (plus petit)...")
    retriever = DatasetRetriever(config=config, dataset_type='inference')
    total_images = retriever.prepare_dataset()
    print(f"\nNombre d'images préparées: {total_images}")
    
    # Test du chargeur de données
    if total_images > 0:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        dataset = DatasetLoader(
            config=config,
            transform=transform,
            dataset_type='inference'
        )
        
        print(f"\nTaille du dataset: {len(dataset)}")
        
        # Test de chargement d'une image
        if len(dataset) > 0:
            sample = dataset[0]
            print("\nDimensions des images:")
            print(f"- N&B: {sample['bw_image'].shape}")
            print(f"- Couleur: {sample['color_image'].shape}")
            print("\nMétadonnées de l'image:")
            print(sample['metadata'])

if __name__ == "__main__":
    test_pipeline() 