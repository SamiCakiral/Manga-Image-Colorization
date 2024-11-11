import os
import sys
from PIL import Image
import torch
from torchvision import transforms

# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset_retriever import DatasetRetriever
from src.data.dataset_loader import DatasetLoader
from src.utils.config import config

def test_pipeline():
    print("=== Test de la pipeline de données ===")
    
    # Configuration minimale pour le test
    test_config = {
        'data': {
            'training_dataset': {
                'gdrive_url': config.config['data']['inference_dataset']['gdrive_url'],  # Notation dictionnaire
                'target_images': config.config['data']['inference_dataset']['target_images']
            },
            'inference_dataset': config.config['data']['inference_dataset'],
            'validation_split': 0.2
        },
        'paths': config.config['paths']
    }
    
    # Créer et préparer le dataset
    print("\n📥 Test de préparation du dataset...")
    retriever = DatasetRetriever(config=test_config, dataset_type='training')  # Utiliser comme dataset d'entraînement
    total_images = retriever.prepare_dataset()
    print(f"\nNombre d'images préparées: {total_images}")
    
    # Test du chargeur de données
    if total_images > 0:
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
        ])
        
        dataset = DatasetLoader(
            config=test_config,
            transform=transform,
            dataset_type='training',  # Utiliser comme dataset d'entraînement
            split='train'  # On peut maintenant utiliser le split train/val
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
            
            # Test du split de validation
            val_dataset = DatasetLoader(
                config=test_config,
                transform=transform,
                dataset_type='training',
                split='val'
            )
            print(f"\nTaille du dataset de validation: {len(val_dataset)}")

if __name__ == "__main__":
    test_pipeline() 