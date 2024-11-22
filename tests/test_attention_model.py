import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
# Ajouter le chemin du projet
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import config
from src.data.dataset_loader import DatasetLoader
from src.models.attention_model import AttentionPointsModel

def visualize_attention_map(image, points, scores, save_path=None):
    """
    Visualise les points d'attention et leur score sur l'image.
    
    Args:
        image (torch.Tensor): Image en noir et blanc [1, H, W]
        points (torch.Tensor): Points d'attention [N, 2]
        scores (torch.Tensor): Scores d'importance [N]
        save_path (str, optional): Chemin pour sauvegarder la visualisation
    """
    # Debug: afficher les dimensions
    print(f"Points shape: {points.shape}")
    print(f"Scores shape: {scores.shape}")
    
    plt.figure(figsize=(12, 6))
    
    # Redimensionner l'image à 224x224 pour correspondre au traitement
    image_resized = F.interpolate(image.unsqueeze(0), size=(224, 224), mode='bilinear').squeeze(0)
    
    # Image originale avec points
    plt.subplot(1, 2, 1)
    plt.imshow(image_resized.squeeze().cpu(), cmap='gray')
    
    # Convertir les coordonnées normalisées en pixels
    H, W = 224, 224  # Utiliser la taille fixe de 224x224
    points_pixels = points.clone().detach().cpu().numpy()
    points_pixels[:, 0] *= W
    points_pixels[:, 1] *= H
    
    # Normaliser les scores pour la taille des points
    scores_np = scores.detach().cpu().numpy()
    sizes = 100 + 400 * scores_np
    
    # Scatter plot avec couleur selon le score
    scatter = plt.scatter(points_pixels[:, 0], points_pixels[:, 1], 
                         c=scores_np, s=sizes, alpha=0.6, 
                         cmap='hot', vmin=0, vmax=1)
    plt.colorbar(scatter, label='Score d\'attention')
    
    # Ajouter les valeurs des scores à côté des points
    for i, (x, y, score) in enumerate(zip(points_pixels[:, 0], points_pixels[:, 1], scores_np)):
        if score > 0:  # N'afficher que les scores non nuls
            plt.annotate(f'{score:.2f}', (x, y), xytext=(5, 5), textcoords='offset points')
    
    plt.title("Points d'attention")
    plt.axis('off')
    
    # Carte de chaleur de l'attention
    plt.subplot(1, 2, 2)
    attention_map = AttentionPointsModel.points_to_attention_map(
        points.unsqueeze(0), 
        scores.unsqueeze(0), 
        (H, W)  # Utiliser 224x224
    )
    plt.imshow(attention_map[0, 0].cpu(), cmap='hot')
    plt.title("Carte d'attention")
    plt.axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()

def test_attention_model():
    print("=== Test du modèle d'attention ===")
    
    # Configuration des transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Changé à 224x224 pour ViT
        transforms.ToTensor(),
    ])
    
    # Chargement direct du dataset de training existant
    dataset = DatasetLoader(
        config=config.config,
        transform=transform,
        dataset_type='training',
        split='val'
    )
    
    if len(dataset) == 0:
        print("❌ Erreur: Le dataset est vide. Vérifiez le dossier data/dataset/training")
        return
        
    print(f"\n✅ Dataset chargé avec {len(dataset)} images")
    
    # Création du DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=2
    )
    
    # Initialisation du modèle d'attention
    device = torch.device(config.config['training']['device'])
    attention_model = AttentionPointsModel(
        max_points=config.config['model']['attention']['max_points']
    ).to(device)
    attention_model.eval()
    
    # Création du dossier pour les visualisations
    viz_dir = os.path.join(config.config['paths']['attention_maps_dir'])
    os.makedirs(viz_dir, exist_ok=True)
    
    print("\n🔍 Test de la génération des points d'attention...")
    
    # Test sur quelques images
    num_test_images = min(5, len(dataset))
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=num_test_images)):
            if i >= num_test_images:
                break
                
            # Préparation des données
            bw_image = batch['bw_image'].to(device)
            
            # Génération des points d'attention
            points, scores = attention_model(bw_image)
            
            # Redimensionner les points pour correspondre à la taille originale
            if config.config['model']['patch']['size'] != 224:
                points[:, :, 0] *= config.config['model']['patch']['size'] / 224
                points[:, :, 1] *= config.config['model']['patch']['size'] / 224
            
            # Visualisation et sauvegarde
            save_path = os.path.join(viz_dir, f'attention_map_{i}.png')
            visualize_attention_map(
                F.interpolate(bw_image, size=(config.config['model']['patch']['size'], 
                                            config.config['model']['patch']['size']), 
                            mode='bilinear')[0].cpu(),
                points[0],
                scores[0],
                save_path=save_path
            )
            
            # Test de l'extraction des patches
            patches, coords = attention_model.get_attention_patches(
                F.interpolate(bw_image, size=(config.config['model']['patch']['size'], 
                                            config.config['model']['patch']['size']), 
                            mode='bilinear'),
                points,
                scores,
                patch_size=config.config['model']['patch']['size']
            )
            
            print(f"\n✅ Carte d'attention {i+1} générée:")
            print(f"  - Sauvegardée dans: {save_path}")
            print(f"  - Nombre de patches: {len(patches[0])}")
            print(f"  - Taille des patches: {patches[0][0].shape}")
    
    print("\n✨ Test terminé!")
    print(f"Les visualisations ont été sauvegardées dans: {viz_dir}")

if __name__ == "__main__":
    test_attention_model()