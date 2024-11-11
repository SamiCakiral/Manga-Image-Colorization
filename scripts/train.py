import sys
from pathlib import Path
import argparse
from datetime import datetime
import torch
from torch.utils.data import DataLoader

# Ajouter le chemin racine au PYTHONPATH
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# Imports depuis la nouvelle structure
from src.utils.config import Config
from src.models.attention_model import AttentionPointsModel
from src.models.colorization_model import ColorizationModel
from src.pipeline.training_pipeline import TrainingPipeline
from src.data.dataset_loader import DatasetLoader
from src.data.patch_extractor import PatchExtractor

def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description='Script d\'entraînement du modèle de colorisation')
    parser.add_argument('--config', type=str, default='default',
                      help='Nom de la configuration à utiliser')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device à utiliser pour l\'entraînement (cuda/cpu)')
    return parser.parse_args()

def setup_dataloaders(config: Config) -> tuple:
    """Configure les dataloaders pour l'entraînement et la validation."""
    # Dataset pour le modèle d'attention
    train_dataset = DatasetLoader(
        bw_dir=config.data['bw_dir'],
        color_dir=config.data['color_dir'],
        metadata_dir=config.data['metadata_dir'],
        transform=None,  # Ajouter les transformations nécessaires
        split='train'
    )
    
    val_dataset = DatasetLoader(
        bw_dir=config.data['bw_dir'],
        color_dir=config.data['color_dir'],
        metadata_dir=config.data['metadata_dir'],
        transform=None,  # Mêmes transformations que pour l'entraînement
        split='val'
    )

    # Création des dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=config.training.get('num_workers', 4),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=config.training.get('num_workers', 4),
        pin_memory=True
    )

    # Datasets pour les modèles de colorisation (primary et secondary)
    primary_train_dataset = DatasetLoader(
        bw_dir=config.data['primary_patches_dir'],
        color_dir=config.data['primary_patches_color_dir'],
        metadata_dir=config.data['metadata_dir'],
        transform=None,
        split='train'
    )
    
    primary_val_dataset = DatasetLoader(
        bw_dir=config.data['primary_patches_dir'],
        color_dir=config.data['primary_patches_color_dir'],
        metadata_dir=config.data['metadata_dir'],
        transform=None,
        split='val'
    )

    secondary_train_dataset = DatasetLoader(
        bw_dir=config.data['secondary_patches_dir'],
        color_dir=config.data['secondary_patches_color_dir'],
        metadata_dir=config.data['metadata_dir'],
        transform=None,
        split='train'
    )
    
    secondary_val_dataset = DatasetLoader(
        bw_dir=config.data['secondary_patches_dir'],
        color_dir=config.data['secondary_patches_color_dir'],
        metadata_dir=config.data['metadata_dir'],
        transform=None,
        split='val'
    )

    # Création des dataloaders pour les patches
    primary_train_loader = DataLoader(
        primary_train_dataset,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=config.training.get('num_workers', 4),
        pin_memory=True
    )
    
    primary_val_loader = DataLoader(
        primary_val_dataset,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=config.training.get('num_workers', 4),
        pin_memory=True
    )

    secondary_train_loader = DataLoader(
        secondary_train_dataset,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=config.training.get('num_workers', 4),
        pin_memory=True
    )
    
    secondary_val_loader = DataLoader(
        secondary_val_dataset,
        batch_size=config.training['batch_size'],
        shuffle=False,
        num_workers=config.training.get('num_workers', 4),
        pin_memory=True
    )

    return (train_loader, val_loader, 
            primary_train_loader, primary_val_loader,
            secondary_train_loader, secondary_val_loader)

def save_experiment_config(config: Config, args):
    """Sauvegarde la configuration de l'expérience."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = config.config_dir / "experiment_configs" / f"run_{timestamp}.yaml"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    config.save(str(save_path))

def main():
    """Fonction principale d'entraînement."""
    args = parse_args()
    config = Config(args.config)
    device = torch.device(args.device)

    # Sauvegarder la configuration
    save_experiment_config(config, args)

    # Initialisation des modèles
    attention_model = AttentionPointsModel(
        max_points=config.model['attention']['max_points']
    )
    
    primary_model = ColorizationModel(
        model_type='primary',
        **config.model['primary']
    )
    
    secondary_model = ColorizationModel(
        model_type='secondary',
        **config.model['secondary']
    )

    # Création du pipeline d'entraînement
    training_pipeline = TrainingPipeline(
        attention_model=attention_model,
        primary_model=primary_model,
        secondary_model=secondary_model,
        device=device
    )

    # Configuration des dataloaders
    (train_loader, val_loader,
     primary_train_loader, primary_val_loader,
     secondary_train_loader, secondary_val_loader) = setup_dataloaders(config)

    # Phase 1: Entraînement du modèle d'attention avec validation
    print("Phase 1: Entraînement du modèle d'attention")
    training_pipeline.train_attention_model(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config.training['attention']['epochs'],
        lr=config.training['attention']['learning_rate']
    )

    # Phase 2: Génération des masques d'attention
    print("Phase 2: Génération des masques d'attention")
    training_pipeline.generate_attention_masks(train_loader)
    training_pipeline.generate_attention_masks(val_loader)

    # Phase 3: Extraction et sauvegarde des patches
    print("Phase 3: Extraction des patches")
    patch_extractor = PatchExtractor(
        patch_size=config.model['patch']['size'],
        overlap=config.model['patch']['overlap']
    )
    
    patches_save_dir = Path(config.paths['patches_dir'])
    patches_save_dir.mkdir(parents=True, exist_ok=True)

    # Phase 4: Entraînement des modèles de colorisation avec validation
    print("Phase 4: Entraînement des modèles de colorisation")
    training_pipeline.train_colorization_models(
        primary_train_loader=primary_train_loader,
        primary_val_loader=primary_val_loader,
        secondary_train_loader=secondary_train_loader,
        secondary_val_loader=secondary_val_loader,
        epochs=config.training['colorization']['epochs'],
        lr=config.training['colorization']['learning_rate']
    )

    print("Entraînement terminé avec succès!")

if __name__ == "__main__":
    main()