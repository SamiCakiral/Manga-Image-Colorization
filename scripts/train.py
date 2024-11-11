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
from src.models.fusion_module import FusionModule
from src.pipeline.training_pipeline import TrainingPipeline
from src.data.dataset_loader import DatasetLoader
from src.data.patch_extractor import PatchExtractor
from src.pipeline.inference_pipeline import InferencePipeline

def parse_args():
    """Parse les arguments en ligne de commande."""
    parser = argparse.ArgumentParser(description='Script d\'entraînement du modèle de colorisation')
    parser.add_argument('--config', type=str, default='configs/default_config.yaml',
                      help='Chemin vers le fichier de configuration')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device à utiliser pour l\'entraînement (cuda/cpu)')
    return parser.parse_args()

def setup_dataloaders(dataset_loader: DatasetLoader, config: Config) -> tuple:
    """Configure les dataloaders pour l'entraînement."""
    train_loader = DataLoader(
        dataset_loader,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=config.training['num_workers'],
        pin_memory=True
    )

    # Créer des dataloaders spécifiques pour les patches primaires et secondaires
    primary_dataset = dataset_loader.get_primary_patches()
    secondary_dataset = dataset_loader.get_secondary_patches()

    primary_loader = DataLoader(
        primary_dataset,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=config.training['num_workers'],
        pin_memory=True
    )

    secondary_loader = DataLoader(
        secondary_dataset,
        batch_size=config.training['batch_size'],
        shuffle=True,
        num_workers=config.training['num_workers'],
        pin_memory=True
    )

    return train_loader, primary_loader, secondary_loader

def save_experiment_config(config: Config, args):
    """Sauvegarde la configuration de l'expérience."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    save_path = root_dir / "configs" / "experiment_configs" / f"run_{timestamp}.yaml"
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

    # Chargement des datasets
    dataset_loader = DatasetLoader(
        data_dir=config.data['root_dir'],
        transform_config=config.data['transforms']
    )

    # Configuration des dataloaders
    train_loader, primary_loader, secondary_loader = setup_dataloaders(dataset_loader, config)

    # Phase 1: Entraînement du modèle d'attention
    print("Phase 1: Entraînement du modèle d'attention")
    training_pipeline.train_attention_model(
        dataloader=train_loader,
        epochs=config.training['attention']['epochs'],
        lr=config.training['attention']['learning_rate']
    )

    # Phase 2: Génération des masques d'attention
    print("Phase 2: Génération des masques d'attention")
    training_pipeline.generate_attention_masks(dataset_loader)

    # Phase 3: Extraction et sauvegarde des patches
    print("Phase 3: Extraction des patches")
    patch_extractor = PatchExtractor(
        patch_size=config.data['patch_size'],
        overlap=config.data['patch_overlap']
    )
    
    patches_save_dir = Path(config.data['patches_dir'])
    patches_save_dir.mkdir(parents=True, exist_ok=True)
    
    training_pipeline.extract_and_save_patches(
        dataset_loader=dataset_loader,
        patch_extractor=patch_extractor,
        save_dir=str(patches_save_dir)
    )

    # Phase 4: Entraînement des modèles de colorisation
    print("Phase 4: Entraînement des modèles de colorisation")
    training_pipeline.train_colorization_models(
        primary_dataloader=primary_loader,
        secondary_dataloader=secondary_loader,
        epochs=config.training['colorization']['epochs'],
        lr=config.training['colorization']['learning_rate']
    )

    print("Entraînement terminé avec succès!")

if __name__ == "__main__":
    main()