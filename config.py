from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    # Vision Transformer configs
    vit_model_name: str = 'vit_b_16'
    vit_pretrained: bool = True
    vit_feature_size: int = 768
    
    # Attention model configs
    max_attention_points: int = 32
    attention_head_hidden: int = 512
    gaussian_sigma: float = 128/3  # Pour la génération des cartes d'attention

    # Patch configs
    patch_size: int = 256
    patch_overlap: int = 32

    # Training configs
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 50
    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Dataset configs
    dataset_dir: str = '/content/dataset/'
    bw_dir: str = f'{dataset_dir}/source/bw'
    color_dir: str = f'{dataset_dir}/source/color'
    metadata_dir: str = f'{dataset_dir}/metadata'
    attention_maps_dir: str = f'{dataset_dir}/attention_maps'
    patches_dir: str = f'{dataset_dir}/patches'

    # Model saving/loading
    models_dir: str = 'models'
    attention_model_path: str = f'{models_dir}/attention_model.pth'
    primary_model_path: str = f'{models_dir}/primary_model.pth'
    secondary_model_path: str = f'{models_dir}/secondary_model.pth'

    # Complexity map weights
    structural_weight: float = 0.7
    transition_weight: float = 0.3

    # Image processing
    target_size: tuple = (1024, 1024)
    quality_threshold: int = 50  # KB

    def __post_init__(self):
        """Crée les répertoires nécessaires"""
        import os
        for path in [self.dataset_dir, self.bw_dir, self.color_dir, 
                    self.metadata_dir, self.attention_maps_dir, 
                    self.patches_dir, self.models_dir]:
            os.makedirs(path, exist_ok=True)

# Configuration globale
config = ModelConfig()
