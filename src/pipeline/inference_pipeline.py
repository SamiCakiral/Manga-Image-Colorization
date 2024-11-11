from models.attention_model import AttentionPointsModel
from models.colorization_model import ColorizationModel
from models.fusion_module import FusionModule
from data.patch_extractor import PatchExtractor
import torch
import numpy as np
from typing import List, Dict
from utils.config import config
class InferencePipeline:
    """
    Classe pour gérer le processus d'inférence sur de nouvelles images.
    """
    
    def __init__(self, attention_model: AttentionPointsModel, primary_model: ColorizationModel, secondary_model: ColorizationModel, fusion_module: FusionModule, device=config.device):
        """
        Initialise le pipeline d'inférence avec les modèles et le module de fusion.
    
        Arguments:
        - attention_model (AttentionPointsModel): Modèle d'attention chargé.
        - primary_model (ColorizationModel): Modèle de colorisation principal chargé.
        - secondary_model (ColorizationModel): Modèle de colorisation secondaire chargé.
        - fusion_module (FusionModule): Module de fusion initialisé.
        - device (torch.device): Appareil sur lequel exécuter les calculs.
        """
        self.attention_model = attention_model.to(device)
        self.primary_model = primary_model.to(device)
        self.secondary_model = secondary_model.to(device)
        self.fusion_module = fusion_module
        self.device = device
        self.patch_extractor = PatchExtractor()
    
    def colorize_image(self, bw_image: np.ndarray) -> np.ndarray:
        """
        Colorise une image noir et blanc en utilisant le pipeline complet.
    
        Arguments:
        - bw_image (np.ndarray): Image noir et blanc de dimensions (H, W).
    
        Retourne:
        - colorized_image (np.ndarray): Image colorisée finale de dimensions (H, W, 3).
        """
        # Validation des entrées
        if bw_image is None:
            raise ValueError("L'image d'entrée ne peut pas être None")
        
        if not isinstance(bw_image, np.ndarray):
            raise TypeError("L'image d'entrée doit être un numpy.ndarray")
        
        if bw_image.ndim != 2:
            raise ValueError("L'image d'entrée doit être en niveaux de gris (2D)")
        
        # Conversion et normalisation
        bw_tensor = torch.from_numpy(bw_image).unsqueeze(0).unsqueeze(0).float()
        bw_tensor = bw_tensor.to(self.device) / 255.0
        
        H, W = bw_image.shape
        
        # Génération du masque d'attention
        self.attention_model.eval()
        with torch.no_grad():
            points, scores = self.attention_model(bw_tensor)
            attention_map = self.attention_model.points_to_attention_map(points, scores, (H, W))
            attention_map = attention_map.squeeze(0).cpu().numpy()
        
        # Extraction des patches
        patches_info = self.patch_extractor.extract_patches(bw_image[..., None], attention_map)
        
        primary_patches = []
        secondary_patches = []
        
        # Colorisation des patches
        for patch_info in patches_info:
            patch = patch_info['patch']  # (h, w, 1)
            patch_tensor = torch.from_numpy(patch.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device) / 255.0  # (1, 1, h, w)
            
            if patch_info['type'] == 'important':
                self.primary_model.eval()
                with torch.no_grad():
                    colorized_patch = self.primary_model(patch_tensor)
            else:
                self.secondary_model.eval()
                with torch.no_grad():
                    colorized_patch = self.secondary_model(patch_tensor)
            
            # Conversion en numpy
            colorized_patch = colorized_patch.squeeze(0).cpu().numpy().transpose(1, 2, 0)  # (h, w, 3)
            colorized_patch = np.clip((colorized_patch + 1) / 2, 0, 1)  # Remettre les valeurs entre 0 et 1
            
            # Ajouter aux listes
            patch_info['patch'] = colorized_patch  # Remplacer le patch N&B par le patch colorisé
            if patch_info['type'] == 'important':
                primary_patches.append(patch_info)
            else:
                secondary_patches.append(patch_info)
        
        # Fusionner les patches pour reconstruire l'image
        colorized_image = self.fusion_module.fuse(
            original_bw=bw_image,
            primary_outputs=primary_patches,
            secondary_outputs=secondary_patches,
            image_size=(H, W)
        )
        
        return colorized_image