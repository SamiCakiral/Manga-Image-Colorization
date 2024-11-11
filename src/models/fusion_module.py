import numpy as np
from typing import Tuple, List, Dict
from PIL import Image
from utils.config import config

class FusionModule:
    """
    Classe pour fusionner les résultats des deux pipelines de colorisation.
    """
    
    def __init__(self):
        """
        Initialise le module de fusion.
        """
        pass
    
    def fuse(self, original_bw: np.ndarray, primary_outputs: List[Dict], secondary_outputs: List[Dict], image_size: Tuple[int, int]) -> np.ndarray:
        """
        Fusionne les sorties des modèles de colorisation pour reconstruire l'image complète.
    
        Arguments:
        - original_bw (np.ndarray): Image originale en noir et blanc de dimensions (H, W).
        - primary_outputs (List[Dict]): Liste des patches colorisés par le modèle principal.
        - secondary_outputs (List[Dict]): Liste des patches colorisés par le modèle secondaire.
        - image_size (Tuple[int, int]): Taille de l'image originale (H, W).
    
        Retourne:
        - fused_image (np.ndarray): Image colorisée finale de dimensions (H, W, 3).
        """
        H, W = image_size
        fused_image = np.zeros((H, W, 3), dtype=np.float32)
        weight_map = np.zeros((H, W), dtype=np.float32)
        
        # Fusion des patches du modèle principal
        for output in primary_outputs:
            patch = output['patch']
            x, y, w, h = output['coordinates']
            weight = self._create_weight_mask((h, w))
            
            fused_image[y:y+h, x:x+w, :] += patch * weight[..., None]
            weight_map[y:y+h, x:x+w] += weight
        
        # Fusion des patches du modèle secondaire
        for output in secondary_outputs:
            patch = output['patch']
            x, y, w, h = output['coordinates']
            weight = self._create_weight_mask((h, w))
            
            fused_image[y:y+h, x:x+w, :] += patch * weight[..., None]
            weight_map[y:y+h, x:x+w] += weight
        
        # Éviter la division par zéro
        weight_map = np.clip(weight_map, a_min=1e-5, a_max=None)
        
        fused_image = fused_image / weight_map[..., None]
        
        # Gérer les valeurs en dehors de [0, 1]
        fused_image = np.clip(fused_image, 0, 1)
        
        return fused_image
    
    def _create_weight_mask(self, size: Tuple[int, int]) -> np.ndarray:
        """
        Crée un masque de poids pour une fusion douce des patches.
    
        Arguments:
        - size (Tuple[int, int]): Taille du patch (h, w).
    
        Retourne:
        - weight_mask (np.ndarray): Masque de poids de dimensions (h, w).
        """
        h, w = size
        y, x = np.mgrid[0:h, 0:w]
        center_y, center_x = h / 2, w / 2
        sigma_y, sigma_x = h / 6, w / 6  # Les valeurs de sigma contrôlent la douceur
        
        weight = np.exp(-(((x - center_x) ** 2) / (2 * sigma_x ** 2) + ((y - center_y) ** 2) / (2 * sigma_y ** 2)))
        weight = weight / weight.max()
        return weight
    
    @staticmethod
    def blend_patches(base_image: np.ndarray, patch: np.ndarray, x: int, y: int, weight_mask: np.ndarray) -> np.ndarray:
        """
        Intègre un patch colorisé dans l'image de base en gérant les chevauchements.
    
        Arguments:
        - base_image (np.ndarray): Image de base où insérer le patch.
        - patch (np.ndarray): Patch colorisé à insérer.
        - x (int): Coordonnée x du coin supérieur gauche du patch.
        - y (int): Coordonnée y du coin supérieur gauche du patch.
        - weight_mask (np.ndarray): Masque de poids pour le blending.
    
        Retourne:
        - base_image (np.ndarray): Image mise à jour avec le patch inséré.
        """
        h, w, _ = patch.shape
        base_image[y:y+h, x:x+w, :] = base_image[y:y+h, x:x+w, :] * (1 - weight_mask[..., None]) + patch * weight_mask[..., None]
        return base_image