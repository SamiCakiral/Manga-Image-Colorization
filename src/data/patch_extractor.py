import numpy as np
from typing import List, Dict
from utils.config import config

class PatchExtractor:
    """
    Classe pour extraire les patches d'attention et les chutes à partir des masques d'attention.
    """

    def __init__(self, patch_size: int = config.patch_size, overlap: int = config.patch_overlap, threshold: float = 0.5):
        """
        Initialise le PatchExtractor avec la taille des patches, le chevauchement et le seuil d'attention.

        Arguments:
        - patch_size (int): Taille des patches carrés.
        - overlap (int): Taille du chevauchement entre les patches.
        - threshold (float): Seuil pour décider si un patch est 'important' ou 'background'.
        """
        self.patch_size = patch_size
        self.overlap = overlap
        self.threshold = threshold

    def extract_patches(self, image: np.ndarray, attention_map: np.ndarray) -> List[Dict]:
        """
        Extrait les patches importants et les chutes à partir de l'image et du masque d'attention.

        Arguments:
        - image (np.ndarray): Image source (H, W, C).
        - attention_map (np.ndarray): Masque d'attention (H, W).

        Retourne:
        - patches (List[Dict]): Liste de dictionnaires contenant les patches et leurs métadonnées.
          Chaque dictionnaire contient:
            - 'patch': le patch extrait (np.ndarray).
            - 'coordinates': (x, y, w, h) position et taille du patch dans l'image originale.
            - 'type': 'important' ou 'background' selon le masque d'attention.
        """
        H, W = attention_map.shape
        patches = []

        step = self.patch_size - self.overlap

        for y in range(0, H - self.patch_size + 1, step):
            for x in range(0, W - self.patch_size + 1, step):
                # Extraire le patch d'attention
                attention_patch = attention_map[y:y + self.patch_size, x:x + self.patch_size]
                # Calculer le score moyen d'attention dans le patch
                mean_attention = np.mean(attention_patch)

                # Déterminer le type du patch
                if mean_attention >= self.threshold:
                    patch_type = 'important'
                else:
                    patch_type = 'background'

                # Extraire le patch de l'image
                image_patch = image[y:y + self.patch_size, x:x + self.patch_size, :]

                patch_info = {
                    'patch': image_patch,
                    'coordinates': (x, y, self.patch_size, self.patch_size),
                    'type': patch_type
                }
                patches.append(patch_info)

        return patches

    @staticmethod
    def save_patches(patches: List[Dict], save_dir: str):
        """
        Sauvegarde les patches dans le répertoire spécifié.

        Arguments:
        - patches (List[Dict]): Liste des patches à sauvegarder.
        - save_dir (str): Répertoire de sauvegarde.
        """
        import os
        from PIL import Image

        os.makedirs(save_dir, exist_ok=True)

        for idx, patch_info in enumerate(patches):
            patch = patch_info['patch']
            patch_type = patch_info['type']
            x, y, w, h = patch_info['coordinates']

            # Créer le nom de fichier
            filename = f"{patch_type}_patch_{idx}_{x}_{y}.png"

            # Sauvegarder le patch
            patch_image = Image.fromarray(patch)
            patch_image.save(os.path.join(save_dir, filename))