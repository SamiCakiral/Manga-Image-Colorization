import numpy as np
from typing import List, Dict
from ..utils.config import config

class ChuteProcessor:
    """
    Classe pour gérer les chutes (zones simples) lors de l'extraction des patches.
    """

    def __init__(self):
        """
        Initialise le ChuteProcessor.
        """
        pass

    def process_chutes(self, patches: List[Dict]) -> List[Dict]:
        """
        Traite les patches des chutes pour une éventuelle optimisation.

        Arguments:
        - patches (List[Dict]): Liste des patches extraits, incluant les chutes.

        Retourne:
        - chutes (List[Dict]): Liste des chutes traitées avec leurs métadonnées.
        """
        chutes = []
        for patch_info in patches:
            if patch_info['type'] == 'background':
                # Traitement spécifique des chutes si nécessaire
                # Par exemple, on pourrait réduire la résolution pour économiser de la mémoire
                # Ici, nous laissons le patch tel quel
                chutes.append(patch_info)
        return chutes

    @staticmethod
    def save_chutes(chutes: List[Dict], save_dir: str):
        """
        Sauvegarde les patches de chutes dans le répertoire spécifié.

        Arguments:
        - chutes (List[Dict]): Liste des chutes à sauvegarder.
        - save_dir (str): Répertoire de sauvegarde.
        """
        import os
        from PIL import Image

        os.makedirs(save_dir, exist_ok=True)

        for idx, chute_info in enumerate(chutes):
            patch = chute_info['patch']
            x, y, w, h = chute_info['coordinates']

            # Créer le nom de fichier
            filename = f"chute_{idx}_{x}_{y}.png"

            # Sauvegarder le patch
            patch_image = Image.fromarray(patch)
            patch_image.save(os.path.join(save_dir, filename))