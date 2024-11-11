import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from typing import Dict
class DatasetLoader(Dataset):
    """
    Classe pour gérer le chargement des datasets pour l'entraînement et l'inférence.
    """

    def __init__(self, bw_dir: str, color_dir: str, metadata_dir: str, transform=None):
        """
        Initialise le DatasetLoader.

        Arguments:
        - bw_dir (str): Chemin vers les images noir et blanc.
        - color_dir (str): Chemin vers les images colorées correspondantes.
        - metadata_dir (str): Chemin vers les métadonnées des images.
        - transform (callable, optional): Transformations à appliquer aux images.
        """
        self.bw_dir = bw_dir
        self.color_dir = color_dir
        self.metadata_dir = metadata_dir
        self.transform = transform

        self.image_ids = [os.path.splitext(f)[0] for f in os.listdir(self.bw_dir) if f.endswith('.png')]

    def __len__(self) -> int:
        """
        Retourne la taille du dataset.

        Retourne:
        - length (int): Nombre d'échantillons dans le dataset.
        """
        return len(self.image_ids)

    def __getitem__(self, idx: int) -> Dict:
        """
        Retourne l'échantillon à l'index donné.

        Arguments:
        - idx (int): Index de l'échantillon.

        Retourne:
        - sample (Dict): Dictionnaire contenant l'image noir et blanc, l'image colorée, et les métadonnées.
        """
        image_id = self.image_ids[idx]
        bw_image_path = os.path.join(self.bw_dir, f"{image_id}.png")
        color_image_path = os.path.join(self.color_dir, f"{image_id}.png")
        metadata_path = os.path.join(self.metadata_dir, f"{image_id}.json")

        bw_image = Image.open(bw_image_path).convert('L')
        color_image = Image.open(color_image_path).convert('RGB')

        # Charger les métadonnées si nécessaire
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        if self.transform:
            bw_image = self.transform(bw_image)
            color_image = self.transform(color_image)

        sample = {
            'bw_image': bw_image,
            'color_image': color_image,
            'metadata': metadata
        }

        return sample