import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
from typing import Dict, List, Tuple
from sklearn.model_selection import train_test_split

class DatasetLoader(Dataset):
    """
    Classe pour gérer le chargement des datasets pour l'entraînement et l'inférence.
    """
    def __init__(self, 
                 bw_dir: str, 
                 color_dir: str, 
                 metadata_dir: str, 
                 transform=None, 
                 split: str = 'train',
                 test_size: float = 0.2,
                 random_state: int = 42):
        """
        Initialise le DatasetLoader.

        Arguments:
        - bw_dir (str): Chemin vers les images noir et blanc.
        - color_dir (str): Chemin vers les images colorées correspondantes.
        - metadata_dir (str): Chemin vers les métadonnées des images.
        - transform (callable, optional): Transformations à appliquer aux images.
        - split (str): 'train' ou 'val' pour choisir l'ensemble de données.
        - test_size (float): Proportion de l'ensemble de validation.
        - random_state (int): Seed pour le random split.
        """
        self.bw_dir = bw_dir
        self.color_dir = color_dir
        self.metadata_dir = metadata_dir
        self.transform = transform

        self.image_ids = [os.path.splitext(f)[0] for f in os.listdir(self.bw_dir) if f.endswith('.png')]

        # Division en ensembles d'entraînement et de validation
        train_ids, val_ids = train_test_split(
            self.image_ids, 
            test_size=test_size, 
            random_state=random_state
        )

        if split == 'train':
            self.image_ids = train_ids
        elif split == 'val':
            self.image_ids = val_ids
        else:
            raise ValueError("Le paramètre 'split' doit être 'train' ou 'val'.")

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
        try:
            image_id = self.image_ids[idx]
            bw_image_path = os.path.join(self.bw_dir, f"{image_id}.png")
            color_image_path = os.path.join(self.color_dir, f"{image_id}.png")
            metadata_path = os.path.join(self.metadata_dir, f"{image_id}.json")

            if not all(os.path.exists(p) for p in [bw_image_path, color_image_path, metadata_path]):
                raise FileNotFoundError(f"Fichiers manquants pour l'image {image_id}")

            bw_image = Image.open(bw_image_path).convert('L')
            color_image = Image.open(color_image_path).convert('RGB')

            # Charger les métadonnées si nécessaire
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            if self.transform:
                try:
                    bw_image = self.transform(bw_image)
                    color_image = self.transform(color_image)
                except Exception as e:
                    raise RuntimeError(f"Erreur lors de la transformation: {str(e)}")

            sample = {
                'bw_image': bw_image,
                'color_image': color_image,
                'metadata': metadata
            }

            return sample

        except Exception as e:
            print(f"Erreur lors du chargement de l'image {idx}: {str(e)}")
            raise