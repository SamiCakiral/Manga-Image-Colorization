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
                 config: dict,
                 transform=None, 
                 dataset_type: str = 'training',
                 split: str = 'train'):
        """
        Initialise le DatasetLoader.

        Arguments:
        - config (dict): Configuration contenant les paramètres du dataset
        - transform (callable, optional): Transformations à appliquer aux images
        - dataset_type (str): 'training' ou 'inference' pour choisir le dataset
        - split (str): 'train' ou 'val' pour choisir l'ensemble de données
        """
        base_dir = os.path.join('/content/dataset', dataset_type)
        self.bw_dir = os.path.join(base_dir, 'source/bw')
        self.color_dir = os.path.join(base_dir, 'source/color')
        self.metadata_dir = os.path.join(base_dir, 'metadata')
        self.transform = transform

        self.image_ids = [os.path.splitext(f)[0] for f in os.listdir(self.bw_dir) if f.endswith('.png')]

        # Ne faire le split que pour le dataset d'entraînement
        if dataset_type == 'training':
            test_size = config['data']['validation_split']
            train_ids, val_ids = train_test_split(
                self.image_ids, 
                test_size=test_size, 
                random_state=42
            )
            self.image_ids = train_ids if split == 'train' else val_ids

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