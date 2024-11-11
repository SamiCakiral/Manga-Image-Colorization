import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from config import config
from PIL import Image
from ChuteProcessor import ChuteProcessor
from AttentionModel import AttentionPointsModel
from ColorizationModel import ColorizationModel
from DatasetLoader import DatasetLoader
from PatchExtractor import PatchExtractor
from FusionModule import FusionModule


class TrainingPipeline:
    """
    Classe pour gérer l'entraînement des modèles d'attention et de colorisation.
    Elle va entrainer les modèles d'attention et de colorisation séparément.
    """

    def __init__(self, attention_model: AttentionPointsModel, primary_model: ColorizationModel, secondary_model: ColorizationModel, device=config.device):
        """
        Initialise le pipeline d'entraînement avec les modèles fournis.

        Arguments:
        - attention_model (AttentionPointsModel): Modèle d'attention.
        - primary_model (ColorizationModel): Modèle de colorisation principal.
        - secondary_model (ColorizationModel): Modèle de colorisation secondaire.
        - device (torch.device, optional): Appareil sur lequel exécuter les calculs.
        """
        self.attention_model = attention_model
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.device = device

        # Envoyer les modèles sur le device
        self.attention_model.to(self.device)
        self.primary_model.to(self.device)
        self.secondary_model.to(self.device)

    def train_attention_model(self, dataloader: DataLoader, epochs: int = config.num_epochs, lr: float = config.learning_rate):
        """
        Entraîne le modèle d'attention.

        Arguments:
        - dataloader (torch.utils.data.DataLoader): DataLoader pour l'entraînement du modèle d'attention.
        - epochs (int): Nombre d'époques.
        - lr (float): Taux d'apprentissage.
        """
        optimizer = torch.optim.Adam(self.attention_model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.attention_model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in tqdm(dataloader, desc=f"Entraînement AttentionModel Époque {epoch+1}/{epochs}"):
                images = batch['bw_image'].to(self.device)

                optimizer.zero_grad()

                # Calculer la carte de complexité comme pseudo vérité terrain
                with torch.no_grad():
                    complexity_maps = self.attention_model.compute_complexity_maps(images)
                    complexity_maps = complexity_maps.to(self.device)

                # Prédire les points d'attention
                points, scores = self.attention_model(images)

                # Convertir les points en carte d'attention
                pred_attention = self.attention_model.points_to_attention_map(points, scores, images.shape[2:])

                # Calculer la perte
                loss = criterion(pred_attention, complexity_maps)

                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            avg_loss = running_loss / len(dataloader)
            print(f"Époque {epoch+1}/{epochs}, Perte moyenne: {avg_loss:.4f}")

        # Sauvegarder le modèle après l'entraînement
        self.attention_model.save_model(config.attention_model_path)

    def generate_attention_masks(self, dataset_loader: DatasetLoader):
        """
        Génère les masques d'attention pour tout le dataset.

        Arguments:
        - dataset_loader (DatasetLoader): Loader pour le dataset complet.
        """
        self.attention_model.eval()
        os.makedirs(config.attention_maps_dir, exist_ok=True)

        with torch.no_grad():
            for idx in tqdm(range(len(dataset_loader)), desc="Génération des masques d'attention"):
                sample = dataset_loader[idx]
                bw_image = sample['bw_image'].unsqueeze(0).to(self.device)
                image_id = os.path.splitext(sample['metadata']['source_path'])[0]

                # Prédire les points d'attention
                points, scores = self.attention_model(bw_image)

                # Convertir les points en carte d'attention
                attention_map = self.attention_model.points_to_attention_map(points, scores, bw_image.shape[2:])
                attention_map = attention_map.squeeze(0).cpu()

                # Sauvegarder le masque d'attention
                torch.save(attention_map, f'{config.attention_maps_dir}/{image_id}_attention_map.pt')

    def extract_and_save_patches(self, dataset_loader: DatasetLoader, patch_extractor: PatchExtractor, save_dir: str):
        """
        Extrait les patches à partir des masques d'attention et les sauvegarde.

        Arguments:
        - dataset_loader (DatasetLoader): Loader pour le dataset complet.
        - patch_extractor (PatchExtractor): Instance pour extraire les patches.
        - save_dir (str): Répertoire où sauvegarder les patches.
        """
        os.makedirs(save_dir, exist_ok=True)
        primary_save_dir = os.path.join(save_dir, 'primary')
        secondary_save_dir = os.path.join(save_dir, 'secondary')
        chutes_save_dir = os.path.join(save_dir, 'chutes')

        os.makedirs(primary_save_dir, exist_ok=True)
        os.makedirs(secondary_save_dir, exist_ok=True)
        os.makedirs(chutes_save_dir, exist_ok=True)

        chute_processor = ChuteProcessor()

        for idx in tqdm(range(len(dataset_loader)), desc="Extraction et sauvegarde des patches"):
            sample = dataset_loader[idx]
            bw_image = sample['bw_image'].numpy().transpose(1, 2, 0)  # Convertir en HWC
            color_image = sample['color_image'].numpy().transpose(1, 2, 0)
            image_id = f"image_{idx:05d}"

            # Charger le masque d'attention
            attention_map_path = f'{config.attention_maps_dir}/{image_id}_attention_map.pt'
            if not os.path.exists(attention_map_path):
                continue
            attention_map = torch.load(attention_map_path).numpy()

            # Extraire les patches
            patches = patch_extractor.extract_patches(bw_image, attention_map)

            # Traiter les chutes
            chutes = chute_processor.process_chutes(patches)

            # Sauvegarder les patches importants et secondaires
            for i, patch_info in enumerate(patches):
                patch_type = patch_info['type']
                if patch_type == 'important':
                    save_path = primary_save_dir
                elif patch_type == 'background':
                    save_path = secondary_save_dir
                else:
                    continue  # Ignorer les autres types

                patch = patch_info['patch']
                x, y, w, h = patch_info['coordinates']

                # Sauvegarder le patch
                filename = f"{image_id}_patch_{i}_{x}_{y}.png"
                patch_image = Image.fromarray(patch)
                patch_image.save(os.path.join(save_path, filename))

            # Sauvegarder les chutes
            chute_processor.save_chutes(chutes, chutes_save_dir)

    def train_colorization_models(self, primary_dataloader: DataLoader, secondary_dataloader: DataLoader, epochs: int, lr: float):
        """
        Entraîne les modèles de colorisation principal et secondaire.

        Arguments:
        - primary_dataloader (torch.utils.data.DataLoader): DataLoader pour le modèle principal.
        - secondary_dataloader (torch.utils.data.DataLoader): DataLoader pour le modèle secondaire.
        - epochs (int): Nombre d'époques.
        - lr (float): Taux d'apprentissage.
        """
        # Entraînement du modèle principal
        optimizer_primary = torch.optim.Adam(self.primary_model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        self.primary_model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in tqdm(primary_dataloader, desc=f"Entraînement Modèle Principal Époque {epoch+1}/{epochs}"):
                inputs = batch['bw_image'].to(self.device)
                targets = batch['color_image'].to(self.device)
                optimizer_primary.zero_grad()
                outputs = self.primary_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_primary.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(primary_dataloader)
            print(f"Modèle Principal - Époque {epoch+1}/{epochs}, Perte moyenne: {avg_loss:.4f}")
        # Sauvegarder le modèle principal
        self.primary_model.save_model('models/primary_model.pth')

        # Entraînement du modèle secondaire
        optimizer_secondary = torch.optim.Adam(self.secondary_model.parameters(), lr=lr)
        self.secondary_model.train()
        for epoch in range(epochs):
            running_loss = 0.0
            for batch in tqdm(secondary_dataloader, desc=f"Entraînement Modèle Secondaire Époque {epoch+1}/{epochs}"):
                inputs = batch['bw_image'].to(self.device)
                targets = batch['color_image'].to(self.device)
                optimizer_secondary.zero_grad()
                outputs = self.secondary_model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer_secondary.step()
                running_loss += loss.item()
            avg_loss = running_loss / len(secondary_dataloader)
            print(f"Modèle Secondaire - Époque {epoch+1}/{epochs}, Perte moyenne: {avg_loss:.4f}")
        # Sauvegarder le modèle secondaire
        self.secondary_model.save_model('models/secondary_model.pth')