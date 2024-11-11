import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional
import logging

from ..utils.config import Config
from ..utils.chute_processor import ChuteProcessor
from ..models.attention_model import AttentionPointsModel
from ..models.colorization_model import ColorizationModel
from ..data.dataset_loader import DatasetLoader
from ..data.patch_extractor import PatchExtractor
from ..metrics.color_metrics import ColorMetrics
from PIL import Image
# Configurer le logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrainingPipeline:
    """
    Pipeline complet pour l'entraînement des modèles d'attention et de colorisation.
    Gère l'entraînement, la validation, la génération des masques d'attention,
    l'extraction des patches, et la sauvegarde des modèles.
    """

    def __init__(self,
                 config: Config,
                 attention_model: AttentionPointsModel,
                 primary_model: ColorizationModel,
                 secondary_model: ColorizationModel):
        """
        Initialise le pipeline d'entraînement.

        Arguments:
            config (Config): Instance de la configuration globale.
            attention_model (AttentionPointsModel): Modèle d'attention initialisé.
            primary_model (ColorizationModel): Modèle de colorisation principal initialisé.
            secondary_model (ColorizationModel): Modèle de colorisation secondaire initialisé.
        """
        self.config = config
        self.device = torch.device(self.config.training['device'])

        # Modèles
        self.attention_model = attention_model.to(self.device)
        self.primary_model = primary_model.to(self.device)
        self.secondary_model = secondary_model.to(self.device)

        # Métriques
        self.metrics = ColorMetrics()

        # Historique de l'entraînement
        self.history = {
            'attention': {'train_loss': [], 'val_loss': []},
            'primary': {'train_loss': [], 'val_loss': [], 'val_quality': []},
            'secondary': {'train_loss': [], 'val_loss': [], 'val_quality': []}
        }

        # Meilleures performances
        self.best_val_loss = float('inf')
        self.best_val_quality_primary = 0.0
        self.best_val_quality_secondary = 0.0

    def train_attention_model(self,
                              train_loader: DataLoader,
                              val_loader: DataLoader,
                              epochs: int,
                              lr: float):
        """
        Entraîne le modèle d'attention avec validation.

        Arguments:
            train_loader (DataLoader): DataLoader pour l'entraînement.
            val_loader (DataLoader): DataLoader pour la validation.
            epochs (int): Nombre d'époques d'entraînement.
            lr (float): Taux d'apprentissage.
        """
        optimizer = torch.optim.Adam(self.attention_model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        for epoch in range(epochs):
            logger.info(f"Époque {epoch+1}/{epochs} - Entraînement du modèle d'attention")
            # Phase d'entraînement
            train_loss = self._train_attention_epoch(train_loader, optimizer, criterion)
            self.history['attention']['train_loss'].append(train_loss)

            # Phase de validation
            val_loss = self._validate_attention_model(val_loader, criterion)
            self.history['attention']['val_loss'].append(val_loss)

            logger.info(f"Perte entraînement: {train_loss:.4f}, Perte validation: {val_loss:.4f}")

            # Sauvegarde du meilleur modèle
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_model(self.attention_model, 'attention_model_best.pth')
                logger.info("Nouveau meilleur modèle d'attention sauvegardé.")

    def _train_attention_epoch(self, dataloader: DataLoader, optimizer, criterion) -> float:
        """Entraîne le modèle d'attention pour une époque."""
        self.attention_model.train()
        running_loss = 0.0

        for batch in tqdm(dataloader, desc="Entraînement modèle d'attention"):
            images = batch['bw_image'].to(self.device)
            optimizer.zero_grad()

            # Calculer la carte de complexité comme pseudo vérité terrain
            with torch.no_grad():
                complexity_maps = self.attention_model.compute_complexity_maps(images)
                complexity_maps = complexity_maps.to(self.device)

            # Prédire les points d'attention
            points, scores = self.attention_model(images)
            pred_attention = self.attention_model.points_to_attention_map(points, scores, images.shape[2:])

            # Calculer la perte
            loss = criterion(pred_attention, complexity_maps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def _validate_attention_model(self, dataloader: DataLoader, criterion) -> float:
        """Valide le modèle d'attention."""
        self.attention_model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation modèle d'attention"):
                images = batch['bw_image'].to(self.device)
                complexity_maps = self.attention_model.compute_complexity_maps(images).to(self.device)

                # Prédire les points d'attention
                points, scores = self.attention_model(images)
                pred_attention = self.attention_model.points_to_attention_map(points, scores, images.shape[2:])

                # Calculer la perte
                loss = criterion(pred_attention, complexity_maps)
                running_loss += loss.item()

        return running_loss / len(dataloader)

    def train_colorization_models(self,
                                  primary_train_loader: DataLoader,
                                  primary_val_loader: DataLoader,
                                  secondary_train_loader: DataLoader,
                                  secondary_val_loader: DataLoader,
                                  epochs: int,
                                  lr: float):
        """
        Entraîne les modèles de colorisation principal et secondaire avec validation.

        Arguments:
            primary_train_loader (DataLoader): DataLoader pour le modèle principal (entraînement).
            primary_val_loader (DataLoader): DataLoader pour le modèle principal (validation).
            secondary_train_loader (DataLoader): DataLoader pour le modèle secondaire (entraînement).
            secondary_val_loader (DataLoader): DataLoader pour le modèle secondaire (validation).
            epochs (int): Nombre d'époques d'entraînement.
            lr (float): Taux d'apprentissage.
        """
        # Entraînement du modèle principal
        self._train_colorization_model(
            model=self.primary_model,
            train_loader=primary_train_loader,
            val_loader=primary_val_loader,
            epochs=epochs,
            lr=lr,
            model_name='primary'
        )

        # Entraînement du modèle secondaire
        self._train_colorization_model(
            model=self.secondary_model,
            train_loader=secondary_train_loader,
            val_loader=secondary_val_loader,
            epochs=epochs,
            lr=lr,
            model_name='secondary'
        )

    def _train_colorization_model(self,
                                  model: ColorizationModel,
                                  train_loader: DataLoader,
                                  val_loader: DataLoader,
                                  epochs: int,
                                  lr: float,
                                  model_name: str):
        """Entraîne un modèle de colorisation (principal ou secondaire)."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        best_val_quality = 0.0

        for epoch in range(epochs):
            logger.info(f"Époque {epoch+1}/{epochs} - Entraînement du modèle {model_name}")
            # Phase d'entraînement
            train_loss = self._train_colorization_epoch(model, train_loader, optimizer, criterion)
            self.history[model_name]['train_loss'].append(train_loss)

            # Phase de validation
            val_loss, val_quality = self._validate_colorization_model(model, val_loader, criterion)
            self.history[model_name]['val_loss'].append(val_loss)
            self.history[model_name]['val_quality'].append(val_quality)

            logger.info(f"Perte entraînement: {train_loss:.4f}, Perte validation: {val_loss:.4f}, Qualité validation: {val_quality:.4f}")

            # Sauvegarde du meilleur modèle
            if val_quality > best_val_quality:
                best_val_quality = val_quality
                self._save_model(model, f'{model_name}_model_best.pth')
                logger.info(f"Nouveau meilleur modèle {model_name} sauvegardé.")

    def _train_colorization_epoch(self, model: ColorizationModel, dataloader: DataLoader, optimizer, criterion) -> float:
        """Entraîne un modèle de colorisation pour une époque."""
        model.train()
        running_loss = 0.0

        for batch in tqdm(dataloader, desc="Entraînement modèle de colorisation"):
            inputs = batch['bw_image'].to(self.device)
            targets = batch['color_image'].to(self.device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        return running_loss / len(dataloader)

    def _validate_colorization_model(self, model: ColorizationModel, dataloader: DataLoader, criterion) -> (float, float):
        """Valide un modèle de colorisation."""
        model.eval()
        running_loss = 0.0
        total_quality = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validation modèle de colorisation"):
                inputs = batch['bw_image'].to(self.device)
                targets = batch['color_image'].to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                running_loss += loss.item()

                # Calculer les métriques de qualité
                outputs_np = outputs.cpu().numpy().transpose(0, 2, 3, 1)
                targets_np = targets.cpu().numpy().transpose(0, 2, 3, 1)
                for pred, target in zip(outputs_np, targets_np):
                    metrics = self.metrics.compute_all_metrics(pred, target)
                    total_quality += metrics['scores']['global_quality']

                num_batches += 1

        avg_loss = running_loss / num_batches
        avg_quality = total_quality / (num_batches * dataloader.batch_size)
        return avg_loss, avg_quality

    def generate_attention_masks(self, dataset_loader: DatasetLoader):
        """
        Génère les masques d'attention pour tout le dataset.

        Arguments:
            dataset_loader (DatasetLoader): Loader pour le dataset complet.
        """
        self.attention_model.eval()
        os.makedirs(self.config.paths['attention_maps_dir'], exist_ok=True)

        with torch.no_grad():
            for idx in tqdm(range(len(dataset_loader)), desc="Génération des masques d'attention"):
                sample = dataset_loader[idx]
                bw_image = sample['bw_image'].unsqueeze(0).to(self.device)
                image_id = sample['metadata']['image_id']

                points, scores = self.attention_model(bw_image)
                attention_map = self.attention_model.points_to_attention_map(points, scores, bw_image.shape[2:])
                attention_map = attention_map.squeeze(0).cpu()

                torch.save(attention_map, os.path.join(self.config.paths['attention_maps_dir'], f"{image_id}_attention_map.pt"))

    def extract_and_save_patches(self, dataset_loader: DatasetLoader, patch_extractor: PatchExtractor):
        """
        Extrait les patches à partir des masques d'attention et les sauvegarde.

        Arguments:
            dataset_loader (DatasetLoader): Loader pour le dataset complet.
            patch_extractor (PatchExtractor): Instance pour extraire les patches.
        """
        os.makedirs(self.config.paths['patches_dir'], exist_ok=True)
        chute_processor = ChuteProcessor()

        for idx in tqdm(range(len(dataset_loader)), desc="Extraction et sauvegarde des patches"):
            sample = dataset_loader[idx]
            bw_image = sample['bw_image'].numpy().transpose(1, 2, 0)
            image_id = sample['metadata']['image_id']

            attention_map_path = os.path.join(self.config.paths['attention_maps_dir'], f"{image_id}_attention_map.pt")
            if not os.path.exists(attention_map_path):
                continue

            attention_map = torch.load(attention_map_path).numpy()
            patches = patch_extractor.extract_patches(bw_image, attention_map)
            chutes = chute_processor.process_chutes(patches)

            self._save_patches(patches, image_id)
            chute_processor.save_chutes(chutes, os.path.join(self.config.paths['patches_dir'], 'chutes'))

    def _save_patches(self, patches: List[Dict], image_id: str):
        """Sauvegarde les patches extraits."""
        for i, patch_info in enumerate(patches):
            patch_type = patch_info['type']
            if patch_type not in ['important', 'background']:
                continue

            save_dir = os.path.join(self.config.paths['patches_dir'], patch_type)
            os.makedirs(save_dir, exist_ok=True)

            patch = patch_info['patch']
            x, y, w, h = patch_info['coordinates']
            filename = f"{image_id}_patch_{i}_{x}_{y}.png"

            patch_image = Image.fromarray(patch.squeeze(), mode='L')
            patch_image.save(os.path.join(save_dir, filename))

    def _save_model(self, model: torch.nn.Module, filename: str):
        """Sauvegarde un modèle au chemin spécifié."""
        save_path = os.path.join(self.config.paths['models_dir'], filename)
        os.makedirs(self.config.paths['models_dir'], exist_ok=True)
        torch.save(model.state_dict(), save_path)