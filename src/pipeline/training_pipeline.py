import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from PIL import Image

# Imports relatifs depuis la nouvelle structure
from ..utils.config import config
from ..utils.chute_processor import ChuteProcessor
from ..models.attention_model import AttentionPointsModel
from ..models.colorization_model import ColorizationModel
from ..data.dataset_loader import DatasetLoader
from ..data.patch_extractor import PatchExtractor

class TrainingPipeline:
    """
    Classe pour gérer l'entraînement des modèles d'attention et de colorisation.
    Elle va entrainer les modèles d'attention et de colorisation séparément.
    """

    def __init__(self, 
                 attention_model: AttentionPointsModel, 
                 primary_model: ColorizationModel, 
                 secondary_model: ColorizationModel, 
                 device=config.training['device']):
        """
        Initialise le pipeline d'entraînement avec les modèles fournis.

        Arguments:
            attention_model (AttentionPointsModel): Modèle d'attention
            primary_model (ColorizationModel): Modèle de colorisation principal
            secondary_model (ColorizationModel): Modèle de colorisation secondaire
            device (torch.device): Appareil sur lequel exécuter les calculs
        """
        self.attention_model = attention_model
        self.primary_model = primary_model
        self.secondary_model = secondary_model
        self.device = device
        self._send_models_to_device()

    def _send_models_to_device(self):
        """Envoie tous les modèles sur le device approprié."""
        self.attention_model.to(self.device)
        self.primary_model.to(self.device)
        self.secondary_model.to(self.device)

    def train_attention_model(self, 
                            dataloader: DataLoader, 
                            epochs: int = config.num_epochs, 
                            lr: float = config.learning_rate):
        """
        Entraîne le modèle d'attention.

        Arguments:
            dataloader (DataLoader): DataLoader pour l'entraînement
            epochs (int): Nombre d'époques d'entraînement
            lr (float): Taux d'apprentissage
        """
        optimizer = torch.optim.Adam(self.attention_model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.attention_model.train()
        for epoch in range(epochs):
            running_loss = self._train_attention_epoch(dataloader, optimizer, criterion, epoch, epochs)
            print(f"Époque {epoch+1}/{epochs}, Perte moyenne: {running_loss:.4f}")

        self.attention_model.save_model(config.attention_model_path)

    def _train_attention_epoch(self, dataloader, optimizer, criterion, epoch, epochs):
        """
        Effectue une époque d'entraînement du modèle d'attention.

        Returns:
            float: Perte moyenne pour l'époque
        """
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Entraînement AttentionModel Époque {epoch+1}/{epochs}"):
            loss = self._train_attention_batch(batch, optimizer, criterion)
            running_loss += loss
        return running_loss / len(dataloader)

    def _train_attention_batch(self, batch, optimizer, criterion):
        """
        Traite un batch pour l'entraînement du modèle d'attention.

        Returns:
            float: Perte pour le batch
        """
        images = batch['bw_image'].to(self.device)
        optimizer.zero_grad()

        with torch.no_grad():
            complexity_maps = self.attention_model.compute_complexity_maps(images)
            complexity_maps = complexity_maps.to(self.device)

        points, scores = self.attention_model(images)
        pred_attention = self.attention_model.points_to_attention_map(points, scores, images.shape[2:])
        
        loss = criterion(pred_attention, complexity_maps)
        loss.backward()
        optimizer.step()

        return loss.item()

    def generate_attention_masks(self, dataset_loader: DatasetLoader):
        """
        Génère les masques d'attention pour tout le dataset.

        Arguments:
            dataset_loader (DatasetLoader): Loader pour le dataset complet
        """
        self.attention_model.eval()
        os.makedirs(config.attention_maps_dir, exist_ok=True)

        with torch.no_grad():
            for idx in tqdm(range(len(dataset_loader)), desc="Génération des masques d'attention"):
                self._generate_single_attention_mask(dataset_loader, idx)

    def _generate_single_attention_mask(self, dataset_loader, idx):
        """
        Génère un masque d'attention pour une seule image.

        Arguments:
            dataset_loader (DatasetLoader): Loader pour le dataset
            idx (int): Index de l'image
        """
        sample = dataset_loader[idx]
        bw_image = sample['bw_image'].unsqueeze(0).to(self.device)
        image_id = os.path.splitext(sample['metadata']['source_path'])[0]

        points, scores = self.attention_model(bw_image)
        attention_map = self.attention_model.points_to_attention_map(points, scores, bw_image.shape[2:])
        attention_map = attention_map.squeeze(0).cpu()

        torch.save(attention_map, f'{config.attention_maps_dir}/{image_id}_attention_map.pt')

    def extract_and_save_patches(self, dataset_loader: DatasetLoader, patch_extractor: PatchExtractor, save_dir: str):
        """
        Extrait les patches à partir des masques d'attention et les sauvegarde.

        Arguments:
            dataset_loader (DatasetLoader): Loader pour le dataset complet
            patch_extractor (PatchExtractor): Instance pour extraire les patches
            save_dir (str): Répertoire où sauvegarder les patches
        """
        self._create_patch_directories(save_dir)
        chute_processor = ChuteProcessor()

        for idx in tqdm(range(len(dataset_loader)), desc="Extraction et sauvegarde des patches"):
            self._process_single_image_patches(
                dataset_loader, patch_extractor, chute_processor, 
                save_dir, idx
            )

    def _create_patch_directories(self, save_dir):
        """Crée les répertoires nécessaires pour sauvegarder les patches."""
        directories = {
            'main': save_dir,
            'primary': os.path.join(save_dir, 'primary'),
            'secondary': os.path.join(save_dir, 'secondary'),
            'chutes': os.path.join(save_dir, 'chutes')
        }
        for directory in directories.values():
            os.makedirs(directory, exist_ok=True)
        return directories

    def _process_single_image_patches(self, dataset_loader, patch_extractor, chute_processor, save_dir, idx):
        """Traite les patches pour une seule image."""
        sample = dataset_loader[idx]
        bw_image = sample['bw_image'].numpy().transpose(1, 2, 0)
        image_id = f"image_{idx:05d}"

        attention_map_path = f'{config.attention_maps_dir}/{image_id}_attention_map.pt'
        if not os.path.exists(attention_map_path):
            return

        attention_map = torch.load(attention_map_path).numpy()
        patches = patch_extractor.extract_patches(bw_image, attention_map)
        chutes = chute_processor.process_chutes(patches)

        self._save_patches(patches, image_id, save_dir)
        chute_processor.save_chutes(chutes, os.path.join(save_dir, 'chutes'))

    def _save_patches(self, patches, image_id, save_dir):
        """Sauvegarde les patches extraits."""
        for i, patch_info in enumerate(patches):
            patch_type = patch_info['type']
            if patch_type not in ['important', 'background']:
                continue

            save_path = os.path.join(save_dir, 
                                   'primary' if patch_type == 'important' else 'secondary')
            
            patch = patch_info['patch']
            x, y, w, h = patch_info['coordinates']
            filename = f"{image_id}_patch_{i}_{x}_{y}.png"
            
            patch_image = Image.fromarray(patch)
            patch_image.save(os.path.join(save_path, filename))

    def train_colorization_models(self, 
                                primary_dataloader: DataLoader, 
                                secondary_dataloader: DataLoader, 
                                epochs: int, 
                                lr: float):
        """
        Entraîne les modèles de colorisation principal et secondaire.

        Arguments:
            primary_dataloader (DataLoader): DataLoader pour le modèle principal
            secondary_dataloader (DataLoader): DataLoader pour le modèle secondaire
            epochs (int): Nombre d'époques d'entraînement
            lr (float): Taux d'apprentissage
        """
        self._train_single_colorization_model(
            self.primary_model, 
            primary_dataloader, 
            epochs, 
            lr, 
            'Principal',
            config.primary_model_path
        )

        self._train_single_colorization_model(
            self.secondary_model, 
            secondary_dataloader, 
            epochs, 
            lr, 
            'Secondaire',
            config.secondary_model_path
        )

    def _train_single_colorization_model(self, model, dataloader, epochs, lr, model_name, save_path):
        """
        Entraîne un seul modèle de colorisation.

        Arguments:
            model (ColorizationModel): Modèle à entraîner
            dataloader (DataLoader): DataLoader pour l'entraînement
            epochs (int): Nombre d'époques
            lr (float): Taux d'apprentissage
            model_name (str): Nom du modèle pour l'affichage
            save_path (str): Chemin pour sauvegarder le modèle
        """
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        model.train()

        for epoch in range(epochs):
            running_loss = self._train_colorization_epoch(
                model, dataloader, optimizer, criterion, epoch, epochs, model_name
            )
            print(f"Modèle {model_name} - Époque {epoch+1}/{epochs}, Perte moyenne: {running_loss:.4f}")

        model.save_model(save_path)

    def _train_colorization_epoch(self, model, dataloader, optimizer, criterion, epoch, epochs, model_name):
        """
        Effectue une époque d'entraînement pour un modèle de colorisation.

        Returns:
            float: Perte moyenne pour l'époque
        """
        running_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Entraînement Modèle {model_name} Époque {epoch+1}/{epochs}"):
            loss = self._train_colorization_batch(model, batch, optimizer, criterion)
            running_loss += loss
        return running_loss / len(dataloader)

    def _train_colorization_batch(self, model, batch, optimizer, criterion):
        """
        Traite un batch pour l'entraînement d'un modèle de colorisation.

        Returns:
            float: Perte pour le batch
        """
        inputs = batch['bw_image'].to(self.device)
        targets = batch['color_image'].to(self.device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        return loss.item()