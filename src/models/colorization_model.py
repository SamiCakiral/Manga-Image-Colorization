import torch
import torch.nn as nn
from utils.config import config

class ColorizationModel(nn.Module):
    """
    Modèle de colorisation pour les zones importantes et les chutes.
    Peut être utilisé pour le pipeline principal ou secondaire.
    """
    
    def __init__(self, model_type: str = 'primary'):
        """
        Initialise le modèle de colorisation.
    
        Arguments:
        - model_type (str): 'primary' pour le pipeline principal, 'secondary' pour le secondaire.
        """
        super(ColorizationModel, self).__init__()
        self.model_type = model_type

        # Définition des couches du modèle selon le type
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, H/8, W/8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (B, 512, H/16, W/16)
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (B, 256, H/8, W/8)
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (B, 128, H/4, W/4)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (B, 64, H/2, W/2)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # (B, 3, H, W)
            nn.Tanh(),  # Valeurs entre -1 et 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passe avant du modèle.
    
        Arguments:
        - x (torch.Tensor): Entrée du modèle de dimensions (batch_size, 1, H, W).
    
        Retourne:
        - output (torch.Tensor): Image colorisée de dimensions (batch_size, 3, H, W).
        """
        # Encodage
        features = self.encoder(x)
        
        # Décodage
        output = self.decoder(features)
        return output
    
    def save_model(self, save_path: str):
        """
        Sauvegarde le modèle de colorisation actuel dans un fichier.
    
        Arguments:
        - save_path (str): Chemin où sauvegarder le modèle.
        """
        torch.save(self.state_dict(), save_path)
    
    @staticmethod
    def load_model(model_path: str, model_type: str = 'primary') -> nn.Module:
        """
        Charge un modèle de colorisation à partir d'un fichier.
    
        Arguments:
        - model_path (str): Chemin vers le fichier du modèle.
        - model_type (str): 'primary' ou 'secondary' pour initialiser le bon type.
    
        Retourne:
        - model (ColorizationModel): Modèle chargé.
        """
        model = ColorizationModel(model_type=model_type)
        model.load_state_dict(torch.load(model_path))
        return model