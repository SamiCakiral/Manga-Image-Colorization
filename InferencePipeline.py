from AttentionModel import AttentionPointsModel
from ColorizationModel import ColorizationModel
from FusionModule import FusionModule
import numpy as np
from typing import List, Dict

class InferencePipeline:
    """
    Classe pour gérer le processus d'inférence sur de nouvelles images.
    """

    def __init__(self, attention_model: AttentionPointsModel, primary_model: ColorizationModel, secondary_model: ColorizationModel, fusion_module: FusionModule):
        """
        Initialise le pipeline d'inférence avec les modèles et le module de fusion.

        Arguments:
        - attention_model (AttentionPointsModel): Modèle d'attention chargé.
        - primary_model (ColorizationModel): Modèle de colorisation principal chargé.
        - secondary_model (ColorizationModel): Modèle de colorisation secondaire chargé.
        - fusion_module (FusionModule): Module de fusion initialisé.
        """
        pass

    def colorize_image(self, bw_image: np.ndarray) -> np.ndarray:
        """
        Colorise une image noir et blanc en utilisant le pipeline complet.

        Arguments:
        - bw_image (np.ndarray): Image noir et blanc de dimensions (H, W).

        Retourne:
        - colorized_image (np.ndarray): Image colorisée finale de dimensions (H, W, 3).
        """
        pass