class ChuteProcessor:
    """
    Classe pour gérer les chutes (zones simples) lors de l'extraction des patches.
    """

    def __init__(self, patch_size: int = 256):
        """
        Initialise le ChuteProcessor.

        Arguments:
        - patch_size (int): Taille des patches carrés (par défaut 256).
        """
        pass

    def extract_chutes(self, image: np.ndarray, attention_mask: np.ndarray) -> List[Dict]:
        """
        Extrait les patches des chutes à partir de l'image et du masque d'attention.

        Arguments:
        - image (np.ndarray): Image source (noir et blanc ou couleur) de dimensions (H, W, C).
        - attention_mask (np.ndarray): Masque d'attention de dimensions (H, W).

        Retourne:
        - chutes (List[Dict]): Liste de dictionnaires contenant les patches de chutes et leurs métadonnées.
        """
        pass

    @staticmethod
    def save_chutes(chutes: List[Dict], save_dir: str):
        """
        Sauvegarde les patches de chutes dans le répertoire spécifié.

        Arguments:
        - chutes (List[Dict]): Liste des chutes à sauvegarder.
        - save_dir (str): Répertoire de sauvegarde.
        """
        pass