class PatchExtractor:
    """
    Classe pour extraire les patches d'attention et les chutes à partir des masques d'attention.
    """

    def __init__(self, patch_size: int = 256, overlap: int = 0):
        """
        Initialise le PatchExtractor avec la taille des patches et le chevauchement.

        Arguments:
        - patch_size (int): Taille des patches carrés (par défaut 256).
        - overlap (int): Taille du chevauchement entre les patches (par défaut 0).
        """
        pass

    def extract_patches(self, image: np.ndarray, attention_mask: np.ndarray) -> List[Dict]:
        """
        Extrait les patches importants et les chutes à partir de l'image et du masque d'attention.

        Arguments:
        - image (np.ndarray): Image source (noir et blanc ou couleur) de dimensions (H, W, C).
        - attention_mask (np.ndarray): Masque d'attention de dimensions (H, W).

        Retourne:
        - patches (List[Dict]): Liste de dictionnaires contenant les patches et leurs métadonnées.
          Chaque dictionnaire contient:
            - 'patch': le patch extrait (np.ndarray).
            - 'coordinates': (x, y) position du patch dans l'image originale.
            - 'type': 'important' ou 'background' selon le masque d'attention.
        """
        pass

    @staticmethod
    def save_patches(patches: List[Dict], save_dir: str):
        """
        Sauvegarde les patches dans le répertoire spécifié.

        Arguments:
        - patches (List[Dict]): Liste des patches à sauvegarder.
        - save_dir (str): Répertoire de sauvegarde.
        """
        pass