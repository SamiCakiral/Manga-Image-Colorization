class DatasetLoader(torch.utils.data.Dataset):
    """
    Classe pour gérer le chargement des datasets pour l'entraînement et l'inférence.
    """

    def __init__(self, bw_dir: str, color_dir: str, attention_mask_dir: str = None, mode: str = 'full'):
        """
        Initialise le DatasetLoader.

        Arguments:
        - bw_dir (str): Chemin vers les images noir et blanc.
        - color_dir (str): Chemin vers les images colorées correspondantes.
        - attention_mask_dir (str, optional): Chemin vers les masques d'attention (si nécessaire).
        - mode (str): 'full', 'primary', ou 'secondary' selon le type de dataset à charger.
        """
        pass

    def __len__(self) -> int:
        """
        Retourne la taille du dataset.

        Retourne:
        - length (int): Nombre d'échantillons dans le dataset.
        """
        pass

    def __getitem__(self, idx: int) -> Dict:
        """
        Retourne l'échantillon à l'index donné.

        Arguments:
        - idx (int): Index de l'échantillon.

        Retourne:
        - sample (Dict): Dictionnaire contenant l'image noir et blanc, l'image colorée, et éventuellement le masque d'attention.
        """
        pass

    @staticmethod
    def collate_fn(batch: List[Dict]) -> Dict:
        """
        Fonction pour regrouper les échantillons dans un batch.

        Arguments:
        - batch (List[Dict]): Liste d'échantillons.

        Retourne:
        - batch (Dict): Batch regroupé.
        """
        pass