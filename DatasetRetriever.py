import os
import zipfile
import gdown
import json
import numpy as np
from PIL import Image
import io
from tqdm import tqdm
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from multiprocessing import Pool, cpu_count, Manager, Lock

class DatasetRetriever:
    """
    Classe pour télécharger, extraire et préparer le dataset à partir d'une source donnée.
    """

    def __init__(self, gdrive_url: str, target_images: int, dataset_dir: str = '/content/dataset/'):
        """
        Initialise le DatasetRetriever avec l'URL de Google Drive et le nombre d'images cibles.

        Arguments:
        - gdrive_url (str): URL de téléchargement du fichier zip contenant les données.
        - target_images (int): Nombre d'images à traiter pour le dataset.
        - dataset_dir (str): Chemin vers le répertoire où le dataset sera stocké.
        """
        self.gdrive_url = gdrive_url
        self.target_images = target_images
        self.dataset_dir = dataset_dir

        # Chemins et structures de répertoires
        self.master_zip_path = os.path.join(self.dataset_dir, 'master.zip')
        self.cbz_extract_path = os.path.join(self.dataset_dir, 'cbz_files')
        self.paths = {
            'bw': os.path.join(self.dataset_dir, 'source', 'bw'),
            'color': os.path.join(self.dataset_dir, 'source', 'color'),
            'metadata': os.path.join(self.dataset_dir, 'metadata'),
            'temp': os.path.join(self.dataset_dir, 'temp_cbz')
        }

        # Création des répertoires nécessaires
        for path in self.paths.values():
            os.makedirs(path, exist_ok=True)

        # Variables pour le traitement
        self.image_counter = 0
        self.counter_lock = threading.Lock()
        self.stop_processing = False
        self.num_workers = multiprocessing.cpu_count()
        self.manager = Manager()
        self.total_images_processed = self.manager.Value('i', 0)
        self.total_lock = self.manager.Lock()
        self.pbar = tqdm(total=target_images, desc="Images traitées")

        # Configuration du traitement des images
        self.target_size = (1024, 1024)
        self.quality_threshold = 50  # Seuil minimal de qualité (KB)
        self.num_processes = cpu_count()  # Nombre de processus pour l'extraction

    def download_and_extract(self) -> bool:
        """
        Télécharge et extrait le fichier zip contenant les données.

        Retourne:
        - success (bool): True si le téléchargement et l'extraction ont réussi, False sinon.
        """
        # Vérification si le fichier existe déjà
        if os.path.exists(self.master_zip_path):
            print("📦 Le fichier master.zip existe déjà")
            print(f"Taille du fichier: {os.path.getsize(self.master_zip_path) / 1024 / 1024:.2f} MB")
        else:
            print("📥 Téléchargement du master zip...")
            try:
                gdown.download(url=self.gdrive_url, output=self.master_zip_path, fuzzy=True)
            except Exception as e:
                print(f"❌ Erreur lors du téléchargement: {str(e)}")
                return False

        if not os.path.exists(self.master_zip_path):
            print("❌ Échec du téléchargement")
            return False

        print(f"✅ Téléchargement terminé: {os.path.getsize(self.master_zip_path) / 1024 / 1024:.2f} MB")

        print("\n📦 Extraction des fichiers CBZ...")
        os.makedirs(self.cbz_extract_path, exist_ok=True)

        try:
            with zipfile.ZipFile(self.master_zip_path, 'r') as zip_ref:
                zip_ref.extractall(self.cbz_extract_path)
            print("✅ Extraction terminée")
            os.remove(self.master_zip_path)
            return True

        except Exception as e:
            print(f"❌ Erreur lors de l'extraction: {str(e)}")
            return False

    def validate_image(self, img: Image.Image) -> Tuple[bool, str]:
        """
        Valide la qualité et les caractéristiques de l'image.

        Arguments:
        - img (Image.Image): Image PIL à valider.

        Retourne:
        - is_valid (bool): True si l'image est valide, False sinon.
        - message (str): Message décrivant le résultat de la validation.
        """
        if img.size[0] < 300 or img.size[1] < 300:
            return False, "Image trop petite"

        # Vérifier le contraste et la netteté
        img_array = np.array(img)
        if img_array.std() < 20:  # Vérification basique du contraste
            return False, "Contraste insuffisant"

        return True, "OK"

    def process_single_image(self, args: Tuple[str, str]):
        """
        Traite une seule image en effectuant la validation et la conversion.

        Arguments:
        - args (Tuple[str, str]): Tuple contenant le chemin de l'image et le répertoire temporaire.
        """
        if self.stop_processing:
            return

        image_path, temp_dir = args
        try:
            img = Image.open(image_path)

            # Validation de l'image
            is_valid, message = self.validate_image(img)
            if not is_valid:
                print(f"Image ignorée ({image_path}): {message}")
                return

            with self.counter_lock:
                if self.image_counter >= self.target_images:
                    self.stop_processing = True
                    return
                image_id = f"image_{self.image_counter:05d}"
                self.image_counter += 1
                with self.total_lock:
                    self.total_images_processed.value += 1
                    self.pbar.update(1)

            # Métadonnées de l'image
            metadata = {
                'original_size': img.size,
                'original_mode': img.mode,
                'source_path': os.path.relpath(image_path, temp_dir),
                'chapter': os.path.basename(os.path.dirname(image_path)),
                'creation_date': os.path.getctime(image_path),
                'validation_message': message
            }

            # Sauvegarder l'image couleur
            color_path = os.path.join(self.paths['color'], f"{image_id}.png")
            img.save(color_path, 'PNG')

            # Convertir et sauvegarder en noir et blanc
            img_gray = img.convert('L')
            gray_path = os.path.join(self.paths['bw'], f"{image_id}.png")
            img_gray.save(gray_path, 'PNG')

            # Sauvegarder les métadonnées
            metadata_path = os.path.join(self.paths['metadata'], f"{image_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)

        except Exception as e:
            print(f"Erreur lors du traitement de l'image {image_path}: {str(e)}")

    def get_filtered_images(self, directory: str) -> List[Tuple[str, str]]:
        """
        Récupère les images filtrées en excluant les pages de couverture et de fin.

        Arguments:
        - directory (str): Répertoire contenant les images extraites.

        Retourne:
        - filtered_images (List[Tuple[str, str]]): Liste de tuples avec le chemin de l'image et le répertoire temporaire.
        """
        chapters = {}
        skip_start = 6
        skip_end = 6

        for root, _, files in os.walk(directory):
            chapter_name = os.path.basename(root)
            if files:
                image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                if image_files:
                    image_files.sort(key=lambda x: int(''.join(filter(str.isdigit, x)) or 0))
                    chapters[chapter_name] = [os.path.join(root, f) for f in image_files]

        filtered_images = []
        chapter_names = sorted(chapters.keys())

        for i, chapter in enumerate(chapter_names):
            images = chapters[chapter]
            if i == 0 or i == len(chapter_names) - 1:
                images = images[skip_start:-skip_end]
            filtered_images.extend((img, directory) for img in images)

        return filtered_images

    def extract_cbz_wrapper(self, cbz_file: str) -> Optional[str]:
        """
        Wrapper pour l'extraction des fichiers CBZ, utilisé pour le multiprocessing.

        Arguments:
        - cbz_file (str): Chemin vers le fichier CBZ à extraire.

        Retourne:
        - temp_dir (Optional[str]): Chemin vers le répertoire temporaire où les fichiers sont extraits.
        """
        if self.stop_processing:
            return None

        temp_dir = os.path.join(self.paths['temp'], os.path.splitext(os.path.basename(cbz_file))[0])
        os.makedirs(temp_dir, exist_ok=True)

        try:
            print(f"📚 Extraction de {os.path.basename(cbz_file)}...")
            with zipfile.ZipFile(cbz_file, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            return temp_dir
        except Exception as e:
            print(f"❌ Erreur lors de l'extraction de {cbz_file}: {str(e)}")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return None

    def create_dataset(self) -> int:
        """
        Crée le dataset en téléchargeant, extrayant et traitant les images.

        Retourne:
        - total_images_processed (int): Nombre total d'images traitées.
        """
        print(f"\nCréation d'un dataset avec {self.target_images} images...")
        print(f"Utilisation de {self.num_processes} processus pour l'extraction")

        cbz_files = [os.path.join(self.cbz_extract_path, f)
                     for f in os.listdir(self.cbz_extract_path)
                     if f.lower().endswith('.cbz')]

        with Pool(processes=self.num_processes) as pool:
            temp_dirs = list(tqdm(
                pool.imap(self.extract_cbz_wrapper, cbz_files),
                total=len(cbz_files),
                desc="Extraction des tomes"
            ))

        temp_dirs = [d for d in temp_dirs if d is not None]

        for temp_dir in tqdm(temp_dirs, desc="Traitement des tomes"):
            if self.stop_processing:
                break

            try:
                image_files = self.get_filtered_images(temp_dir)
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    list(executor.map(self.process_single_image, image_files))
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        self.pbar.close()

        # Sauvegarde des métadonnées du dataset
        dataset_metadata = {
            'total_images': self.total_images_processed.value,
            'creation_date': os.path.getctime(self.dataset_dir),
            'structure_version': '2.0',
            'paths': {k: os.path.relpath(v, self.dataset_dir) for k, v in self.paths.items()}
        }

        with open(os.path.join(self.dataset_dir, 'dataset_info.json'), 'w') as f:
            json.dump(dataset_metadata, f, indent=4)

        print(f"\n✅ Traitement terminé !")
        print(f"Nombre total d'images traitées : {self.total_images_processed.value}")
        print(f"Dataset créé dans : {self.dataset_dir}")

        # Nettoyage final
        shutil.rmtree(self.cbz_extract_path, ignore_errors=True)
        shutil.rmtree(self.paths['temp'], ignore_errors=True)

        return self.total_images_processed.value

    def prepare_dataset(self) -> int:
        """
        Prépare le dataset complet en téléchargeant et en traitant les données.

        Retourne:
        - total_images_processed (int): Nombre total d'images traitées.
        """
        print("🚀 Démarrage du processus...")

        if not self.download_and_extract():
            print("❌ Échec de la préparation des fichiers")
            return 0

        print("\n🎨 Démarrage de la création du dataset...")
        return self.create_dataset()