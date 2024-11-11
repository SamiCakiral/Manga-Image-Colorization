import numpy as np
import cv2
import pandas as pd
import os
from datetime import datetime
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Tuple, List
import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from src.utils.config import config

class QualityMetrics:
    """
    Classe pour calculer et stocker les métriques de qualité de la colorisation.
    """
    def __init__(self, metrics_dir: str = os.path.join(config.dataset_dir, 'training', 'metrics')):
        self.metrics_dir = metrics_dir
        os.makedirs(metrics_dir, exist_ok=True)
        
        # Fichier CSV pour stocker toutes les métriques
        self.csv_path = os.path.join(metrics_dir, 'colorization_metrics.csv')
        
        # Initialiser les métriques torch
        self.psnr = PeakSignalNoiseRatio()
        self.ssim_metric = StructuralSimilarityIndexMeasure()
        
        # Créer ou charger le DataFrame des métriques
        if os.path.exists(self.csv_path):
            self.metrics_df = pd.read_csv(self.csv_path)
        else:
            self.metrics_df = pd.DataFrame(columns=[
                'image_id',
                'timestamp',
                'color_accuracy',
                'transition_coherence',
                'global_quality',
                'psnr',
                'ssim',
                'color_distribution_score',
                'edge_preservation',
                'local_consistency'
            ])

    def compute_all_metrics(self, pred: np.ndarray, target: np.ndarray, image_id: str) -> Dict[str, float]:
        """
        Calcule toutes les métriques pour une image.
        
        Arguments:
        - pred: Image prédite (colorisée) [0,1]
        - target: Image cible (vérité terrain) [0,1]
        - image_id: Identifiant unique de l'image
        
        Returns:
        - Dict contenant toutes les métriques
        """
        metrics = {
            'image_id': image_id,
            'timestamp': datetime.now().isoformat(),
            'color_accuracy': self.compute_color_accuracy(pred, target),
            'transition_coherence': self.compute_transition_coherence(pred, target),
            'global_quality': self.compute_global_quality(pred, target),
            'psnr': self.compute_psnr(pred, target),
            'ssim': self.compute_ssim(pred, target),
            'color_distribution_score': self.compute_color_distribution(pred, target),
            'edge_preservation': self.compute_edge_preservation(pred, target),
            'local_consistency': self.compute_local_consistency(pred, target)
        }
        
        # Ajouter au DataFrame et sauvegarder
        self.metrics_df = pd.concat([self.metrics_df, pd.DataFrame([metrics])], ignore_index=True)
        self.metrics_df.to_csv(self.csv_path, index=False)
        
        return metrics

    def compute_color_accuracy(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calcule la précision des couleurs en utilisant la distance LAB.
        """
        # Convertir en LAB
        pred_lab = cv2.cvtColor((pred * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        target_lab = cv2.cvtColor((target * 255).astype(np.uint8), cv2.COLOR_RGB2LAB)
        
        # Calculer la différence moyenne en LAB
        diff = np.mean(np.sqrt(np.sum((pred_lab - target_lab) ** 2, axis=2)))
        
        # Normaliser entre 0 et 1 (0 = parfait, 1 = mauvais)
        return 1 - min(diff / 100.0, 1.0)

    def compute_transition_coherence(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Évalue la cohérence des transitions de couleur.
        """
        # Calculer les gradients
        pred_grad = np.gradient(pred)
        target_grad = np.gradient(target)
        
        # Calculer la différence des gradients
        grad_diff = np.mean([np.mean(np.abs(pg - tg)) for pg, tg in zip(pred_grad, target_grad)])
        
        # Normaliser
        return 1 - min(grad_diff * 5, 1.0)  # * 5 pour amplifier les différences

    def compute_global_quality(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calcule un score global de qualité combinant plusieurs métriques.
        """
        color_acc = self.compute_color_accuracy(pred, target)
        trans_coh = self.compute_transition_coherence(pred, target)
        ssim_score = self.compute_ssim(pred, target)
        
        # Moyenne pondérée
        weights = [0.4, 0.3, 0.3]  # Importance relative de chaque métrique
        return np.average([color_acc, trans_coh, ssim_score], weights=weights)

    def compute_psnr(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calcule le Peak Signal-to-Noise Ratio.
        """
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
        target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
        return self.psnr(pred_t, target_t).item()

    def compute_ssim(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Calcule le Structural Similarity Index.
        """
        pred_t = torch.from_numpy(pred).permute(2, 0, 1).unsqueeze(0)
        target_t = torch.from_numpy(target).permute(2, 0, 1).unsqueeze(0)
        return self.ssim_metric(pred_t, target_t).item()

    def compute_color_distribution(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Compare les distributions de couleurs entre les images.
        """
        def get_color_hist(img):
            hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 1, 0, 1, 0, 1])
            return cv2.normalize(hist, hist).flatten()
        
        pred_hist = get_color_hist((pred * 255).astype(np.uint8))
        target_hist = get_color_hist((target * 255).astype(np.uint8))
        
        return 1 - cv2.compareHist(pred_hist, target_hist, cv2.HISTCMP_BHATTACHARYYA)

    def compute_edge_preservation(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Évalue la préservation des contours.
        """
        def get_edges(img):
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
            return cv2.Canny(gray, 100, 200)
        
        pred_edges = get_edges(pred)
        target_edges = get_edges(target)
        
        return np.mean(pred_edges == target_edges)

    def compute_local_consistency(self, pred: np.ndarray, target: np.ndarray) -> float:
        """
        Évalue la cohérence locale des couleurs.
        """
        window_size = 8
        stride = 4
        h, w = pred.shape[:2]
        scores = []
        
        for i in range(0, h-window_size, stride):
            for j in range(0, w-window_size, stride):
                pred_patch = pred[i:i+window_size, j:j+window_size]
                target_patch = target[i:i+window_size, j:j+window_size]
                
                # Calculer la corrélation locale
                correlation = np.corrcoef(pred_patch.reshape(-1), target_patch.reshape(-1))[0,1]
                scores.append(correlation)
        
        return np.mean(scores)

    def generate_metrics_report(self, output_path: str = None) -> pd.DataFrame:
        """
        Génère un rapport statistique des métriques.
        """
        if output_path is None:
            output_path = os.path.join(self.metrics_dir, 'metrics_report.csv')
            
        # Calculer les statistiques
        stats = self.metrics_df.describe()
        
        # Ajouter des métriques supplémentaires
        stats.loc['median'] = self.metrics_df.median()
        stats.loc['mode'] = self.metrics_df.mode().iloc[0]
        
        # Sauvegarder le rapport
        stats.to_csv(output_path)
        
        return stats

    def plot_metrics_evolution(self, metric_name: str, save_path: str = None):
        """
        Trace l'évolution d'une métrique au fil du temps.
        """
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics_df[metric_name])
        plt.title(f'Evolution of {metric_name}')
        plt.xlabel('Image Index')
        plt.ylabel(metric_name)
        
        if save_path:
            plt.savefig(save_path)
        plt.close() 